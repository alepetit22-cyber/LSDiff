import json
import os
import sys
import time
import torch
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from config import config, load_config
from dataset import PatientDataset, collate_fn, compute_latent_scale
from autoencoder import VAE1D
from diffusion_engine import DiffusionEngine, FlowMatchingScheduler, CFGWrapper
from loss_functions import FlowMatchingLoss
from metrics import DatasetEvaluator

# Configuration du logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def main():    
    ###################################################
    # CONFIGURATION DU PARSER
    ###################################################
    # Configuration du parser
    parser = argparse.ArgumentParser(description="Entraînement des VAE de LSDiff")
    parser.add_argument("--config", type=str, default=None, help="Chemin du fichier config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    num_cpu = multiprocessing.cpu_count()
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")

    # Configuration du mode conditionnel
    has_event = len(config.dataset.event_code_index) > 0 if config.dataset.event_code_index is not None else False
    if has_event:
        logger.info("Mode guidage par les événements")
    else:
        logger.info("Mode sans guidage par les événements")

    checkpoint_dir = os.path.dirname(config.autoencoder.scaler_path)  # ex: "results/vae_nh_8/checkpoints"
    exp_dir = os.path.dirname(checkpoint_dir)
    
    ###################################################
    # CONFIGURATION DU DATASET
    ###################################################
    with open(config.dataset.json_path, 'r', encoding='utf-8') as f:
        all_patients_raw = json.load(f)

    # Récupération des identifiants
    all_patient_ids = [p.get("patient_id", f"unknown_{i}") for i, p in enumerate(all_patients_raw)]

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    unique_patients = list(set(all_patient_ids))
    random.shuffle(unique_patients)

    n = len(unique_patients)
    train_ratio = 0.8
    val_ratio = 0.1

    train_ids = set(unique_patients[:int(train_ratio * n)])
    val_ids   = set(unique_patients[int(train_ratio * n):int((train_ratio + val_ratio) * n)])
    test_ids  = set(unique_patients[int((train_ratio + val_ratio) * n):])

    # Filtrage des données brutes pour chaque ensemble
    train_raw = [p for p in all_patients_raw if p.get("patient_id", "unknown") in train_ids]
    val_raw   = [p for p in all_patients_raw if p.get("patient_id", "unknown") in val_ids]
    test_raw  = [p for p in all_patients_raw if p.get("patient_id", "unknown") in test_ids]

    # Création des datasets
    common_params = {
        "target_len": config.dataset.target_len,
        "hist_len": config.dataset.hist_len,
        "continuous_indices": config.dataset.continuous_indices,
        "discrete_indices": config.dataset.discrete_indices,
        "categorical_indices": config.dataset.categorical_indices,
        "normalization": config.dataset.normalization,
        "scaler_path": config.autoencoder.scaler_path,
        "meta_config": config.dataset.meta_config,
        "cat_embed_dim": config.dataset.cat_embed_dim,
        "cat_seed": config.dataset.cat_seed,
        "event_code_index": config.dataset.event_code_index,
    }

    train_dataset = PatientDataset(
        raw_data=train_raw,
        fit_stats=True,
        **common_params
    )

    val_dataset = PatientDataset(
        raw_data=val_raw,
        fit_stats=False,
        **common_params
    )

    test_dataset = PatientDataset(
        raw_data=test_raw,
        fit_stats=False,
        **common_params
    )
    test_dataset = torch.utils.data.Subset(test_dataset, range(1000))


    # Création des DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=True,
        num_workers=max(1, num_cpu//8),
        collate_fn=collate_fn, 
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size, 
        shuffle=True,
        num_workers=max(1, num_cpu//8),
        collate_fn=collate_fn, 
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size, 
        shuffle=True,
        num_workers=max(1, num_cpu//8),
        collate_fn=collate_fn, 
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )

    ###################################################
    # CONFIGURATION DES VAEs
    ###################################################
    cat_vocab_sizes = [
        len(train_dataset.cat_vocabs[col]) for col in config.dataset.categorical_indices
    ] if config.dataset.cat_mode == "embedded" else []

    vae = VAE1D(
        num_input_channels=config.dataset.num_effective_float_channels,
        num_continuous=train_dataset.num_continuous,
        num_discrete=train_dataset.num_discrete,
        latent_channel=config.autoencoder.latent_channel,
        stride=config.autoencoder.stride,
        seq_len=config.autoencoder.seq_len,
        enc_hidden_dims=config.autoencoder.enc_hidden_dims,
        dec_hidden_dims=config.autoencoder.dec_hidden_dims,
        num_groups=config.autoencoder.num_groups,
        dropout=config.autoencoder.dropout,
        num_heads=config.autoencoder.num_heads,
        kernel_size_stride=config.autoencoder.kernel_size_stride,
        kernel_size_res=config.autoencoder.kernel_size_res,
        padding=config.autoencoder.padding,
        logvar_clip_min=config.autoencoder.logvar_clip_min,
        logvar_clip_max=config.autoencoder.logvar_clip_max,
        cat_mode=config.dataset.cat_mode,
        cat_embed_dim=config.dataset.cat_embed_dim,
        cat_vocab_sizes=cat_vocab_sizes
    ).to(device)
    vae.load_state_dict(
        torch.load(
            config.autoencoder.best_model_path,
            map_location=device,
            weights_only=True
        )
    )
    vae.eval()

    hist_vae = VAE1D(
        num_input_channels=config.dataset.num_effective_float_channels,
        num_continuous=train_dataset.num_continuous,
        num_discrete=train_dataset.num_discrete,
        latent_channel=config.history_autoencoder.latent_channel,
        stride=config.history_autoencoder.stride,
        seq_len=config.history_autoencoder.seq_len,
        enc_hidden_dims=config.history_autoencoder.enc_hidden_dims,
        dec_hidden_dims=config.history_autoencoder.dec_hidden_dims,
        num_groups=config.history_autoencoder.num_groups,
        dropout=config.history_autoencoder.dropout,
        num_heads=config.history_autoencoder.num_heads,
        kernel_size_stride=config.history_autoencoder.kernel_size_stride,
        kernel_size_res=config.history_autoencoder.kernel_size_res,
        padding=config.history_autoencoder.padding,
        logvar_clip_min=config.history_autoencoder.logvar_clip_min,
        logvar_clip_max=config.history_autoencoder.logvar_clip_max,
        cat_mode=config.dataset.cat_mode,
        cat_embed_dim=config.dataset.cat_embed_dim,
        cat_vocab_sizes=cat_vocab_sizes
    ).to(device)
    hist_vae.load_state_dict(
        torch.load(
            config.history_autoencoder.best_model_path,
            map_location=device,
            weights_only=True
        )
    )
    hist_vae.eval()

    # Échelles des Latents
    latent_scale = compute_latent_scale(
        vae,
        train_loader,
        device,
        config.diffusion.latent_scale_path,
        is_history=False,
        max_batches=1000
    )
    hist_latent_scale = compute_latent_scale(
        hist_vae,
        train_loader,
        device,
        config.diffusion.hist_latent_scale_path,
        is_history=True,
        max_batches=1000
    )
    
    # Détermination de la forme latente
    with torch.no_grad():
        dummy_f = torch.zeros(1, config.dataset.num_effective_float_channels, config.autoencoder.seq_len).to(device)
        dummy_c = torch.zeros(1, len(cat_vocab_sizes), config.autoencoder.seq_len, dtype=torch.long).to(device) if cat_vocab_sizes else None
        mu, _ = vae.encode(dummy_f, dummy_c)
        latent_shape = (mu.shape[1], mu.shape[2])

    ###################################################
    # CONFIGURATION DU DiT
    ###################################################
    engine_config = argparse.Namespace(
        embed_dim=config.diffusion.embed_dim,
        latent_channel=config.autoencoder.latent_channel,
        latent_channel_hist=config.history_autoencoder.latent_channel,
        num_layers=config.diffusion.num_layers,
        num_classes=config.diffusion.num_classes,
        num_heads=config.diffusion.num_heads,
        ff_mult=config.diffusion.ff_mult,
        dropout=config.diffusion.dropout,
        vae_stride=config.diffusion.vae_stride,
        latent_seq_len=latent_shape[1],
        meta_config=config.dataset.meta_config,
        has_event=has_event
    )
    
    model = DiffusionEngine(engine_config).to(device)
    if torch.cuda.is_available():
        model = torch.compile(model)
    
    # EMA
    ema_model = torch.optim.swa_utils.AveragedModel(
        model, 
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
    )

    # Optimiseur
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.lr_diffusion,
        weight_decay=1e-4
        )
        
    # Scheduler
    noise_scheduler = FlowMatchingScheduler(
        num_inference_steps=config.diffusion.num_inference_steps
        )
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.training.lr_diffusion * 1.5,
        steps_per_epoch=len(train_loader),      # A chaque batch
        epochs=config.training.epochs_diffusion,
        pct_start=0.05,                         # 5% d'époques pour le Warmup
        anneal_strategy='cos',
        div_factor=25.0,                        # max_lr / 25
        final_div_factor=10000.0                # LR final minuscule
    )
    
    # Scaler
    scaler = torch.amp.GradScaler(device='cuda') if torch.cuda.is_available() else None

    # Loss
    loss_function = FlowMatchingLoss()

    logger.info(f"Nombre de séquences d'entraînement : {len(train_dataset)}")
    logger.info(f"Nombre de séquences de validation : {len(val_dataset)}")
    logger.info(f"Nombre de séquences de test : {len(test_dataset)}")

    ###################################################
    # ENTRAINEMENT
    ###################################################    
    logger.info("=" * 60)
    logger.info("[Début de l'entraînement]")
    debut_train = time.time()
    logger.info(f"Heure de début de l'entraînement : {time.strftime('%Y-%m-%d %H:%M:%S')}")

    best_val_loss = float('inf')

    for epoch in range(1, config.training.epochs_diffusion + 1):
        model.train()
        train_loss, train_mse, train_cos, train_mse_accum, train_cos_accum = 0.0, 0.0, 0.0, 0.0, 0.0
        
        for batch_idx, (x_float, hist_float, meta_dict, x_cat, hist_cat, event_seq) in enumerate(train_loader):
            real_vitals_float = x_float.to(device, non_blocking=True)
            hist_feat_float   = hist_float.to(device, non_blocking=True)
            real_vitals_cat   = x_cat.to(device, non_blocking=True)
            hist_feat_cat     = hist_cat.to(device, non_blocking=True)
            meta_dict         = {k: v.to(device, non_blocking=True) for k, v in meta_dict.items()}
            
            # Conditionnement si has_event
            cond_idx = event_seq.to(device, non_blocking=True) if has_event else None
            
            # CFG Dropout
            drop_meta = torch.rand(real_vitals_float.shape[0], device=device) < config.diffusion.cfg_dropout
            drop_hist = torch.rand(real_vitals_float.shape[0], device=device) < config.diffusion.cfg_dropout
            force_uncond_event = torch.rand(real_vitals_float.shape[0], device=device) < config.diffusion.cfg_dropout
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.no_grad():
                mu_main, _ = vae.encode(real_vitals_float, real_vitals_cat)
                x_0 = mu_main * latent_scale
                
                mu_hist, _ = hist_vae.encode(hist_feat_float, hist_feat_cat)
                z_hist = mu_hist * hist_latent_scale

            t = FlowMatchingScheduler.sample_logit_normal_t(x_0.shape[0], device)
            x_noisy, target = noise_scheduler.add_noise(x_0, t)
            
            # Utilisation d'Autocast pour le Forward Pass
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=torch.cuda.is_available()):
                v_pred = model(
                    x_noisy, t.view(-1, 1), 
                    cond_seq=cond_idx,
                    meta_dict=meta_dict,
                    z_hist=z_hist,
                    drop_meta=drop_meta,
                    drop_hist=drop_hist,
                    force_uncond_event=force_uncond_event if has_event else None
                )
            
                train_mse, train_cos = loss_function(v_pred, target)
                loss = train_mse
            
            # Backward Pass avec le Scaler
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer) # Unscale pour le clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            lr_scheduler.step()
            ema_model.update_parameters(model)

            # Accumulation pour les logs
            train_loss += loss.item() 
            train_mse_accum += train_mse.item()
            train_cos_accum += train_cos.item()
            
        # VALIDATION
        model.eval()
        val_loss, val_mse, val_cos, val_mse_accum, val_cos_accum = 0.0, 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for x_float, hist_float, meta_batch, x_cat, hist_cat, event_seq in val_loader:
                real_vitals_float = x_float.to(device, non_blocking=True)
                hist_feat_float   = hist_float.to(device, non_blocking=True)
                real_vitals_cat   = x_cat.to(device, non_blocking=True)
                hist_feat_cat     = hist_cat.to(device, non_blocking=True)
                meta_batch_dev    = {k: v.to(device, non_blocking=True) for k, v in meta_batch.items()}
                
                # Conditionnement si has_event
                cond_idx = event_seq.to(device, non_blocking=True) if has_event else None
                
                # Pas de dropout
                drop_meta = torch.zeros(real_vitals_float.shape[0], dtype=torch.bool, device=device)
                drop_hist = torch.zeros(real_vitals_float.shape[0], dtype=torch.bool, device=device)
                force_uncond_event = torch.zeros(real_vitals_float.shape[0], dtype=torch.bool, device=device)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.no_grad():
                    mu_main, _ = vae.encode(real_vitals_float, real_vitals_cat)
                    x_0 = mu_main * latent_scale
                    
                    mu_hist, _ = hist_vae.encode(hist_feat_float, hist_feat_cat)
                    z_hist = mu_hist * hist_latent_scale
                
                t = FlowMatchingScheduler.sample_logit_normal_t(x_0.shape[0], device)
                x_noisy, target = noise_scheduler.add_noise(x_0, t)

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=torch.cuda.is_available()):
                    v_pred = model(
                        x_noisy, t.view(-1, 1), 
                        cond_seq=cond_idx, 
                        meta_dict=meta_batch_dev, 
                        z_hist=z_hist
                    )
                    val_mse, val_cos = loss_function(v_pred, target)
                    batch_val_loss = val_mse

                val_loss += batch_val_loss.item()
                val_mse_accum += val_mse.item()
                val_cos_accum += val_cos.item()

        train_loss /= len(train_loader)
        train_mse_accum /= len(train_loader)
        train_cos_accum /= len(train_loader)

        val_loss /= len(val_loader)
        val_mse_accum /= len(val_loader)
        val_cos_accum /= len(val_loader)
        
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:03d} | Loss: {train_loss:.4f}/{val_loss:.4f} " 
                        f"| MSE : {train_mse_accum:.4f}/{val_mse_accum:.4f} "
                        f"| COS : {train_cos_accum:.4f}/{val_cos_accum:.4f} "
                        f"| LR: {lr_scheduler.get_last_lr()[0]:.6f}")
            cfg_model = CFGWrapper(ema_model, cfg_scale=config.diffusion.cfg_scale)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ema_model.state_dict(), config.diffusion.best_model_path)
            logger.info(f"Meilleur modèle sauvegardé. Val Loss: {best_val_loss:.6f}")

    torch.save(ema_model.state_dict(), config.diffusion.checkpoint_path)
    fin_train = time.time()
    logger.info(f"[Entraînement terminé] Meilleure Val Loss: {best_val_loss:.6f}")
    logger.info(f"Heure de fin de l'entraînement : {time.strftime('%Y-%m-%d %H:%M:%S')}")


    ###################################################
    # INFERENCE
    ###################################################
    logger.info("=" * 60)
    logger.info("[Début de l'inférence]")
    debut_inf = time.time()
    logger.info(f"Heure de début de l'inférence : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    ema_model.eval()
    model_for_cfg = ema_model.module if hasattr(ema_model, 'module') else ema_model
    cfg_model = CFGWrapper(model_for_cfg, cfg_scale=config.diffusion.cfg_scale)
    
    real_vitals_list = []
    fake_vitals_list = []

    with torch.no_grad():
        for x_float, hist_float, meta_batch, x_cat, hist_cat, event_seq in tqdm(test_loader, desc="Inférence Test"):
            real_vitals_float = x_float.to(device, non_blocking=True)
            hist_feat_float   = hist_float.to(device, non_blocking=True)
            real_vitals_cat   = x_cat.to(device, non_blocking=True)
            hist_feat_cat     = hist_cat.to(device, non_blocking=True)
            meta_batch_dev    = {k: v.to(device, non_blocking=True) for k, v in meta_batch.items()}
            
            b = real_vitals_float.shape[0]

            # Préparation des conditionnements latents
            mu_hist, _ = hist_vae.encode(hist_feat_float, hist_feat_cat)
            z_hist = mu_hist * hist_latent_scale

            # Conditionnement si has_event
            cond_idx = event_seq.to(device, non_blocking=True) if has_event else None
                
            # Échantillonnage via le Flow Matching Scheduler
            recon_samples = []
            for _ in range(config.diffusion.num_samples):
                x_gen = noise_scheduler.sample(
                    model_wrapper=cfg_model,
                    shape=(b, *latent_shape),
                    device=device,
                    latent_scale=latent_scale,
                    cond_seq=cond_idx,
                    z_hist=z_hist,
                    meta_dict=meta_batch_dev
                )
                
                # Décodage par le VAE principal
                recon_float, recon_cat_logits = vae.decode(x_gen)
                if config.dataset.cat_mode == "embedded" and recon_cat_logits is not None:
                    cat_preds = torch.stack([logits.argmax(dim=1) for logits in recon_cat_logits], dim=1)
                    recon_full = torch.cat([recon_float, cat_preds.float()], dim=1)
                else:
                    recon_full = recon_float

                recon_samples.append(recon_full.cpu().numpy())
                
            recon_out = np.median(np.stack(recon_samples, axis=0), axis=0)

            if config.dataset.cat_mode == "embedded":
                real_full = torch.cat([real_vitals_float, real_vitals_cat.float()], dim=1)
                real_vitals_list.append(real_full.cpu().numpy())
            else:
                real_vitals_list.append(real_vitals_float.cpu().numpy())

            fake_vitals_list.append(recon_out)

    # Concaténation globale de tous les lots
    real_np = np.concatenate(real_vitals_list, axis=0)  # [Total_B, C_total, L]
    gen_np = np.concatenate(fake_vitals_list, axis=0)  # [Total_B, C_total, L]

    # Permutation des axes
    real_np = np.transpose(real_np, (0, 2, 1))
    gen_np = np.transpose(gen_np, (0, 2, 1))

    # Dénormalisation
    if config.dataset.cat_mode == "duplicated":
        real_denorm = train_dataset.denormalize(real_np)
        fake_denorm = train_dataset.denormalize(gen_np)

        if train_dataset.num_categorical > 0:
            real_cat_agg = train_dataset.aggregate_cat_duplicates(real_denorm)
            fake_cat_agg = train_dataset.aggregate_cat_duplicates(fake_denorm)
            
            real_data = np.concatenate([real_denorm[..., :train_dataset.cat_map_start_idx], real_cat_agg], axis=-1)
            gen_data  = np.concatenate([fake_denorm[..., :train_dataset.cat_map_start_idx], fake_cat_agg], axis=-1)
        else:
            real_data = real_denorm
            gen_data  = fake_denorm

    elif config.dataset.cat_mode == "embedded":
        # En mode embedded, le scaler s'applique uniquement sur les canaux flottants
        num_float_ch = train_dataset.num_effective_float_channels
        
        real_float_denorm = train_dataset.denormalize(real_np[..., :num_float_ch])
        fake_float_denorm = train_dataset.denormalize(gen_np[..., :num_float_ch])
        
        real_cat = real_np[..., num_float_ch:]
        fake_cat = gen_np[..., num_float_ch:]
        
        real_data = np.concatenate([real_float_denorm, real_cat], axis=-1)
        gen_data  = np.concatenate([fake_float_denorm, fake_cat], axis=-1)

    fin_inf = time.time()
    logger.info("[Inférence terminée]")
    logger.info(f"Heure de fin de l'inférence : {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Clip de la SpO2
    gen_data[:,:,5] = np.clip(gen_data[:,:,5], a_min=0.0, a_max=100.0)
    # Arrondi des événements
    gen_data[:,:,7] = np.round(gen_data[:, :, 7])

    # Sauvegarde finale des fichiers NumPy
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)
    real_path = f"{exp_dir}/checkpoints/real_data_diffusion.npy"
    gen_path = f"{exp_dir}/checkpoints/gen_data_diffusion.npy"

    np.save(real_path, real_data)
    np.save(gen_path, gen_data)


    logger.info("Sauvegarde des données d'inférence Terminée")
    logger.info(f"Fichier Réel   : {real_path} (Shape: {real_data.shape})")
    logger.info(f"Fichier Généré : {gen_path} (Shape: {gen_data.shape})")

    # Calcul du temps
    durée_train = fin_train - debut_train
    durée_inf = fin_inf - debut_inf

    # Formatage en minutes:secondes
    m_train, s_train = divmod(durée_train, 60)
    m_inf, s_inf = divmod(durée_inf, 60)

    logger.info("=" * 60)
    logger.info("[Evaluation de la durée d'entrainement et d'inférece]")
    logger.info(f"Temps d'entraînement : {int(m_train)} min {int(s_train)} s (Total: {durée_train:.2f} secondes)")
    logger.info(f"Temps d'inférence    : {int(m_inf)} min {int(s_inf)} s (Total: {durée_inf:.2f} secondes)")
    logger.info(f"Temps total du run   : {int((durée_train + durée_inf) // 60)} min {int((durée_train + durée_inf) % 60)} s")
    logger.info("=" * 60)
      
    ###################################################
    # TEST
    ###################################################
    evaluator = DatasetEvaluator(
        real_data, 
        gen_data,
        col_names=["FC", "PAS", "PAM", "PAD", "Temp", "SpO2", "FR", "event_code"],
        path_dir=f"{exp_dir}/checkpoints/diffusion/"
    )

    evaluator.run_full_analysis()
    
if __name__ == "__main__":
    main()
