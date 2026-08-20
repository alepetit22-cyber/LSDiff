import json
import os
import sys
import time
import torch
import random
import logging
import argparse
import numpy as np
import multiprocessing
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from config import config, load_config
from dataset import PatientDataset, collate_fn
from autoencoder import VAE1D
from loss_functions import HybridVAELoss
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
    # Paramètres selon le mode
    parser = argparse.ArgumentParser(description="Entraînement des VAE de LSDiff")
    parser.add_argument("--config", type=str, default=None, help="Chemin du fichier config.yaml")
    parser.add_argument("--mode", type=str, choices=["main", "history"], required=True, help="Mode d'entraînement")
    args = parser.parse_args()
    if args.mode == "main":
        config = load_config(args.config)
        vae_config = config.autoencoder
        scaler_path = vae_config.scaler_path
        target_len = vae_config.seq_len
    else:
        config = load_config(args.config)
        vae_config = config.history_autoencoder
        scaler_path = vae_config.hist_scaler_path
        target_len = vae_config.seq_len

    num_cpu = multiprocessing.cpu_count()
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")

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
        "scaler_path": scaler_path,
        "meta_config": config.dataset.meta_config,
        "cat_embed_dim": config.dataset.cat_embed_dim,
        "cat_mode": config.dataset.cat_mode,
        "cat_seed": config.dataset.cat_seed,
        "mode": args.mode,  # "main", "history" ou "DiT"
        "event_code_index": config.dataset.event_code_index
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
    # CONFIGURATION DU VAE
    ###################################################
    cat_vocab_sizes = [
        len(train_dataset.cat_vocabs[col]) for col in config.dataset.categorical_indices
    ] if config.dataset.cat_mode == "embedded" else []

    model = VAE1D(
        num_input_channels=config.dataset.num_effective_float_channels,
        num_continuous=train_dataset.num_continuous,
        num_discrete=train_dataset.num_discrete,
        latent_channel=vae_config.latent_channel,
        stride=vae_config.stride,
        seq_len=target_len,
        enc_hidden_dims=vae_config.enc_hidden_dims,
        dec_hidden_dims=vae_config.dec_hidden_dims,
        num_groups=vae_config.num_groups,
        dropout=vae_config.dropout,
        num_heads=vae_config.num_heads,
        kernel_size_stride=vae_config.kernel_size_stride,
        kernel_size_res=vae_config.kernel_size_res,
        padding=vae_config.padding,
        logvar_clip_min=vae_config.logvar_clip_min,
        logvar_clip_max=vae_config.logvar_clip_max,
        cat_mode=config.dataset.cat_mode,
        cat_embed_dim=config.dataset.cat_embed_dim,
        cat_vocab_sizes=cat_vocab_sizes
    )
    model = model.to(device)

    # Optimiseur
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.lr_vae,
        weight_decay=1e-5
    )
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.epochs_vae,
        eta_min=1e-6
    )

    # Loss
    loss_function = HybridVAELoss(
        spectral_weight=vae_config.spectral_weight,
        cce_weight=0.1,
        kld_weight=0.1
    )

    logger.info(f"Nombre de séquences d'entraînement : {len(train_dataset)}")
    logger.info(f"Nombre de séquences de validation : {len(val_dataset)}")
    logger.info(f"Nombre de séquences de test : {len(test_dataset)}")

    ###################################################
    # ENTRAINEMENT
    ###################################################
    logger.info("=" * 60)
    logger.info("[Début de l'entraînement]")
    debut_train = time.time()
    logger.info(f"Heure de début de l'entraînement ({args.mode}) : {time.strftime('%Y-%m-%d %H:%M:%S')}")

    best_val_mse = float('inf')
    
    for epoch in range(1, config.training.epochs_vae + 1):
        # TRAIN
        model.train()
        train_metrics = {"mse": 0, "cce": 0,  "kld": 0, "spec": 0}
        
        for batch_idx, (x_float, hist_float, meta_dict, x_cat, hist_cat, _) in enumerate(train_loader):
            optimizer.zero_grad()
            
            if args.mode == "main":
                inputs_float, inputs_cat = x_float.to(device), x_cat.to(device)
            else:
                inputs_float, inputs_cat = hist_float.to(device), hist_cat.to(device)

            recon_float, recon_cat_logits, mu, logvar = model(inputs_float, inputs_cat)
            
            _ , mse, cce, kld, spec = loss_function(
                mu,
                logvar,
                recon_float,
                recon_cat_logits,
                inputs_float,
                target_cat=inputs_cat if config.dataset.cat_mode == "embedded" else None
            )
            
            loss = mse
            (loss).backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_metrics["mse"]  += mse.item()
            train_metrics["cce"]  += cce.item()
            train_metrics["kld"]  += kld.item()
            train_metrics["spec"] += spec.item()

        # VALIDATION
        model.eval()
        val_metrics = {"mse": 0, "cce": 0, "kld": 0, "spec": 0}
        with torch.no_grad():
            for x_float, hist_float, meta_dict, x_cat, hist_cat, _ in val_loader:
            
                if args.mode == "main":
                    inputs_float, inputs_cat = x_float.to(device), x_cat.to(device)
                else:
                    inputs_float, inputs_cat = hist_float.to(device), hist_cat.to(device)
                    
                recon_float, recon_cat_logits, mu, logvar = model(inputs_float, inputs_cat)

                loss, mse, cce, kld, spec = loss_function(
                    mu,
                    logvar,
                    recon_float,
                    recon_cat_logits,
                    inputs_float,
                    target_cat=inputs_cat if config.dataset.cat_mode == "embedded" else None
                )
                
                val_metrics["mse"]  += mse.item()
                val_metrics["cce"]  += cce.item()
                val_metrics["kld"]  += kld.item()
                val_metrics["spec"] += spec.item()

        # Moyennes
        for k in train_metrics: train_metrics[k] /= len(train_loader)
        for k in val_metrics: val_metrics[k] /= len(val_loader)
        
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:04d} "
                  f"| MSE: {train_metrics['mse']:.4f}/{val_metrics['mse']:.4f} " 
                  f"| CCE: {train_metrics['cce']:.0f}/{val_metrics['cce']:.0f} "
                  f"| KLD: {train_metrics['kld']:.0f}/{val_metrics['kld']:.0f} "
                  f"| SPEC: {train_metrics['spec']:.4f}/{val_metrics['spec']:.6f} ")

        # Sauvegarde
        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), vae_config.best_model_path)
            logger.info(f"Nouveau meilleur modèle sauvegardé avec Val MSE: {best_val_mse:.6f}")
            
    torch.save(model.state_dict(), vae_config.checkpoint_path)
    fin_train = time.time()
    logger.info(f"[Entraînement terminé] Meilleure Val Loss: {best_val_mse:.6f}")
    logger.info(f"Heure de fin de l'entraînement : {time.strftime('%Y-%m-%d %H:%M:%S')}")

    ###################################################
    # INFERENCE
    ###################################################
    logger.info("=" * 60)
    logger.info("[Début de l'inférence]")
    debut_inf = time.time()
    logger.info(f"Heure de début de l'inférence : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    model.eval()
    real_data_list = []
    gen_data_list = []

    logger.info(f"Extraction des données en cours ({args.mode})")

    with torch.no_grad():
        for x_float, hist_float, _, x_cat, hist_cat, _ in test_loader:
            if args.mode == "main":
                inputs_float, inputs_cat = x_float.to(device), x_cat.to(device)
            else:
                inputs_float, inputs_cat = hist_float.to(device), hist_cat.to(device)
                
            recon_float, recon_cat_logits = model.decode(model.encode(inputs_float, inputs_cat)[0])

            if config.dataset.cat_mode == "embedded" and recon_cat_logits is not None:
                # Prédiction Argmax pour les catégories
                cat_preds = torch.stack([logits.argmax(dim=1) for logits in recon_cat_logits], dim=1) # [B, Num_Cat, L]
                inputs_full = torch.cat([inputs_float, inputs_cat], dim=1)
                recon_full  = torch.cat([recon_float, cat_preds.float()], dim=1)
            else:
                inputs_full = inputs_float
                recon_full  = recon_float

            # Stockage en NumPy (conversion CPU + détachement du graphe)
            real_data_list.append(inputs_full.cpu().numpy())
            gen_data_list.append(recon_full.cpu().numpy())

    # Concaténation des batchs
    real_np = np.concatenate(real_data_list, axis=0)
    gen_np = np.concatenate(gen_data_list, axis=0)

    # Dénormalisation
    real_denorm = train_dataset.denormalize(np.transpose(real_np, (0, 2, 1)))
    recon_denorm = train_dataset.denormalize(np.transpose(gen_np, (0, 2, 1)))

    # Aggrégation des variables catégorielles
    if config.dataset.cat_mode == "duplicated" and train_dataset.num_categorical > 0:
        real_cat_agg = train_dataset.aggregate_cat_duplicates(real_denorm)
        recon_cat_agg = train_dataset.aggregate_cat_duplicates(recon_denorm)
        
        real_data = np.concatenate([real_denorm[..., :train_dataset.cat_map_start_idx], real_cat_agg], axis=-1)
        gen_data = np.concatenate([recon_denorm[..., :train_dataset.cat_map_start_idx], recon_cat_agg], axis=-1)
    else:
        real_data = real_denorm
        gen_data = recon_denorm

    fin_inf = time.time()
    logger.info("[Inférence terminée]")
    logger.info(f"Heure de fin de l'inférence : {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Clip de la SpO2
    gen_data[:,:,5] = np.clip(gen_data[:,:,5], a_min=0.0, a_max=100.0)
    # Arrondi des événements
    gen_data[:,:,7] = np.round(gen_data[:, :, 7])

    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)
    real_path = f"{exp_dir}/checkpoints/real_data_{args.mode}.npy"
    gen_path = f"{exp_dir}/checkpoints/gen_data_{args.mode}.npy"

    np.save(real_path, real_data)
    np.save(gen_path, gen_data)

    logger.info(f"Sauvegarde terminée avec succès")
    logger.info(f"Fichier Réel    : {real_path} (Shape: {real_data.shape})")
    logger.info(f"Fichier Généré  : {gen_path} (Shape: {gen_data.shape})")
    
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
        path_dir=f"{exp_dir}/checkpoints/{args.mode}/"
    )

    evaluator.run_full_analysis()

if __name__ == "__main__":
    main()
