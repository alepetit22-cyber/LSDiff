import json
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from diff_models import diff_CSDI
from torch.utils.data import Dataset, DataLoader, RandomSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder

def create_windows(data_raw, history_length, pred_length):
    """
    Découpe les données brutes des patients en fenêtres glissantes de taille fixe,
    avec un padding au début pour que l'historique puisse démarrer à -history_length.
    """
    window_size = history_length + pred_length
    windowed_data = []

    for patient in data_raw:
        # Conversion en array numpy
        donnees_np = np.array(patient['donnees'])
        longueur_totale = donnees_np.shape[0]
        num_features = donnees_np.shape[1]
                
        # Boucle de génération des fenêtres glissantes
        for start_idx in range(-history_length, longueur_totale - window_size + 1):
            
            # Initialisation d'une matrice vide pour la fenêtre
            window = np.zeros((window_size, num_features), dtype=donnees_np.dtype)
            
            if start_idx < 0:
                nb_pads = abs(start_idx)
                fin_idx = min(start_idx + window_size, longueur_totale)
                partie_reelle = donnees_np[0 : fin_idx, :]
                
                # Insertion des vraies données à la suite du padding
                window[nb_pads : nb_pads + len(partie_reelle), :] = partie_reelle
            else:
                fin_idx = min(start_idx + window_size, longueur_totale)
                partie_reelle = donnees_np[start_idx : fin_idx, :]
                
                window[0 : len(partie_reelle), :] = partie_reelle
                
            patient_virtuel = {
                "patient_id": patient["patient_id"],
                "metadata": patient["metadata"],
                "donnees": window
            }
            windowed_data.append(patient_virtuel)
            
    return windowed_data

class CustomDataset(Dataset):
    def __init__(self, windowed_data, history_length, pred_length, 
                 scaler_cont, enc_cat, enc_class, enc_surg):
        
        self.data = windowed_data
        self.history_length = history_length
        self.pred_length = pred_length
        self.window_size = history_length + pred_length
        
        # Les scalers et encodeurs sont désormais obligatoirement fournis (ajustés en amont)
        self.scaler_cont = scaler_cont
        self.enc_cat = enc_cat
        self.enc_class = enc_class
        self.enc_surg = enc_surg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient = self.data[idx]
        
        # Les données sont déjà fenêtrées et paddées !
        window = patient["donnees"]
        
        tp = window[:, 0]
        cont_data = self.scaler_cont.transform(window[:, 1:8])
        
        # Application de l'encodeur sur les événements
        cat_raw = window[:, 8]
        known_mask = np.isin(cat_raw, self.enc_cat.classes_)
        cat_data = np.zeros_like(cat_raw, dtype=int)
        cat_data[known_mask] = self.enc_cat.transform(cat_raw[known_mask])

        # Création des masques pour CSDI
        obs_mask = np.zeros(self.window_size)
        obs_mask[:self.history_length] = 1.0 # 1 sur l'historique
        
        gt_mask = np.zeros(self.window_size)
        gt_mask[self.history_length:] = 1.0  # 1 sur la prédiction

        # Métadonnées
        pc = patient["metadata"]["patient_class"]
        st = patient["metadata"]["surgery_type"]
        meta_cat = [
            self.enc_class.transform([pc])[0] if pc in self.enc_class.classes_ else 0,
            self.enc_surg.transform([st])[0] if st in self.enc_surg.classes_ else 0
        ]

        return {
            "cont_data": torch.tensor(cont_data.T, dtype=torch.float32), 
            "cat_data": torch.tensor(cat_data, dtype=torch.long),        
            "obs_mask": torch.tensor(obs_mask, dtype=torch.float32),     
            "gt_mask": torch.tensor(gt_mask, dtype=torch.float32),       
            "tp": torch.tensor(tp, dtype=torch.float32),
            "meta_cat": torch.tensor(meta_cat, dtype=torch.long)
        }
    
def get_dataloaders(json_path, pred_length=16, history_length=32, batch_size=128, 
                    samples_per_epoch=5000, num_test_samples=1000):
    # Chargement des données
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Calcul des indices
    n = len(data)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    # Création des subsets
    data_train = data[:train_end]
    data_val = data[train_end:val_end]
    data_test = data[val_end:]
    
    # Scaler
    all_cont, all_cat, all_class, all_surg = [], [], [], []
    for p in data_train:
        donnees = np.array(p["donnees"])
        all_cont.append(donnees[:, 1:8])
        all_cat.append(donnees[:, 8])
        all_class.append(p["metadata"]["patient_class"])
        all_surg.append(p["metadata"]["surgery_type"])
        
    scaler_cont = StandardScaler().fit(np.vstack(all_cont))
    enc_cat = LabelEncoder().fit(np.concatenate(all_cat))
    enc_class = LabelEncoder().fit(all_class)
    enc_surg = LabelEncoder().fit(all_surg)

    # Génération des fenetres glissantes
    train_windows = create_windows(data_train, history_length, pred_length)
    val_windows = create_windows(data_val, history_length, pred_length)
    test_windows = create_windows(data_test, history_length, pred_length)

    # Mélange aléatoire du dataset
    random.seed(42)
    random.shuffle(train_windows)
    random.shuffle(val_windows)
    random.shuffle(test_windows)

    if len(test_windows) > num_test_samples:
        test_windows = random.sample(test_windows, num_test_samples)

    # Initialisation des dataset
    train_dataset = CustomDataset(
        train_windows, history_length, pred_length,
        scaler_cont, enc_cat, enc_class, enc_surg
    )
    val_dataset = CustomDataset(
        val_windows, history_length, pred_length,
        scaler_cont, enc_cat, enc_class, enc_surg
    )
    test_dataset = CustomDataset(
        test_windows, history_length, pred_length,
        scaler_cont, enc_cat, enc_class, enc_surg
    )

    # Initialisation des dataloaders
    g = torch.Generator()
    g.manual_seed(42)
    
    train_sampler = RandomSampler(
        train_dataset, 
        replacement=True, 
        num_samples=samples_per_epoch,
        generator=g
    )
    val_sampler = RandomSampler(
        val_dataset, 
        replacement=True, 
        num_samples=int(samples_per_epoch/4),
        generator=g
    )
    test_sampler = RandomSampler(
        test_dataset, 
        replacement=True, 
        num_samples=int(samples_per_epoch/25),
        generator=g
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
    
    num_classes_event = len(enc_cat.classes_)
    
    return train_loader, val_loader, test_loader, train_dataset, num_classes_event

class CSDI_Custom(nn.Module):
    def __init__(self, config, device, num_cont_features, num_event_classes, num_classes_pat, num_classes_surg):
        super(CSDI_Custom, self).__init__()
        self.device = device
        
        self.num_cont = num_cont_features
        self.num_event_classes = num_event_classes
        self.cat_emb_dim = 8
        self.target_dim = self.num_cont + self.cat_emb_dim
        
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        
        self.event_embedding = nn.Embedding(num_event_classes, self.cat_emb_dim)
        self.event_classifier = nn.Linear(self.cat_emb_dim, num_event_classes) 
        
        self.meta_dim = 16
        self.pat_class_emb = nn.Embedding(num_classes_pat, self.meta_dim)
        self.surg_type_emb = nn.Embedding(num_classes_surg, self.meta_dim)

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim + self.meta_dim
        if not self.is_unconditional:
            self.emb_total_dim += 1
            
        self.embed_layer = nn.Embedding(num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim)
        
        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        
        self.diffmodel = diff_CSDI(config_diff, inputdim=1 if self.is_unconditional else 2)
        
        self.num_steps = config_diff["num_steps"]
        self.beta = np.linspace(config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps) ** 2
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        
        self.cce_weight = config["train"]["cce_weight"]
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def get_side_info(self, observed_tp, cond_mask, meta_emb):
        B, K, L = cond_mask.shape
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        
        feature_embed = self.embed_layer(torch.arange(self.target_dim).to(self.device))
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        
        meta_expanded = meta_emb.unsqueeze(1).unsqueeze(2).expand(-1, L, K, -1)
        
        side_info = torch.cat([time_embed, feature_embed, meta_expanded], dim=-1).permute(0, 3, 2, 1)
        
        if not self.is_unconditional:
            side_info = torch.cat([side_info, cond_mask.unsqueeze(1)], dim=1)
        return side_info

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, batch):
        cont_data = batch["cont_data"].to(self.device)
        cat_data = batch["cat_data"].to(self.device)
        obs_mask_1d = batch["obs_mask"].to(self.device)
        gt_mask_1d = batch["gt_mask"].to(self.device)
        
        cat_emb = self.event_embedding(cat_data).permute(0, 2, 1)
        x_target = torch.cat([cont_data, cat_emb], dim=1)
        
        B, K_total, L = x_target.shape
        cond_mask = obs_mask_1d.unsqueeze(1).expand(-1, K_total, -1)
        target_mask = gt_mask_1d.unsqueeze(1).expand(-1, K_total, -1)
        
        meta_1 = self.pat_class_emb(batch["meta_cat"][:, 0].to(self.device))
        meta_2 = self.surg_type_emb(batch["meta_cat"][:, 1].to(self.device))
        meta_total = meta_1 + meta_2 

        side_info = self.get_side_info(batch["tp"].to(self.device), cond_mask, meta_total)
        
        t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]
        noise = torch.randn_like(x_target)
        noisy_data = (current_alpha ** 0.5) * x_target + (1.0 - current_alpha) ** 0.5 * noise

        if self.is_unconditional:
            total_input = noisy_data.unsqueeze(1)
        else:
            cond_obs = (cond_mask * x_target).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)

        predicted_noise = self.diffmodel(total_input, side_info, t)
        
        residual = (noise - predicted_noise) * target_mask
        loss_mse = (residual ** 2).sum() / target_mask.sum().clamp(min=1)
        
        x0_pred = (noisy_data - (1.0 - current_alpha) ** 0.5 * predicted_noise) / (current_alpha ** 0.5)
        cat_emb_pred = x0_pred[:, self.num_cont:, :]
        logits = self.event_classifier(cat_emb_pred.permute(0, 2, 1))
        
        gt_mask_flat = gt_mask_1d.view(-1)
        logits_flat = logits.reshape(-1, self.num_event_classes)
        cat_true_flat = cat_data.view(-1).clone()
        cat_true_flat[gt_mask_flat == 0] = -1 
        
        loss_ce = self.ce_loss(logits_flat, cat_true_flat)

        return loss_mse + self.cce_weight * loss_ce 
    
def inference(model, test_loader, scaler_cont, device, path="checkpoints"):
    model.eval()
    all_real = []
    all_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            cont_data = batch["cont_data"].to(device)
            cat_data = batch["cat_data"].to(device)
            obs_mask_1d = batch["obs_mask"].to(device)
            B, K_cont, L = cont_data.shape
            
            meta_1 = model.pat_class_emb(batch["meta_cat"][:, 0].to(device))
            meta_2 = model.surg_type_emb(batch["meta_cat"][:, 1].to(device))
            meta_total = meta_1 + meta_2
            
            cond_mask = obs_mask_1d.unsqueeze(1).expand(-1, model.target_dim, -1)
            side_info = model.get_side_info(batch["tp"].to(device), cond_mask, meta_total)

            cat_emb_real = model.event_embedding(cat_data).permute(0, 2, 1) # Vraies données embedding
            true_x_target = torch.cat([cont_data, cat_emb_real], dim=1) # (B, Target_dim, L)
            
            n_samples = 10
            samples_pred = torch.zeros(B, n_samples, model.target_dim, L).to(device)
            
            real_target = torch.cat([cont_data, cat_data.unsqueeze(1).float()], dim=1) # (B, 8, L)
            
            for i in range(n_samples):
                current_sample = torch.randn(B, model.target_dim, L).to(device)
                for t in range(model.num_steps - 1, -1, -1):
                    t_tensor = torch.tensor([t]).to(device)
                    
                    if not model.is_unconditional:
                        cond_obs = (cond_mask * true_x_target).unsqueeze(1) 
                        noisy_tgt = ((1 - cond_mask) * current_sample).unsqueeze(1)
                        diff_input = torch.cat([cond_obs, noisy_tgt], dim=1)
                    else:
                        diff_input = current_sample.unsqueeze(1)
                    
                    predicted = model.diffmodel(diff_input, side_info, t_tensor)
                    
                    coeff1 = 1 / model.alpha_hat[t] ** 0.5
                    coeff2 = (1 - model.alpha_hat[t]) / (1 - model.alpha[t]) ** 0.5
                    current_sample = coeff1 * (current_sample - coeff2 * predicted)
                    
                    if t > 0:
                        sigma = ((1.0 - model.alpha[t - 1]) / (1.0 - model.alpha[t]) * model.beta[t]) ** 0.5
                        current_sample += sigma * torch.randn_like(current_sample)
                
                current_sample = (1 - cond_mask) * current_sample + cond_mask * true_x_target
                samples_pred[:, i] = current_sample.detach()

            final_pred = samples_pred.mean(dim=1)
            
            # Découpage et classification
            pred_cont = final_pred[:, :model.num_cont, :].clone()
            pred_cat_emb = final_pred[:, model.num_cont:, :]
            logits = model.event_classifier(pred_cat_emb.permute(0, 2, 1))
            pred_cat = torch.argmax(logits, dim=-1).unsqueeze(1).float()
            
            # Dénormalisation
            for b in range(B):
                pred_cont[b] = torch.tensor(scaler_cont.inverse_transform(pred_cont[b].cpu().numpy().T).T)
                real_target[b, :model.num_cont, :] = torch.tensor(scaler_cont.inverse_transform(real_target[b, :model.num_cont, :].cpu().numpy().T).T)

            full_pred = torch.cat([pred_cont.cpu(), pred_cat.cpu()], dim=1)
            all_real.append(real_target.cpu().numpy())
            all_pred.append(full_pred.numpy())
            
    np_real = np.concatenate(all_real, axis=0)
    np_pred = np.concatenate(all_pred, axis=0)
    
    return np_real, np_pred