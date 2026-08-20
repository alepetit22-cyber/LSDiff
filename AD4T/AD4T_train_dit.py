#############
# ADiff4TPP #
#############

###########################
# Author: Adrien Le Petit #
# Release: June 2026      #
###########################

import os
import time
import json
import sys
import random
import torch
import argparse
import numpy as np
import torch.nn as nn
from DiT_models import DiT
from torchdiffeq import odeint
from async_lib import AsyncMatrix
from train_vae.model import Model_VAE
from torch.utils.data import Dataset, DataLoader

sys.path.append("../")
from Data.metrics import DatasetEvaluator 
from Data.utils import create_windows, analyser_evenements_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TPPSequenceDataset(Dataset):
    def __init__(self, sequences, window_len, num_mean, num_std, cat_to_id):
        self.window_len = window_len
        self.sequences_num = []
        self.sequences_cat = []
        
        for num_feat, cat_feat in sequences:
            # Normalisation
            num_feat = (torch.tensor(num_feat, dtype=torch.float32) - num_mean) / num_std
            cat_feat = torch.tensor([cat_to_id[c] for c in cat_feat], dtype=torch.long).unsqueeze(-1)
            
            self.sequences_num.append(num_feat)
            self.sequences_cat.append(cat_feat)
            
    def __len__(self):
        return len(self.sequences_num)

    def __getitem__(self, idx):
        num_seq = self.sequences_num[idx]
        cat_seq = self.sequences_cat[idx]
        seq_len = len(num_seq)
        
        return num_seq, cat_seq, seq_len
    
def create_sequences(data):
    all_cat = []
    sequences = []
    for patient in data:
        donnees = np.array(patient['donnees'])
        if len(donnees) == 0:
            continue
            
        # Séparation des colonnes
        temps = donnees[:, 0]
        delta_t = np.zeros_like(temps)
        delta_t[1:] = temps[1:] - temps[:-1]
        
        # Features numériques: [delta_t, FC, PAS, PAM, PAD, Temp, SpO2, FR]
        num_features = np.column_stack([delta_t, donnees[:, 1:8]])
        # Feature catégorielle: event_code
        cat_features = donnees[:, 8].astype(int)
        
        all_cat.append(cat_features)
        sequences.append((num_features, cat_features))

    return sequences, all_cat

#####################################################
# PARAMÉTRAGE DES VARIABLES                         #
#####################################################
# Initialisation du parser
def parse_args():
    parser = argparse.ArgumentParser(description='ADiff4TPP Training Script')
    # VAE
    parser.add_argument('--num_layer', type=int, default=3, help='Number of layers')
    parser.add_argument('--num_head_VAE', type=int, default=4, help='Number of heads in the VAE')
    parser.add_argument('--factor', type=int, default=16, help='Factor of dimensionality expansion for VAE')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--cce_weight', type=float, default=0.1, help='CCE weight')
    # DIT
    parser.add_argument('--hidden_size', type=int, default=160, help='Latent dimension')
    parser.add_argument('--depth', type=int, default=7, help='Depth of the DIT')
    parser.add_argument('--num_head_DIT', type=int, default=8, help='Number of heads in the DIT')



    args = parser.parse_args()
    return args

arguments = parse_args()

# Initialisation de la longueur des données
prediction_len = 16
history_len = 32 
window_len = history_len + prediction_len

class Args:
    # VAE
    vae_epochs = 3
    d_latent = 8
    window_len = window_len
    num_layer = arguments.num_layer
    batch_size = arguments.batch_size
    num_head = arguments.num_head_VAE
    factor = arguments.factor
    cce_weight = arguments.cce_weight
    # DIT
    dit_epochs = 3
    num_rows = window_len
    latent_size = 10
    hidden_size = arguments.hidden_size
    depth = arguments.depth
    num_head_DIT = arguments.num_head_DIT
    mlp_ratio = 4
    # Inférence
    num_samples = 10
    

args = Args()

####################################################
# PARAMÉTRAGE DES DATASETS                         #
####################################################
# Chargement des données
chemin_fichier = '../Data/db_meta_2000_FR.json'
with open(chemin_fichier, 'r') as f:
    data = json.load(f)

# Calcul des indices
n = len(data)
train_end = int(0.8 * n)
val_end = int(0.9 * n)

# Création des subsets
data_train = data[:train_end]
data_val = data[train_end:val_end]
data_test = data[val_end:]

# Configuration du dossier de sortie
path_dir = f"checkpoints_pred16_hist32_VAE_nl{args.num_layer}_nh{args.num_head}_f{args.factor}_bs{args.batch_size}_cce{args.cce_weight}_DIT_hd{args.hidden_size}_d{args.depth}_nh{args.num_head_DIT}/"
os.makedirs(path_dir, exist_ok=True)
print("=" * 60)
print(f"[Dossier de sauvegarde] : {path_dir}")

# Découpage des datasets
data_train = create_windows(data_train, history_len, prediction_len)
data_val = create_windows(data_val, history_len, prediction_len)
data_test = create_windows(data_test[:500], history_len, prediction_len)

# Mélange aléatoire du dataset
random.seed(42)
random.shuffle(data_train)
random.shuffle(data_val)
random.shuffle(data_test)

train_size = int(0.8 * len(data))
test_size = int(0.1 * len(data))

print("=" * 60)
print("[Analyse des événements]")
analyser_evenements_dataset(data_train, "Train")
analyser_evenements_dataset(data_val, "Val")
analyser_evenements_dataset(data_test, "Test")
print("=" * 60)

train_sequences, all_cat = create_sequences(data_train)
val_sequences, _ = create_sequences(data_val)
test_sequences, _ = create_sequences(data_test)

# Normalisation des features numériques
flat_train_num = np.vstack([seq[0] for seq in train_sequences])
num_mean = torch.tensor(flat_train_num.mean(axis=0), dtype=torch.float32)
num_std = torch.tensor(flat_train_num.std(axis=0) + 1e-6, dtype=torch.float32)

# Mapping des catégories
flat_cat = np.concatenate(all_cat)
unique_cats = np.unique(flat_cat)
cat_to_id = {c: i for i, c in enumerate(unique_cats)}
id_to_cat = {i: c for i, c in enumerate(unique_cats)}
num_categories = len(unique_cats)

# Création des dataloaders
train_dataset = TPPSequenceDataset(train_sequences, args.window_len, num_mean, num_std, cat_to_id)
val_dataset = TPPSequenceDataset(val_sequences, args.window_len, num_mean, num_std, cat_to_id)
test_dataset = TPPSequenceDataset(test_sequences, args.window_len, num_mean, num_std, cat_to_id)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Métadonnées d'affichage
print("=" * 60)
print("[Paramètres du dataset]")
print(f"Dimensions continues : {flat_train_num.shape[1]}")
print(f"Nombre de classes d'événements uniques : {num_categories}")
print(f"Taille du set d'entraînement : {len(train_dataset)} séquences")
print(f"Taille du set de validation  : {len(val_dataset)} séquences")
print(f"Taille du set de test        : {len(test_dataset)} séquences")
print(f"Dimension max de la séquence : {args.window_len}")

print("=" * 60)
print("[Arguments de la Grid Search]")
for arg, value in vars(arguments).items():
    print(f"   {arg:<22}: {value}")

##################################################
# ENTRAINEMENT DU VAE                            #
##################################################

d_numerical = 8  # Temps + 7 constantes
categories = [num_categories]  # Nombre de valeurs pour event_code
n_tokens = 1 + d_numerical + len(categories)

# Initialisation du VAE
vae = Model_VAE(
    num_layers=args.num_layer, 
    d_numerical=d_numerical, 
    categories=categories, 
    d_token=args.d_latent, 
    n_head=args.num_head, 
    factor=args.factor, 
    bias=True, 
    transformer=True
).to(device)

optimizer_vae = torch.optim.Adam(vae.parameters(), lr=1e-3)
ce_loss_fn = nn.CrossEntropyLoss()

print("=" * 60)
print("[Début de l'entraînement VAE]")
debut_train_vae = time.time()
print(f"Heure de début de l'entraînement : {time.strftime('%Y-%m-%d %H:%M:%S')}")

for epoch in range(args.vae_epochs):
    vae.train()
    train_loss, val_loss = 0, 0
    train_metrics = {"mse": 0, "kld": 0, "cce": 0}
    
    for batch_num, batch_cat, _ in train_dataloader:
        batch_num, batch_cat = batch_num.to(device), batch_cat.to(device)
        
        flat_num = batch_num.view(-1, d_numerical)
        flat_cat = batch_cat.view(-1, len(categories))
        
        optimizer_vae.zero_grad()
        Recon_X_num, Recon_X_cat, mu_z, logvar_z = vae(flat_num, flat_cat)
        
        mse = (flat_num - Recon_X_num).pow(2).mean()
        cce = ce_loss_fn(Recon_X_cat[0], flat_cat[:, 0])
        
        temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
        kld = -0.5 * torch.mean(temp.mean(-1).mean())
        
        loss = mse + args.cce_weight * cce + 0.01 * kld
        train_metrics["mse"] += mse.item()
        train_metrics["cce"] += cce.item()
        train_metrics["kld"] += kld.item()

        loss.backward()
        optimizer_vae.step()
        train_loss += loss.item()
    
    vae.eval()
    val_metrics = {"mse": 0, "kld": 0, "cce": 0}
    with torch.no_grad():
        for batch_num, batch_cat, _ in val_dataloader:
            batch_num, batch_cat = batch_num.to(device), batch_cat.to(device)
            
            flat_num = batch_num.view(-1, d_numerical)
            flat_cat = batch_cat.view(-1, len(categories))
            
            Recon_X_num, Recon_X_cat, mu_z, logvar_z = vae(flat_num, flat_cat)
            
            mse = (flat_num - Recon_X_num).pow(2).mean()
            temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
            kld = -0.5 * torch.mean(temp.mean(-1).mean())
            cce = ce_loss_fn(Recon_X_cat[0], flat_cat[:, 0])
            
            loss = mse + args.cce_weight * cce + 0.01 * kld
            
            val_metrics["mse"] += mse.item()
            val_metrics["cce"] += cce.item()
            val_metrics["kld"] += kld.item()
            val_loss += loss.item()

    # Calcul des moyennes pour l'affichage hégémonique
    for k in train_metrics: train_metrics[k] /= len(train_dataloader)
    for k in val_metrics: val_metrics[k] /= len(val_dataloader)    
    
    print(f"Epoch {epoch + 1}/{args.vae_epochs}, Train Loss: {train_loss / len(train_dataloader):.4f}, Val Loss: {val_loss / len(val_dataloader):.4f} "
          f"| MSE: {train_metrics['mse']:.4f}/{val_metrics['mse']:.4f} " 
          f"| CCE: {train_metrics['cce']:.4f}/{val_metrics['cce']:.6f} "
          f"| KLD: {train_metrics['kld']:.0f}/{val_metrics['kld']:.0f}")

fin_train_vae = time.time()
print("[Entraînement VAE terminé]")
print(f"Heure de fin de l'entraînement : {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Sauvegarde du modèle faim
modele_path = os.path.join(path_dir, "vae_custom_model.pt")
torch.save(vae.state_dict(), modele_path)
print(f"[Modèle sauvegardé] : {modele_path}")

#####################################
# INFÉRENCE DU VAE
#####################################
vae.eval()

print("=" * 60)
print("[Début de l'inférence VAE]")
debut_inf_vae = time.time()
print(f"Heure de début de l'inférence : {time.strftime('%Y-%m-%d %H:%M:%S')}")

real_num_list = []
real_cat_list = []
gen_num_list = []
gen_cat_list = []

with torch.no_grad():
    for batch_num, batch_cat, _ in test_dataloader:
        batch_num = batch_num.to(device)
        batch_cat = batch_cat.to(device)
        
        flat_num = batch_num.view(-1, d_numerical)
        flat_cat = batch_cat.view(-1, len(categories))
        
        recon_x_num, recon_x_cat, _, _ = vae(flat_num, flat_cat)
        recon_num = recon_x_num.view(batch_num.shape)
        recon_cat_preds = recon_x_cat[0].argmax(dim=-1).view(batch_cat.shape)
        
        real_num_list.append(batch_num.cpu().numpy())
        real_cat_list.append(batch_cat.cpu().numpy())
        gen_num_list.append(recon_num.cpu().numpy())
        gen_cat_list.append(recon_cat_preds.cpu().numpy())

real_num_all = np.concatenate(real_num_list, axis=0)
real_cat_all = np.concatenate(real_cat_list, axis=0)
gen_num_all = np.concatenate(gen_num_list, axis=0)
gen_cat_all = np.concatenate(gen_cat_list, axis=0)

num_mean_np = num_mean.numpy()
num_std_np = num_std.numpy()

real_num_denorm = real_num_all * num_std_np + num_mean_np
gen_num_denorm = gen_num_all * num_std_np + num_mean_np

vectorized_id_to_cat = np.vectorize(id_to_cat.get)
real_cat_orig = vectorized_id_to_cat(real_cat_all)
gen_cat_orig = vectorized_id_to_cat(gen_cat_all)

real_data = np.concatenate([real_num_denorm, real_cat_orig], axis=-1)
gen_data = np.concatenate([gen_num_denorm, gen_cat_orig], axis=-1)

real_data = real_data[:,:,1:]
gen_data = gen_data[:,:,1:]

np.save(f"{path_dir}real_data_vae.npy", real_data)
np.save(f"{path_dir}gen_data_vae.npy", gen_data)

fin_inf_vae = time.time()
print("[Inférence terminée]")
print(f"Heure de fin de l'inférence : {time.strftime('%Y-%m-%d %H:%M:%S')}")

print(f"Format de real_data : {real_data.shape}  (Batch, Temporel, Constantes)")
print(f"Format de gen_data  : {gen_data.shape}  (Batch, Temporel, Constantes)")
print(f"Datasets sauvegardés avec succès dans : {path_dir}")

#############################################
# EVALUATION DES RÉSULTATS                  #
#############################################
# Clip de la SpO2
gen_data[:,:,5] = np.clip(gen_data[:,:,5], a_min=0.0, a_max=100.0)

# Arrondi des événements
gen_data[:,:,7] = np.round(gen_data[:, :, 7])

evaluator = DatasetEvaluator(
    real_data, 
    gen_data,
    col_names=["FC", "PAS", "PAM", "PAD", "Temp", "SpO2", "FR", "event_code"],
    path_dir=f"{path_dir}vae/"
)

evaluator.run_full_analysis()

#####################################
# ENTRAINEMENT DU DIT               #
#####################################

# Dimension latente aplatie = d_latent * nombre_de_tokens
latent_dim = args.d_latent * n_tokens 
print(f"Nombre de dimensions latentes : {latent_dim}")

flow_model = DiT(
    num_rows=args.num_rows,
    latent_size=latent_dim, # Important: 10 * 8 = 80
    hidden_size=args.hidden_size,        # 320 est un multiple de 80 et de 8
    depth=args.depth,
    num_heads=args.num_head_DIT,
    mlp_ratio=args.mlp_ratio,
    learn_sigma=False
).to(device)

optimizer_flow = torch.optim.AdamW(flow_model.parameters(), lr=1e-4)

# On s'assure que le VAE est bien en mode évaluation (ses poids sont figés)
vae.eval()
print("=" * 60)
print("[Début de l'entraînement DIT]")
debut_train_dit = time.time()
print(f"Heure de début de l'entraînement : {time.strftime('%Y-%m-%d %H:%M:%S')}")
for epoch in range(args.dit_epochs):
    
    flow_model.train()
    train_loss = 0
    for batch_num, batch_cat, batch_len in train_dataloader:
        
        batch_num = batch_num.to(device)
        batch_cat = batch_cat.to(device)
        
        # 1. Matrice asynchrone (temps de diffusion)
        A = AsyncMatrix(batch_len, args.num_rows).to(device)
        attn_mask = A.attn_mask.to(device)

        # 2. Extraction des latents via le VAE figé
        with torch.no_grad():
            flat_num = batch_num.view(-1, d_numerical)
            flat_cat = batch_cat.view(-1, 1)
            
            z_token = vae.VAE.Tokenizer(flat_num, flat_cat)
            z = vae.VAE.get_embedding(z_token)
            
            z = z.view(batch_num.shape[0], args.num_rows, -1)  
            
        # 3. Processus de bruitage (Flow)
        noise_fixed = torch.randn_like(z)
        
        t = torch.rand(z.shape[0]).view(-1, 1).to(device) 
        
        A_t = A(t)
        A_t_dot = A.derivative(t).unsqueeze(-1)
        
        zt = A_t.unsqueeze(-1) * z + (1 - A_t.unsqueeze(-1)) * noise_fixed
        target = z - noise_fixed 
        
        # 4. Prédiction et perte
        optimizer_flow.zero_grad()
        pred = flow_model(zt, A_t, attn_mask)
        
        loss = torch.mean(A_t_dot.abs() * (pred - target)**2)
        loss.backward()
        optimizer_flow.step()
        
        train_loss += loss.item()
    
    flow_model.eval()
    val_loss = 0
    with torch.no_grad(): # On désactive le calcul des gradients
        for batch_num, batch_cat, batch_len in val_dataloader:
            
            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)
            
            # 1. Matrice asynchrone
            A = AsyncMatrix(batch_len, args.num_rows).to(device)
            attn_mask = A.attn_mask.to(device)

            # 2. Extraction des latents
            flat_num = batch_num.view(-1, d_numerical)
            flat_cat = batch_cat.view(-1, 1)
            
            z_token = vae.VAE.Tokenizer(flat_num, flat_cat)
            z = vae.VAE.get_embedding(z_token)
            z = z.view(batch_num.shape[0], args.num_rows, -1)  
                
            # 3. Processus de bruitage
            noise_fixed = torch.randn_like(z)
            t = torch.rand(z.shape[0]).view(-1, 1).to(device) 
            
            A_t = A(t)
            A_t_dot = A.derivative(t).unsqueeze(-1)
            
            zt = A_t.unsqueeze(-1) * z + (1 - A_t.unsqueeze(-1)) * noise_fixed
            target = z - noise_fixed 
            
            # 4. Prédiction et perte (sans backward ni step)
            pred = flow_model(zt, A_t, attn_mask)
            loss = torch.mean(A_t_dot.abs() * (pred - target)**2)
            
            val_loss += loss.item()
            
    # Affichage des métriques d'entraînement et de validation
    print(f"Epoch {epoch+1}/{args.dit_epochs} | Loss Flow Train: {train_loss/len(train_dataloader):.6f} | Loss Flow Val: {val_loss/len(val_dataloader):.6f}")
print("[Entraînement DIT terminé]")
fin_train_dit = time.time()
print(f"Heure de fin de l'entraînement : {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Sauvegarde du modèle
modele_path = os.path.join(path_dir, "dit_custom_model.pt")
torch.save(flow_model.state_dict(), modele_path)
print(f"[Modèle sauvegardé] : {modele_path}")

#########################################
# INFERENCE DU DiT
#########################################
vae.eval()
flow_model.eval()

# Nombre de simulations par patient
num_samples = args.num_samples  

all_real_num = []
all_real_cat = []
all_gen_num = []
all_gen_cat = []

print("=" * 60)
print(f"[Début de l'inférence DiT]")
debut_inf_dit = time.time()
print(f"Heure de début : {time.strftime('%Y-%m-%d %H:%M:%S')}")

with torch.no_grad():
    for batch_num, batch_cat, _ in test_dataloader:
        current_batch_size = batch_num.shape[0]
        
        # Dénormalisation et remapping des variables réelles (cible)
        real_num = batch_num.cpu() * num_std + num_mean
        real_cat_indices = batch_cat.cpu().squeeze(-1).numpy()
        real_cat = np.vectorize(id_to_cat.get)(real_cat_indices)
        
        all_real_num.append(real_num.numpy())
        all_real_cat.append(real_cat[..., np.newaxis])
        
        # Extraction des latents réels (pour l'historique de conditionnement)
        flat_num = batch_num.to(device).view(-1, d_numerical)
        flat_cat = batch_cat.to(device).view(-1, 1)
        
        z_token = vae.VAE.Tokenizer(flat_num, flat_cat)
        z_true = vae.VAE.get_embedding(z_token)
        z_true = z_true.view(current_batch_size, args.num_rows, -1) # Shape: (B, 48, latent_dim)
        
        simulated_len = torch.ones(current_batch_size, dtype=torch.long) * args.num_rows
        A_infer = AsyncMatrix(simulated_len, args.num_rows).to(device)
        
        # Liste pour stocker les résultats latents de chaque simulation
        batch_samples_z = []
        
        # Boucle de génération num_sample fois
        for sample_idx in range(num_samples):
            # Chaque simulation part d'un bruit initial différent
            noise_start = torch.randn((current_batch_size, args.num_rows, latent_dim)).to(device)
            
            def ode_func(t, x):
                t_batch = t.expand(current_batch_size).view(-1, 1).to(device)
                A_t = A_infer(t_batch)
                A_t_dot = A_infer.derivative(t_batch).unsqueeze(-1)
                
                # Conditionnement linéaire de l'historique
                A_t_unsp = A_t.unsqueeze(-1)
                x_conditioned = x.clone()
                x_conditioned[:, :history_len, :] = (
                    A_t_unsp[:, :history_len, :] * z_true[:, :history_len, :] + 
                    (1 - A_t_unsp[:, :history_len, :]) * noise_start[:, :history_len, :]
                )
                
                causal_mask = A_infer.attn_mask.to(device) # version précédente fonctionnelle: torch.ones(1, 1, args.num_rows, args.num_rows).to(device)
                v = flow_model(x_conditioned, A_t, causal_mask)
                return A_t_dot * v

            # Résolution de l'ODE pour cette simulation
            solution = odeint(ode_func, noise_start, A_infer.times, method='euler')
            z_sample = solution[-1] # Shape: (B, 48, latent_dim)
            
            # Forcer le vrai historique dans le latent généré pour cette simulation
            z_sample[:, :history_len, :] = z_true[:, :history_len, :]
            
            batch_samples_z.append(z_sample.unsqueeze(0)) # Ajoute une dimension d'échantillon
            
        # Calcul de la médiane à travers les simulations
        # Concaténation le long d'un nouvel axe -> Shape: (num_samples, B, 48, latent_dim)
        stacked_z = torch.cat(batch_samples_z, dim=0)
        # torch.median renvoie un tuple (values, indices), on ne prend que les valeurs [0]
        z_median = torch.median(stacked_z, dim=0)[0] # Shape: (B, 48, latent_dim)
        
        # Décodage de la trajectoire médiane unique via le VAE
        z_tokens = z_median.view(-1, n_tokens, args.d_latent)
        pred_num, pred_cat = vae.get_decoding(z_tokens)
        
        # Reconstruction des dimensions (Batch, MaxLen, Features)
        pred_num = pred_num.view(current_batch_size, args.num_rows, d_numerical).cpu()
        pred_cat_indices = pred_cat[0].argmax(dim=-1).view(current_batch_size, args.num_rows).cpu().numpy()
        
        # Dénormalisation et remapping final
        pred_num = pred_num * num_std + num_mean
        pred_cat = np.vectorize(id_to_cat.get)(pred_cat_indices)
        
        all_gen_num.append(pred_num.numpy())
        all_gen_cat.append(pred_cat[..., np.newaxis])

# Regroupement et concaténation finale
real_num_np = np.concatenate(all_real_num, axis=0)
real_cat_np = np.concatenate(all_real_cat, axis=0)
real_data = np.concatenate([real_num_np, real_cat_np], axis=-1)

gen_num_np = np.concatenate(all_gen_num, axis=0)
gen_cat_np = np.concatenate(all_gen_cat, axis=0)
gen_data = np.concatenate([gen_num_np, gen_cat_np], axis=-1)

fin_inf_dit = time.time()
print("[Inférence terminée]")
print(f"Heure de fin : {time.strftime('%Y-%m-%d %H:%M:%S')}")

real_data = real_data[:,32:,1:]
gen_data = gen_data[:,32:,1:]

# Sauvegarde des fichiers Numpy
np.save(f'{path_dir}real_data_dit.npy', real_data)
np.save(f'{path_dir}gen_data_dit.npy', gen_data)
print(f"Fichiers 'real_data_dit.npy' et 'gen_data_dit.npy' sauvegardés dans {path_dir}")


#####################################################
# ÉVALUATION DES DURÉES                             #
#####################################################
durée_train_vae = fin_train_vae - debut_train_vae
durée_train_dit = fin_train_dit - debut_train_dit
durée_train = durée_train_vae + durée_train_dit
durée_inf_vae = fin_inf_vae - debut_inf_vae
durée_inf_dit = fin_inf_dit - debut_inf_dit

m_train, s_train = divmod(durée_train_vae + durée_train_dit, 60)
m_inf, s_inf = divmod(durée_inf_dit, 60)

print("=" * 60)
print("[Evaluation de la durée d'entrainement et d'inférence]")
print(f"Temps d'entraînement VAE : {int(durée_train_vae // 60)} min {int(durée_train_vae % 60)} s")
print(f"Temps d'entraînement DIT  : {int(durée_train_dit // 60)} min {int(durée_train_dit % 60)} s")
print(f"Temps d'entraînement : {int(m_train)} min {int(s_train)} s (Total: {durée_train:.2f} secondes)")
print(f"Temps d'inférence    : {int(m_inf)} min {int(s_inf)} s (Total: {durée_inf_dit:.2f} secondes)")
print(f"Temps total du run   : {int((durée_train + durée_inf_dit) // 60)} min {int((durée_train + durée_inf_dit) % 60)} s")
print("=" * 60)

#############################################
# EVALUATION DES RÉSULTATS                  #
#############################################
# Clip de la SpO2
gen_data[:,:,5] = np.clip(gen_data[:,:,5], a_min=0.0, a_max=100.0)

# Arrondi des événements
gen_data[:,:,7] = np.round(gen_data[:, :, 7])

evaluator = DatasetEvaluator(
    real_data, 
    gen_data,
    col_names=["FC", "PAS", "PAM", "PAD", "Temp", "SpO2", "FR", "event_code"],
    path_dir=f"{path_dir}DiT/"
)

evaluator.run_full_analysis()