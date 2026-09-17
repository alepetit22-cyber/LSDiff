############
# CSDI     #
############

###########################
# Author: Adrien Le Petit #
# Release: June 2026      #
###########################

import os
import sys
import time
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

sys.path.append("../../")
from csdi_utils_last import get_dataloaders, inference, CSDI_Custom 
from metrics2 import DatasetEvaluator 
from Data.utils import analyser_evenements_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

#####################################################
# PARAMÉTRAGE DES VARIABLES                         #
#####################################################
def parse_args():
    parser = argparse.ArgumentParser(description='CSDI Training Script')
    parser.add_argument('--num_layer', type=int, default=4, help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of head')
    parser.add_argument('--channels', type=int, default=128, help='Number of residual channels')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--diffusion_embedding_dim', type=int, default=128, help='diffusion_embedding_dim')
    parser.add_argument('--cce_weight', type=float, default=0.2, help='CCE weight')

    args=parser.parse_args()
    return args

args = parse_args()
num_layer = args.num_layer
channels = args.channels
num_heads = args.num_heads
batch_size = args.batch_size
diffusion_embedding_dim = args.diffusion_embedding_dim
cce_weight = args.cce_weight

# Dimension de contexte et d'inférence
prediction_len = 16
history_len = 32 
window_len = prediction_len + history_len

# Configuration du modèle
config = {
    "train": {
        "epochs": 50,
        "batch_size": batch_size,
        "lr": 1.0e-3,
        "itr_per_epoch": 1000,
        "cce_weight": cce_weight
    },
    "diffusion": {
        "layers": num_layer,
        "channels": channels,
        "nheads": num_heads,
        "diffusion_embedding_dim": diffusion_embedding_dim,
        "beta_start": 0.0001,
        "beta_end": 0.5,
        "num_steps": 50,
        "schedule": "quad",
        "is_linear": True
    },
    "model": {
        "is_unconditional": False, # False pour prédire à partir de l'historique
        "timeemb": 64,
        "featureemb": 16
    }
}

# Création du dossier de sortie
path = f"checkpoints_pred{prediction_len}_hist{history_len}_nl{num_layer}_c{channels}_nh{num_heads}_bs{batch_size}_ded{diffusion_embedding_dim}_cce{cce_weight}/"
os.makedirs(path, exist_ok=True)
print("=" * 60)
print(f"[Dossier de sauvegarde] : {path}")

####################################################
# PARAMÉTRAGE DES DATASETS                         #
####################################################

# Récupération de données
train_loader, val_loader, test_loader, train_dataset, num_classes_event = get_dataloaders(
    "../Data/db_200.json",
    pred_length=prediction_len,
    history_length=history_len,
    batch_size=config["train"]["batch_size"],
    )

# Configuration des métadonnées
num_classes_pat = len(train_dataset.enc_class.classes_)
num_classes_surg = len(train_dataset.enc_surg.classes_)

print("=" * 60)
print("[Paramètres du dataset]")
print(f"Nombre de classes de patients uniques : {num_classes_pat}")
print(f"Nombre de types de chirurgies uniques : {num_classes_surg}")
print(f"Nombre de patients Train: {len(train_loader.dataset)}")
print(f"Nombre de patients Val: {len(val_loader.dataset)}")
print(f"Nombre de patients Test: {len(test_loader.dataset)}")
print(f"Dimension du context: {history_len}")
print(f"Dimension de l'inférence: {prediction_len}")

print("=" * 60)
print("[Analyse des événements]")
analyser_evenements_dataset(train_loader.dataset.data, "Train")
analyser_evenements_dataset(val_loader.dataset.data, "Val")
analyser_evenements_dataset(test_loader.dataset.data, "Test")
print("=" * 60)

print("[Arguments de la Grid Search]")
for arg, value in vars(args).items():
    print(f"   {arg:<22}: {value}")
print("[Paramètres du modèle]")
for key, value in config.items():
    print(f"   {key:<22}: {value}")

##################################################
# ENTRAINEMENT DU MODÈLE                         #
##################################################

# Initialisation du modèle
model = CSDI_Custom(config, device, num_cont_features=7, num_event_classes=num_classes_event, 
                    num_classes_pat=num_classes_pat, num_classes_surg=num_classes_surg).to(device)

# Configuration de l'optimiseur
optimizer = optim.Adam(model.parameters(), lr=config["train"]["lr"])

best_val_loss = float('inf')

# Boucle d'entraînement
print("=" * 60)
print("[Début de l'entraînement]")
debut_train = time.time()
print(f"Heure de début de l'entraînement : {time.strftime('%Y-%m-%d %H:%M:%S')}")

for epoch in range(config["train"]["epochs"]):
    model.train()
    total_loss = 0
    
    steps_train = 0
    for batch in train_loader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        steps_train += 1
        
    moyenne_train_loss = total_loss / steps_train
        
    model.eval()
    val_loss = 0
    steps_val = 0
    
    with torch.no_grad():
        for batch in val_loader:
            loss = model(batch)
            val_loss += loss.item()
            steps_val += 1

    moyenne_val_loss = val_loss / steps_val
    
    # Sauvegarde
    if moyenne_val_loss < best_val_loss:
        best_val_loss = moyenne_val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, os.path.join(path, "best_model.pt"))
        
    print(f"Epoch {epoch + 1}/{config['train']['epochs']} - Train Loss: {moyenne_train_loss:.5f} | Val Loss: {moyenne_val_loss:.5f}")

fin_train = time.time()
print("[Entraînement terminé]")
print(f"Heure de fin de l'entraînement : {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Sauvegarde du modèle
modele_path = os.path.join(path, "csdi_custom_model.pt")
torch.save(model.state_dict(), modele_path)
print(f"[Modèle sauvegardé] : {modele_path}")

#####################################
# INFÉRENCE                         #
#####################################
# Chargment du meilleur modèle
if best_state is not None:
    model.load_state_dict(best_state)
    model.to(device)

print("=" * 60)
print("[Début de l'inférence]")
debut_inf = time.time()
print(f"Heure de début de l'inférence : {time.strftime('%Y-%m-%d %H:%M:%S')}")
real_data, gen_data =inference(
    model,
    test_loader,
    train_dataset.scaler_cont,
    device,
    path=path
    )
fin_inf = time.time()
print("[Inférence terminée]")
print(f"Heure de fin de l'inférence : {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Transposition de la matrice
if real_data.ndim == 3 and real_data.shape[2] == window_len:
    real_data = real_data.transpose(0, 2, 1)
    gen_data  = gen_data.transpose(0, 2, 1)

# Retrait de l'historique
if real_data.shape[1] == window_len:
    real_data = real_data[:,history_len:,:]
    gen_data  = gen_data[:,history_len:,:]

print(f"Format de real_data : {real_data.shape}  (Batch, Temporel, Constantes)")
print(f"Format de gen_data  : {gen_data.shape}  (Batch, Temporel, Constantes)")

# Enregistrement des datasets
np.save(os.path.join(path, "real_data.npy"), real_data)
np.save(os.path.join(path, "gen_data.npy"), gen_data)

print(f"Datasets et modèle sauvegardés dans : {path}")

# Calcul du temps
durée_train = fin_train - debut_train
durée_inf = fin_inf - debut_inf

# Formatage en minutes:secondes
m_train, s_train = divmod(durée_train, 60)
m_inf, s_inf = divmod(durée_inf, 60)

print("=" * 60)
print("[Evaluation de la durée d'entrainement et d'inférece]")
print(f"Temps d'entraînement : {int(m_train)} min {int(s_train)} s (Total: {durée_train:.2f} secondes)")
print(f"Temps d'inférence    : {int(m_inf)} min {int(s_inf)} s (Total: {durée_inf:.2f} secondes)")
print(f"Temps total du run   : {int((durée_train + durée_inf) // 60)} min {int((durée_train + durée_inf) % 60)} s")
print("=" * 60)

#############################################
# EVALUATION
#############################################

# Clip de la SpO2
gen_data[:,:,5] = np.clip(gen_data[:,:,5], a_min=0.0, a_max=100.0)

# Arrondi des événements
gen_data[:,:,7] = np.round(gen_data[:, :, 7])

evaluator = DatasetEvaluator(
    real_data, 
    gen_data,
    col_names=["FC", "PAS", "PAM", "PAD", "Temp", "SpO2", "FR", "event_code"],
    path_dir=path
)

evaluator.run_full_analysis()