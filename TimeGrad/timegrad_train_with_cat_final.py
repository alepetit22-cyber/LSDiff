############
# TimeGrad #
############

###########################
# Author: Adrien Le Petit #
# Release: June 2026      #
###########################

import os
import sys
import json
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from pathlib import Path
from trainer import Trainer
from collections import Counter
from gluonts.dataset.common import ListDataset
from time_grad_estimator import TimeGradEstimator
from gluonts.evaluation.backtest import make_evaluation_predictions


sys.path.append("../../")
from Data.metrics import DatasetEvaluator
from Data.utils import LogFilterProxy, create_windows, analyser_evenements_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TQDM_DISABLE"] = "1"

# Fonction de gestion des métadonnées
def preparer_donnees_patient(patient, dict_class, dict_surgery):
    # Base des données temporelles numériques
    donnees = np.array(patient["donnees"], dtype=np.float32) # Forme : (N, 9)
    nb_lignes = donnees.shape[0]
    
    # Récupération des valeurs textuelles
    text_class = patient["metadata"]["patient_class"]
    text_surgery = patient["metadata"]["surgery_type"]
    
    # Conversion du texte en entier grâce aux dictionnaires
    num_class = dict_class[text_class]
    num_surgery = dict_surgery[text_surgery]
    
    # Création de colonnes de valeurs numériques répétées sur N lignes
    col_class = np.full((nb_lignes, 1), num_class, dtype=np.float32)
    col_surgery = np.full((nb_lignes, 1), num_surgery, dtype=np.float32)
    
    # Concaténation horizontale de nos colonnes float32
    donnees_completes = np.hstack((donnees, col_class, col_surgery)) # Forme : (N, 11)
    
    # Extraction et transposition
    target_matrix = donnees_completes[:, target_indices].T # Forme : (10, N)
    return target_matrix

# Application du filtre sur la sortie standard et les erreurs
sys.stdout = LogFilterProxy(sys.stdout)
sys.stderr = LogFilterProxy(sys.stderr)

#####################################################
# PARAMÉTRAGE DES VARIABLES                         #
#####################################################
def parse_args():
    parser = argparse.ArgumentParser(description='TimeGrad Training Script')
    parser.add_argument('--num_layer', type=int, default=3, help='Number of layers')
    parser.add_argument('--num_cell', type=int, default=80, help='Number of cells')
    parser.add_argument('--residual_channels', type=int, default=16, help='Number of residual channels')
    parser.add_argument('--residual_layers', type=int, default=8, help='Number of residual layers')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout_rate')
    parser.add_argument('--cce_weight', type=float, default=0.5, help='CCE weight')

    args=parser.parse_args()
    return args

args = parse_args()
num_layer = args.num_layer
num_cell = args.num_cell
residual_channels = args.residual_channels
residual_layers = args.residual_layers
batch_size = args.batch_size
dropout = args.dropout
cce_weight = args.cce_weight

# Chargement des données
chemin_fichier = '../Data/db_200.json'
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

# Initialisation de la longueur des données
prediction_len = 16
history_len = 32 
window_len = history_len + prediction_len

# Configuration du dossier de sauvegarde
path_dir = f"checkpoints_pred{prediction_len}_hist{history_len}_nl{num_layer}_nc{num_cell}_rc{residual_channels}_rl{residual_layers}_bs{batch_size}_d{dropout}_cce{cce_weight}/"
os.makedirs(path_dir, exist_ok=True)
print("=" * 60)
print(f"[Dossier de sauvegarde] : {path_dir}")

# Initialisation des variables
# 1="FC", 2="PAS", 3="PAM", 4="PAD", 5="temp", 6="SpO2", 7="FR", 8="event", 9="patient_class", 10="surgery_type"
target_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
target_dim = len(target_indices)

# Data fictive pour gluonTS
dummy_start = pd.Timestamp("2024-01-01 00:00:00")

# Découpage des datasets
data_train = create_windows(data_train, history_len, prediction_len)
data_val = create_windows(data_val, history_len, prediction_len)
data_test = create_windows(data_test, history_len, prediction_len)

# Mélange aléatoire du dataset
random.seed(42)
random.shuffle(data_train)
random.shuffle(data_val)
random.shuffle(data_test)

# Création des dictionnaires de métadonnées 
classes_uniques = sorted(list(set(patient["metadata"]["patient_class"] for patient in data_train)))
chirurgies_uniques = sorted(list(set(patient["metadata"]["surgery_type"] for patient in data_train)))

dict_patient_class = {nom: i for i, nom in enumerate(classes_uniques)}
dict_surgery_type = {nom: i for i, nom in enumerate(chirurgies_uniques)}

print("=" * 60)
print("[Paramètres du dataset]")
print(f"Nombre de classes de patients uniques : {len(dict_patient_class)}")
print(f"Nombre de types de chirurgies uniques : {len(dict_surgery_type)}")

#####################################################
# PARAMÉTRAGE DES DATASETS
#####################################################

print("=" * 60)
print("[Analyse des événements]")
analyser_evenements_dataset(data_train, "Train")
analyser_evenements_dataset(data_val, "Val")
analyser_evenements_dataset(data_test, "Test")
print("=" * 60)

# Initialisation des listes
train_ds_list = []
val_ds_list = []
test_ds_list = []

# Remplissage du Train
for patient in data_train:
    target_matrix = preparer_donnees_patient(patient, dict_patient_class, dict_surgery_type)
    if target_matrix.shape[1] == window_len:
        train_ds_list.append({
            "target": target_matrix[0:8, :],
            "start": dummy_start,
            "item_id": patient["patient_id"],
            "feat_static_cat": [int(target_matrix[8, 0]), int(target_matrix[9, 0])]
        })

# Remplissage du Val
for patient in data_val:
    target_matrix = preparer_donnees_patient(patient, dict_patient_class, dict_surgery_type)
    if target_matrix.shape[1] == window_len:
        val_ds_list.append({
            "target": target_matrix[0:8, :],
            "start": dummy_start,
            "item_id": patient["patient_id"],
            "feat_static_cat": [int(target_matrix[8, 0]), int(target_matrix[9, 0])]
        })

# Remplissage du Test
for patient in data_test:
    target_matrix = preparer_donnees_patient(patient, dict_patient_class, dict_surgery_type)
    if target_matrix.shape[1] == window_len:
        test_ds_list.append({
            "target": target_matrix[0:8, :],
            "start": dummy_start,
            "item_id": patient["patient_id"],
            "feat_static_cat": [int(target_matrix[8, 0]), int(target_matrix[9, 0])]
        })

# Conversion finale en ListDataset GluonTS
train_data = ListDataset(train_ds_list, freq="1min", one_dim_target=False)
val_data = ListDataset(val_ds_list, freq="1min", one_dim_target=False)
test_data = ListDataset(test_ds_list[:200], freq="1min", one_dim_target=False)

print(f"Nombre de patients Train: {len(train_data)}")
print(f"Nombre de patients Val: {len(val_data)}")
print(f"Nombre de patients Test: {len(test_data)}")

#####################################################
# CONFIGURATION DU MODÈLE
#####################################################
estimator = TimeGradEstimator(
    # Variables à prédire
    target_dim=8,
    prediction_length=prediction_len, # Définir à 16
    context_length=history_len, # Définir à 32
    conditioning_length=history_len, # Par défaut 100 --> Adapter selon context_length

    # Paramètres pour event_code
    num_event_classes=100,
    event_embed_dim=8,
    cce_weight=cce_weight,
    
    # Métadonnées
    cardinality=[30, 4],
    embedding_dimension = 5,
    
    # Paramètres du RNN
    input_size=38, # target_dim * (1+n_lags) + n_meta + n_cat_dim
    cell_type='GRU',
    freq="1min",
    num_layers=num_layer, # Par défaut 2 --> À modifier [2,3,4]
    num_cells=num_cell, # Par défaut 40 --> À modifier [40, 80, 160]
    time_features=None,
    lags_seq=[1, 2, 4],
    
    # Paramètres d'entrainement
    dropout_rate=dropout, # Défaut 0.1
    loss_type='l2', # Défaut l2
    scaling=True, # Défaut True
    diff_steps=100, # Défaut 100
    beta_end=0.1, # Défaut 0.1
    beta_schedule="linear", # Défaut "linear"
    residual_layers=residual_layers, # Défaut 8
    residual_channels=residual_channels, # Défaut 8 --> Faire varier à [16, 32, 64]
    
    # Paramètres du trainer
    trainer=Trainer(
        device=device,
        epochs=50, # Défaut 100 --> Faire varier selon les performances du modèle
        learning_rate=1e-3, # Défaut 1e-3
        num_batches_per_epoch=80, # Défaut 50 --> Adapter pour voir chaque patient à chaque époque
        batch_size=batch_size # Défaut 32 --> [64,128,256]
    )
)
# Arguments de la Grid Search
print("[Arguments de la Grid Search]")
for arg, value in vars(args).items():
    print(f"   {arg:<22}: {value}")

# 2. Paramètres internes de l'Estimator
print("[Paramètres du TimeGradEstimator]")
print(f"   target_dim            : {estimator.target_dim}")
print(f"   prediction_length     : {estimator.prediction_length}")
print(f"   context_length        : {estimator.context_length}")
print(f"   conditioning_length   : {estimator.conditioning_length}")
print(f"   num_event_classes     : {estimator.num_event_classes}")
print(f"   event_embed_dim       : {estimator.event_embed_dim}")
print(f"   cce_weight            : {estimator.cce_weight}")
print(f"   num_layers (RNN)      : {estimator.num_layers}")
print(f"   num_cells (RNN)       : {estimator.num_cells}")
print(f"   cell_type             : {estimator.cell_type}")
print(f"   residual_layers       : {estimator.residual_layers}")
print(f"   residual_channels     : {estimator.residual_channels}")
print(f"   diff_steps (Diffusion): {estimator.diff_steps}")
print(f"   loss_type             : {estimator.loss_type}")
print(f"   beta_schedule         : {estimator.beta_schedule}")
print(f"   scaling               : {estimator.scaling}")

# 3. Paramètres du Trainer
print("[Paramètres du Trainer]")
print(f"   device                : {estimator.trainer.device}")
print(f"   epochs                : {estimator.trainer.epochs}")
print(f"   batch_size            : {estimator.trainer.batch_size}")
print(f"   num_batches_per_epoch : {estimator.trainer.num_batches_per_epoch}")
print(f"   learning_rate         : {estimator.trainer.learning_rate}")
print(f"   weight_decay          : {estimator.trainer.weight_decay}")

#####################################################
# ENTRAINEMENT DU MODÈLE
#####################################################
print("=" * 60)
print("[Début de l'entraînement]")
debut_train = time.time()
print(f"Heure de début de l'entraînement : {time.strftime('%Y-%m-%d %H:%M:%S')}")
predictor = estimator.train(
    training_data=train_data,
    validation_data=val_data, 
    num_workers=0, 
    prefetch_factor=None
    )
fin_train = time.time()
print("[Entraînement terminé]")
print(f"Heure de fin de l'entraînement : {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Sauvegarde du modèle
predictor.serialize(path=Path(path_dir))
print(f"Modèle sauvegardé dans : {path_dir}")

#####################################################
# INFÉRENCE
#####################################################

print("=" * 60)
print("[Début de l'inférence]")
debut_inf = time.time()
print(f"Heure de début de l'inférence : {time.strftime('%Y-%m-%d %H:%M:%S')}")
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_data,
    predictor=predictor,
    num_samples=50
)


# Conversion des itérateurs en listes
forecasts = list(forecast_it)
tss = list(ts_it)

# Extraire la première prédiction
forecast = forecasts[0]
target = tss[0]

print(f"Dimension de la prédiction : {forecast.samples.shape}")
liste_real = []
liste_gen = []

for forecast, ts in tqdm(zip(forecasts, tss)):
    # Extraction real_data
    vrais_valeurs = ts.iloc[-prediction_len:].values
    liste_real.append(vrais_valeurs)
    
    # Extraction gen_data
    samples_continuous = forecast.samples[:, :, :7]
    samples_event = forecast.samples[:, :, 7]
    
    # Médiane pour les continues
    mediane_continue = np.median(samples_continuous, axis=0)
    
    # Mode pour les événements
    mode_result = stats.mode(samples_event, axis=0)
    mode_event = np.squeeze(mode_result.mode)
    
    prediction_hybride = np.concatenate([mediane_continue, mode_event[:, None]], axis=-1)
    liste_gen.append(prediction_hybride)
    
# Conversion en tableaux NumPy
real_data = np.array(liste_real)
gen_data = np.array(liste_gen)
fin_inf = time.time()
print("[Inférence terminée]")
print(f"Heure de fin de l'inférence : {time.strftime('%Y-%m-%d %H:%M:%S')}")

print(f"Format de real_data : {real_data.shape}  (Batch, Temporel, Constantes)")
print(f"Format de gen_data  : {gen_data.shape}  (Batch, Temporel, Constantes)")

# Enregistrement des datasets
np.save(f"{path_dir}real_data.npy", real_data)
np.save(f"{path_dir}gen_data.npy", gen_data)

print(f"Datasets et modèle sauvegardés dans : {path_dir}")

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
    path_dir=path_dir
)

evaluator.run_full_analysis()