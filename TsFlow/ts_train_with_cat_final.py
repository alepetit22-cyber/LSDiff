############
# TSFlow   #
############

###########################
# Author: Adrien Le Petit #
# Release: June 2026      #
###########################

import os
import re
import sys
import json
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm
import pytorch_lightning as pl
from gluonts.torch.batchify import batchify
from tsflow.callback import EvaluateCallback
from tsflow.utils.util import create_splitter
from sklearn.preprocessing import LabelEncoder
from gluonts.dataset.common import ListDataset
from tsflow.model.tsflow_cond import TSFlowCond
from gluonts.dataset.field_names import FieldName
from tsflow.utils.transforms import AddMeanFeature
from gluonts.dataset.loader import TrainDataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.transform import Chain, AddObservedValuesIndicator, AddTimeFeatures
from gluonts.transform.split import InstanceSplitter
from gluonts.transform.sampler import TestSplitSampler
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.transform.sampler import ExpectedNumInstanceSampler
from gluonts.dataset.field_names import FieldName

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


sys.path.append("../")
from Data.metrics import DatasetEvaluator 
from Data.utils import analyser_evenements_dataset

os.environ["PT_HPU_LAZY_MODE"] = "0" 
torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = True

print(f"GPU disponible: {torch.cuda.is_available()}")

# Fonction de fenêtragre
def create_windows(data_raw, history_len, pred_len):
    """
    Découpe les données brutes des patients en fenêtres glissantes de taille fixe,
    avec un padding au début pour que l'historique puisse démarrer à -history_len.
    """
    window_size = history_len + pred_len
    windowed_data = []

    for patient in data_raw:
        # Conversion en array numpy
        data_np = np.array(patient['donnees'])
        total_len = data_np.shape[0]
        num_features = data_np.shape[1]
                
        # Boucle de génération des fenêtres glissantes
        for start_idx in range(-history_len, total_len - window_size + 1):
            
            # Initialisation d'une matrice vide pour la fenêtre
            window = np.zeros((window_size, num_features), dtype=data_np.dtype)
            
            if start_idx < 0:
                nb_pads = abs(start_idx)
                end_idx = min(start_idx + window_size, total_len)
                real_part = data_np[0 : end_idx, :]
                
                # Insertion des vraies données à la suite du padding
                window[nb_pads : nb_pads + len(real_part), :] = real_part
            else:
                end_idx = min(start_idx + window_size, total_len)
                real_part = data_np[start_idx : end_idx, :]
                
                window[0 : len(real_part), :] = real_part
                
            windowed_patient = {
                "patient_id": patient["patient_id"],
                "metadata": patient["metadata"],
                "colonnes": patient["colonnes"],
                "donnees": window
            }
            windowed_data.append(windowed_patient)
            
    return windowed_data

# Fonction de transformation des données
def tsflow_dataset(data_raw):
    """
    Instanciation du dataset.
    """
    data_list = []
    for i, patient in enumerate(data_raw):
        data_np = np.array(patient['donnees'])
        
        # Signes vitaux + event_code (colonnes 1 à 8)
        target = data_np[:, 1:9].T 
        
        # Encodage des métadonnées
        cat_class = le_class.transform([patient['metadata']['patient_class']])[0]
        cat_surgery = le_surgery.transform([patient['metadata']['surgery_type']])[0]
        
        # Tableau de la même longueur que la série temporelle
        seq_len = target.shape[1]
        canal_class = np.full((1, seq_len), cat_class)
        canal_surgery = np.full((1, seq_len), cat_surgery)
        
        # Concaténation
        target_enrichie = np.vstack([target, canal_class, canal_surgery]).astype(np.float32)
        
        data_list.append({
            "target": target_enrichie,
            "start": pd.Timestamp("2024-01-01 00:00:00"),
            "item_id": patient['item_id']
        })
    return data_list

#####################################################
# PARSER CONFIGURATION
#####################################################
def parse_args():
    parser = argparse.ArgumentParser(description='TimeGrad Training Script')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Number of hidden dimension')
    parser.add_argument('--num_steps', type=int, default=16, help='Number of denoising steps')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for optimizer')
    parser.add_argument('--gamma', type=float, default=0.05, help='Gamma parameter for prior')
    parser.add_argument('--cce_weight', type=float, default=0.1, help='Weight for CCE loss')


    args=parser.parse_args()
    return args

args = parse_args()
hidden_dim = args.hidden_dim
num_steps = args.num_steps
batch_size = args.batch_size
weight_decay = args.weight_decay
gamma = args.gamma
cce_weight = args.cce_weight


#####################################################
# DATASET INITIALIZATION
#####################################################

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
    
# Initialisation de la longueur des données
prediction_len = 16
history_len = 32 
window_len = history_len + prediction_len

# Configuration du dossier de sauvegarde
path_dir = f"checkpoints_pred{prediction_len}_hist{history_len}_hd{hidden_dim}_ns{num_steps}_bs{batch_size}_wd{weight_decay}_g{gamma}_cce{args.cce_weight}/"
os.makedirs(path_dir, exist_ok=True)
print("=" * 60)
print(f"[Dossier de sauvegarde] : {path_dir}")

#####################################################
# PARAMÉTRAGE DES DATASETS
#####################################################
# Découpage des datasets
data_train = create_windows(data_train, history_len, prediction_len)
data_val = create_windows(data_val, history_len, prediction_len)
data_test = create_windows(data_test, history_len, prediction_len)

# Mélange aléatoire du dataset
random.seed(42)
random.shuffle(data_train)
random.shuffle(data_val)
random.shuffle(data_test)

# Préparation des encodeurs pour les métadonnées
patient_classes = sorted(list(set(p['metadata']['patient_class'] for p in data)))
surgery_types = sorted(list(set(p['metadata']['surgery_type'] for p in data)))

le_class = LabelEncoder().fit(patient_classes)
le_surgery = LabelEncoder().fit(surgery_types)

print("=" * 60)
print("[Paramètres du dataset]")
print(f"Nombre de classes de patients uniques : {len(patient_classes)}")
print(f"Nombre de types de chirurgies uniques : {len(surgery_types)}")


print("=" * 60)
print("[Analyse des événements]")
analyser_evenements_dataset(data_train, "Train")
analyser_evenements_dataset(data_val, "Val")
analyser_evenements_dataset(data_test, "Test")
print("=" * 60)


# Configuration des index patients
for d in (data_train, data_val, data_test):
    for global_id, patient in enumerate(d):
        patient['item_id'] = global_id


# Création des datasets
train_data = ListDataset(tsflow_dataset(data_train), freq="30s", one_dim_target=False)
val_data   = ListDataset(tsflow_dataset(data_val), freq="30s", one_dim_target=False)
test_data  = ListDataset(tsflow_dataset(data_test), freq="30s", one_dim_target=False)

print(f"Nombre de patients Train: {len(train_data)}")
print(f"Nombre de patients Val: {len(val_data)}")
print(f"Nombre de patients Test: {len(test_data)}")
print("=" * 60)


#####################################################
# MODEL CONFIGURATION
#####################################################
config = {
    "setting": "multivariate",
    "target_dim": 10,
    "event_dim": 8,                            # Dimension de l'embedding de la variable catégorielle

    "backbone_params":{
        "input_dim": 1, # Important: Essai à 10 mais 1 normalement # 7 signes vitaux + 1 event_code + 2 métadonnées
        "output_dim": 10,                      # Cohérent avec input_dim
        "step_emb": 64,                        # Par défaut 64
        "num_residual_blocks": 4,              # Par défaut 3
        "residual_block": "s4",                # Pas le choix de l'implémentation, à laisser par défaut
        "hidden_dim": hidden_dim,              # Par défaut 64 --> Faire varier à [64, 128, 256]
        "dropout": 0.1,                        # Par défaut 0.0 --> Définir 0.1
        "init_skip": False,                    # Par défaut False
        "feature_skip": True                   # Par défaut True
    },

    "context_length": history_len,
    "prediction_length": prediction_len,
    "frequency": "30s",                        # Par défaut H --> Non approprié pour notre dataset
    "normalization": "mean",                   # Par défaut longMean --> Normalisation par StdScaler$ pour les variables continues

    "use_ema": True,                           # Par défaut True
    "use_lags": False,                         # Par défaut True

    "num_steps": 16,                           # Par défaut 16 --> Par curiosité, 32 pour évaluer l'impact sur la performance du modèle
    "solver": "euler",                         # Par défaut euler
    "matching": "random",                      # Par défaut random --> Par curiosité, ot pour évaluer l'impact sur la performance du modèle

    "optimizer_params":{ 
        "lr": 5.e-4,                           # Par défaut 1.e-3
        "weight_decay": weight_decay           # Par défaut 0 --> Essayer de définir à 1.e-4 pour voir l'impact sur la performance du modèle
    },

    "prior_params":{
        "kernel": "ou",                        # Par défaut ou
        "gamma": gamma,                        # Par défaut 1 --> Faire varier à [0.1, 1, 10] pour évaluer l'impact sur la performance du modèle
        "context_freqs": 2                     # Définir à 2 : context_length * context_freqs = history_length
    },                

    "ema_params":{ 
        "beta": 0.9999,                        # Par défaut 0.9999
        "update_after_step": 5,                # Délais de démarrage pour l'EMA, par défaut 128 --> A adapter selon le nombre d'époques
        "update_every": 1                      # Par défaut 1
    },

    "cce_weight": cce_weight,                  # Pondération de la CCE

    "dataset_params":{                         # Dataset custom
        "num_batches_per_epoch": 50,         # Par défaut 128 --> Adapter pour voir chaque patient à chaque époque
        "batch_size": batch_size                      # Par défaut 64 --> Défini [64,128,256]
    },
    
    "trainer_params":{
        "gradient_clip_val": 0.5,              # Par défaut 0.5 --> Permet de définir la valeur maximale du gradient pour éviter l'explosion du gradient
        "max_epochs": 40,                      # Par défaut 400 --> Adapter selon les performances du modèle
        "num_sanity_val_steps": 0,              # Par défaut 0 --> Vérification de la boucle val en amont de l'entraînement, pas indispensable
        "log_every_n_steps": 2
    },
        
    "evaluation_params":{
        "num_samples": 16,                     # Par défaut 16 --> Nombre d'échantillons pour l'évaluation du modèle
        "use_validation_set": True,            # Par défaut True --> Utiliser le set de validation pour l'évaluation
        "eval_every": 5,                       # Par défaut 20 --> Evaluer tous les 20 epochs pour optimisation de l'entrainement
        "do_final_eval": True                  # Par défaut True --> Rédige une évaluation finale du meilleur modèle
        },   
    
    "seed": 42
}

# Initialisation du modèle Lightning
model = TSFlowCond(
    setting=config["setting"],
    target_dim=config["target_dim"],  # 7 signes vitaux + 1 event_code + 2 métadonnées
    event_dim=config["event_dim"],
    context_length=config["context_length"],
    prediction_length=config["prediction_length"],
    frequency=config["frequency"],
    normalization=config["normalization"],
    use_ema=config["use_ema"],
    use_lags=config["use_lags"],
    num_steps=config["num_steps"],
    solver=config["solver"],
    matching=config["matching"],
    backbone_params=config["backbone_params"],
    prior_params=config["prior_params"],
    optimizer_params=config["optimizer_params"],
    ema_params=config["ema_params"],
    cce_weight=config["cce_weight"],
)

#####################################################
# PARAMETRAGE DU DATASET
#####################################################

# Préparation du Splitter
training_splitter = create_splitter(
    past_length=config["context_length"],
    future_length=config["prediction_length"],
    mode="train"
)


max_lag = max(model.lags_seq) if model.use_lags else 0

training_splitter = InstanceSplitter(
    target_field="target",
    is_pad_field="is_pad",
    start_field="start",
    forecast_start_field="forecast_start",
    instance_sampler=ExpectedNumInstanceSampler(
        num_instances=1,
        min_past=config["context_length"], 
        min_future=config["prediction_length"]
    ),
    past_length=config["context_length"] + max_lag,  
    future_length=config["prediction_length"],
    dummy_value=0.0, # Padding avec des zéros pour les données manquantes
    time_series_fields=["time_feat", "observed_values"], 
)

transform_pipeline = Chain([
    AddObservedValuesIndicator(
        target_field=FieldName.TARGET,
        output_field=FieldName.OBSERVED_VALUES,
    ),
    AddTimeFeatures(
        start_field=FieldName.START,
        target_field=FieldName.TARGET,
        output_field=FieldName.FEAT_TIME,
        time_features=time_features_from_frequency_str(config["frequency"]), # Utilisera "30s"
        pred_length=config["prediction_length"],
    ),
    AddMeanFeature(
        target_field=FieldName.TARGET,
        output_field="mean",
        train_length=1,
        setting="multivariate"
    ),
    training_splitter
])

# Création du DataLoader
data_loader = TrainDataLoader(
    train_data, 
    batch_size=config["dataset_params"]["batch_size"], 
    stack_fn=batchify, 
    transform=transform_pipeline,
    num_batches_per_epoch=config["dataset_params"]["num_batches_per_epoch"]
)

feature_pipeline = Chain([
    AddObservedValuesIndicator(
        target_field="target",
        output_field="observed_values",
    ),
    AddTimeFeatures(
        start_field="start",
        target_field="target",
        output_field="time_feat",
        time_features=time_features_from_frequency_str(config["frequency"]),
        pred_length=config["prediction_length"],
    ),
    AddMeanFeature(
        target_field=FieldName.TARGET,
        output_field="mean",
        train_length=1,
        setting="multivariate"
    ),
])

# On applique le pipeline de features sur le dataset de validation
_ = list(feature_pipeline.apply(val_data, is_train=True))
transformed_valdata = list(feature_pipeline.apply(val_data, is_train=False))

# 2. On configure l'EvaluateCallback de TSFlow
eval_callback = EvaluateCallback(
    context_length=config["context_length"],
    prediction_length=config["prediction_length"],
    model=model,
    check_val_every_n_epoch=5,
    datasets={"val": transformed_valdata},
    logdir=path_dir,
    num_samples=1
)

# 3. On configure le Checkpoint
checkpoint_callback = ModelCheckpoint(
    monitor='train_loss',       # On surveille la métrique générée par l'EvaluateCallback
    dirpath=path_dir,
    filename='tsflow_best_{epoch:02d}_{val_CRPS:.4f}',
    save_top_k=2,
    mode='min',
    save_last=True
)
# Affichage de la loss
class LossPrintCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Récupération de la perte moyenne calculée sur l'époque
        train_loss = trainer.callback_metrics.get("train_loss")
        train_mse = trainer.callback_metrics.get("train_mse")
        train_cce = trainer.callback_metrics.get("train_cce")

        if train_loss is not None:
            print(f"\n[Époque {(trainer.current_epoch + 1):02d}/{trainer.max_epochs:02d}] - loss: {train_loss:.5f} / mse: {train_mse:.5f} / cce: {train_cce:.5f}")
        else:
            print(f"\n[Époque {(trainer.current_epoch + 1):02d}/{trainer.max_epochs:02d}] - train_loss: non disponible")

# Initialisation du callback de print
print_loss_callback = LossPrintCallback()

# Configuration du Trainer (PyTorch Lightning)
trainer = pl.Trainer(
    max_epochs=config["trainer_params"]["max_epochs"],
    accelerator="auto",
    callbacks=[checkpoint_callback, print_loss_callback],
    limit_train_batches=config["dataset_params"]["num_batches_per_epoch"],
    gradient_clip_val=config["trainer_params"]["gradient_clip_val"],
    devices=1 if torch.cuda.is_available() else None,
    log_every_n_steps=1,
    enable_progress_bar=True,
)

#####################################################
# ENTRAINEMENT DU MODÈLE
#####################################################
print("\n" + "="*60)
print("[Configuration du modèle]")
print("="*60)

for key, value in config.items():
    if isinstance(value, dict):
        print(f"\n[{key}]")
        for sub_key, sub_value in value.items():
            print(f"  {sub_key:<25} : {sub_value}")
    else:
        print(f"{key:<27} : {value}")

print("=" * 60)
print("[Début de l'entraînement]")
print("=" * 60)

debut_train = time.time()
print(f"Heure de début de l'entraînement : {time.strftime('%Y-%m-%d %H:%M:%S')}")
trainer.fit(model, data_loader)
fin_train = time.time()
print("[Entraînement terminé]")
print(f"Heure de fin de l'entraînement : {time.strftime('%Y-%m-%d %H:%M:%S')}")

#####################################################
# INFÉRENCE
#####################################################

# Prétraitement des données
inference_ds = list(feature_pipeline.apply(test_data, is_train=False))

# Création du Splitter pour l'inférence
max_lag = max(model.lags_seq) if model.use_lags else 0

inference_splitter = InstanceSplitter(
    target_field="target",
    is_pad_field="is_pad",
    start_field="start",
    forecast_start_field="forecast_start",
    instance_sampler=TestSplitSampler(),
    past_length=config["context_length"] + max_lag, 
    future_length=config["prediction_length"],
    dummy_value=0.0,
    time_series_fields=["time_feat", "observed_values"] # "mean" exclu
)

# Création du Predictor
predictor = model.get_predictor(
    input_transform=inference_splitter,
    batch_size=config["dataset_params"]["batch_size"],
    device="cuda:0" if torch.cuda.is_available() else "cpu" # "cpu" si plus de VRAM
)

# Génération des prédictions
forecast_it, ts_it = make_evaluation_predictions(
    dataset=inference_ds[:500],
    predictor=predictor,
    num_samples=3 # Nombre de scénarios générés par patient
)

print("=" * 60)
print("[Début de l'inférence]")
print("=" * 60)
debut_inf = time.time()
print(f"Heure de début de l'inférence : {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Inférence
forecasts = list(tqdm(forecast_it, total=500))
tss = list(ts_it)

# Extraction et Formatage des données
real_list = []
gen_list = []

for ts, forecast in zip(tss, forecasts):
    # Les vraies valeurs (les 64 derniers pas de la fin, 8 premières colonnes)
    real_target = ts.values[-config["prediction_length"]:, :8]
    real_list.append(real_target)
    
    # Médiane pour les variables continues
    gen_cont = np.median(forecast.samples[:, :, :7], axis=0)
    
    # Mode pour la variable catégorielle (canal 7)
    gen_evt, _ = stats.mode(forecast.samples[:, :, 7], axis=0, keepdims=False)
    
    # Recombinaison
    gen_target = np.column_stack((gen_cont, gen_evt))
    gen_list.append(gen_target)

real_data = np.array(real_list).astype(np.float32)
gen_data = np.array(gen_list).astype(np.float32)

fin_inf = time.time()
print("[Inférence terminée]")
print(f"Heure de fin de l'inférence : {time.strftime('%Y-%m-%d %H:%M:%S')}")

print(f"Format de real_data : {real_data.shape}  (Batch, Temporel, Constantes)")
print(f"Format de gen_data  : {gen_data.shape}  (Batch, Temporel, Constantes)")


# sauvegarde des données
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
gen_pred = gen_data.copy()
# Clip de la SpO2
gen_pred[:,:,5] = np.clip(gen_data[:,:,5], a_min=0.0, a_max=100.0)

# Arrondi des événements
gen_pred[:,:,7] = np.round(gen_pred[:, :, 7])

evaluator = DatasetEvaluator(
    real_data, 
    gen_pred,
    col_names=["FC", "PAS", "PAM", "PAD", "Temp", "SpO2", "FR", "event_code"],
    path_dir=path_dir
)

evaluator.run_full_analysis()