import os
import sys
import json
import logging
import argparse
import subprocess
from config import config
    
# Récupération des arguments CLI
parser = argparse.ArgumentParser(description="Pipeline LSDiff")
parser.add_argument("--vae",      type=int, default=1, help="Entraîner le VAE principal")
parser.add_argument("--hist_vae", type=int, default=1, help="Entraîner le HistVAE")
parser.add_argument("--dit",      type=int, default=1, help="Entraîner le DiT")
parser.add_argument("--config",   type=str, default=None, help="Chemin du fichier config.yaml")
cli_args = parser.parse_args()

VAE     = cli_args.vae
HistVAE = cli_args.hist_vae
DiT     = cli_args.dit




# --- Configuration du Logging ---
LOG_DIR = f"logs_m{config.autoencoder.seq_len}_h{config.history_autoencoder.seq_len}"
os.makedirs(LOG_DIR, exist_ok=True)

# On configure les handlers proprement pour pouvoir les "flush" manuellement
log_file_path = os.path.join(LOG_DIR, "pipeline.log")
file_handler = logging.FileHandler(log_file_path)
stream_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[file_handler, stream_handler]
)


def load_config():
    """Charge le fichier config.yaml via config.py."""
    from config import config
    return config

def log_params(config, etape):
    """Extrait et logge dynamiquement les paramètres depuis l'objet config."""
    logging.info(f"--- PARAMÈTRES D'ENTRAÎNEMENT : {etape.upper()} ---")

    config_dict = config.model_dump()

    # 1. On affiche toujours les infos du dataset
    logging.info("  [DATASET]")
    for k, v in config_dict.get('dataset', {}).items():
        logging.info(f"    - {k:<25}: {v}")

    # 2. Paramètres d'entraînement globaux (en filtrant ce qui ne sert pas)
    logging.info("  [TRAINING]")
    for k, v in config_dict.get('training', {}).items():
        # Petite astuce pour ne pas afficher le LR de la diffusion pendant le VAE (et inversement)
        if etape == "vae" and "diffusion" in k:
            continue
        if etape == "diffusion" and "vae" in k:
            continue
        logging.info(f"    - {k:<25}: {v}")

    # 3. Paramètres spécifiques au modèle en cours
    if etape == "vae":
        logging.info("  [AUTOENCODER]")
        for k, v in config_dict.get('autoencoder', {}).items():
            logging.info(f"    - {k:<25}: {v}")

    elif etape == "history_autoencoder":
        logging.info("  [HISTORY AUTOENCODER]")
        for k, v in config_dict.get('history_autoencoder', {}).items():
            logging.info(f"    - {k:<25}: {v}")

    elif etape == "diffusion":
        logging.info("  [DIFFUSION]")
        for k, v in config_dict.get('diffusion', {}).items():
            logging.info(f"    - {k:<25}: {v}")

    logging.info("-" * 60)

def run_script(script_name, args=None):
    """
    Exécute un script Python et logge sa sortie en temps réel.
    """
    args_str = " ".join(args) if args else ""
    logging.info(f">>> Lancement de : {script_name} {args_str}")
    
    cmd = [sys.executable, "-u", script_name]
    if args:
        cmd.extend(args)
        
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        if line:
            logging.info(f"[{script_name}] {line}")
            for handler in logging.root.handlers:
                handler.flush()
        
    process.wait()
    
    if process.returncode != 0:
        logging.error(f"Erreur critique dans {script_name} (Code: {process.returncode})")
        return False
    
    logging.info(f"Terminé avec succès : {script_name}")
    return True


def main():
    logging.info("=== DÉBUT DE LA PIPELINE DE DIFFUSION CLINIQUE ===")

    if not os.path.exists("config.yaml"):
        logging.error("Fichier config.yaml introuvable. Annulation.")
        return

    # Chargement de la configuration
    config = load_config()

    # --- Étape 1 : VAE ---
    if VAE:
        logging.info("Étape 1/3 : Entraînement de l'Auto-encodeur (VAE)")
        log_params(config, "vae")
        if not run_script("train_VAE.py", args=["--mode", "main", "--config", "config.yaml"]):
            logging.error("La pipeline s'est arrêtée à l'étape du VAE.")
            return

    # --- Étape 2 : HistVAE ---
    if HistVAE:
        logging.info("Étape 2/3 : Entraînement de l'HistVAE")
        log_params(config, "history_autoencoder")
        if not run_script("train_VAE.py", args=["--mode", "history", "--config", "config.yaml"]):
            logging.error("La pipeline s'est arrêtée à l'étape de l'HistVAE.")
            return

    # --- Étape 3 : Diffusion ---
    if DiT:
        logging.info("Étape 3/3 : Entraînement du moteur de Diffusion (Transformer)")
        log_params(config, "diffusion")
        if not run_script("train_DiT.py", args=["--config", "config.yaml"]):
            logging.error("La pipeline s'est arrêtée à l'étape de la Diffusion.")
            return

    logging.info("=== PIPELINE TERMINÉE AVEC SUCCÈS ===")
    logging.info("Les modèles sont disponibles dans le dossier 'checkpoints/'.")

if __name__ == "__main__":
    main()