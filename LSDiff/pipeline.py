import os
import sys
import json
import logging
import argparse
import subprocess
from config import config
    
# CLI arguments
parser = argparse.ArgumentParser(description="Pipeline LSDiff")
parser.add_argument("--vae",      type=int, default=1, help="Entraîner le VAE principal")
parser.add_argument("--hist_vae", type=int, default=1, help="Entraîner le HistVAE")
parser.add_argument("--dit",      type=int, default=1, help="Entraîner le DiT")
parser.add_argument("--config",   type=str, default=None, help="Chemin du fichier config.yaml")
cli_args = parser.parse_args()

VAE     = cli_args.vae
HistVAE = cli_args.hist_vae
DiT     = cli_args.dit




# --- Logging Configuration ---
LOG_DIR = f"logs_m{config.autoencoder.seq_len}_h{config.history_autoencoder.seq_len}"
os.makedirs(LOG_DIR, exist_ok=True)

log_file_path = os.path.join(LOG_DIR, "pipeline.log")
file_handler = logging.FileHandler(log_file_path)
stream_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[file_handler, stream_handler]
)


def load_config():
    """Load the config.yaml file via config.py."""
    from config import config
    return config

def log_params(config, etape):
    """Extract and log dynamically the parameters from the config object."""
    logging.info(f"--- Training parameters for {etape.upper()} ---")

    config_dict = config.model_dump()

    # Dataset information
    logging.info("  [DATASET]")
    for k, v in config_dict.get('dataset', {}).items():
        logging.info(f"    - {k:<25}: {v}")

    # Global training parameters
    logging.info("  [TRAINING]")
    for k, v in config_dict.get('training', {}).items():
        if etape == "vae" and "diffusion" in k:
            continue
        if etape == "diffusion" and "vae" in k:
            continue
        logging.info(f"    - {k:<25}: {v}")

    # Model's parameters
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
    Execute a Python script and log its output in real-time.
    """
    args_str = " ".join(args) if args else ""
    logging.info(f">>> Running : {script_name} {args_str}")
    
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
        logging.error(f"Critical error in {script_name} (Code: {process.returncode})")
        return False
    
    logging.info(f"Completed successfully : {script_name}")
    return True


def main():
    logging.info("=== START OF DIFFUSION PIPELINE ===")

    if not os.path.exists("config.yaml"):
        logging.error("The config.yaml file is missing.")
        return

    # Load configuration
    config = load_config()

    # --- Step 1 : VAE ---
    if VAE:
        logging.info("Step 1/3 : Training the Auto-encoder (VAE)")
        log_params(config, "vae")
        if not run_script("train_VAE.py", args=["--mode", "main", "--config", "config.yaml"]):
            logging.error("The pipeline stopped at the VAE step.")
            return

    # --- Step 2 : HistVAE ---
    if HistVAE:
        logging.info("Step 2/3 : Training the HistVAE")
        log_params(config, "history_autoencoder")
        if not run_script("train_VAE.py", args=["--mode", "history", "--config", "config.yaml"]):
            logging.error("The pipeline stopped at the HistVAE step.")
            return

    # --- Step 3 : Diffusion ---
    if DiT:
        logging.info("Step 3/3 : Training the Diffusion engine (Transformer)")
        log_params(config, "diffusion")
        if not run_script("train_DiT.py", args=["--config", "config.yaml"]):
            logging.error("The pipeline stopped at the Diffusion step.")
            return

    logging.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
    logging.info("The models are available in the 'checkpoints/' folder.")

if __name__ == "__main__":
    main()