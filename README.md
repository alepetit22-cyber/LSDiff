# LSDiff
Study on the creation and adaptation of machine learning models leveraging the diffusion principle for time series.

**Developed model:**
* LSDiFF

**Reference models / Baselines:**
* TimeGrad
* CSDI
* TSFlow
* ADiff4TPP

**Author:** Le Petit Adrien  
**Release:** 2026

# LSDiff: Latent Diffusion Model for Clinical Time Series Generation

## 📋 Overview

LSDiff is a two-stage generative framework designed for synthesizing patient vital signs and clinical event sequences:

1. **Stage 1 — Variational Autoencoders (VAEs):**
   - A **Main VAE** compresses the target vital-sign window into a compact latent space.
   - A **History VAE** encodes the preceding clinical history into a separate latent representation.
   - Both VAEs support **mixed inputs** via either duplication-based soft-encoding or learned embeddings for categorical variables.

2. **Stage 2 — Diffusion Transformer (DiT):**
   - A **Flow Matching** diffusion model operates in the latent space.
   - Conditioned on:
     - **Clinical history** (via HistVAE latents)
     - **Patient metadata** (heterogeneous categorical / continuous features)
     - **Clinical event sequences** (via Perceiver-style cross-attention resampling)
   - Trained with **Classifier-Free Guidance (CFG)** for controllable generation.

The pipeline also includes a full evaluation suite (fidelity, diversity, and class-imbalance metrics).

---

## 🏗️ Architecture

```
                    ┌─────────────────────┐
   Raw Vitals ────► │   Main VAE (1D)     │ ──►  z_target  ──┐
                    └─────────────────────┘                  │
                                                             ▼
   History   ────►  ┌─────────────────────┐               ┌──────────────┐
   Vitals           │   History VAE (1D)  │ ──►  z_hist ─►│  Diffusion   │──► z_gen ──► Decoder ──► Synthetic
                    └─────────────────────┘               │  Transformer │                          Vitals
   Metadata  ────────────────────────────────────────────►│  (DiT + CFG) │
   Events (optional)  ───────────────────────────────────►└──────────────┘
```

### Key Components

| Module | Description |
|---|---|
| `autoencoder.py` | 1D VAE with residual blocks, self-attention, and spectral normalization |
| `diffusion_engine.py` | Flow Matching engine, DiT backbone, CFG wrapper, metadata encoder |
| `transformer_block.py` | DiT block with AdaLN-Zero, RoPE, FlashAttention, cross-attention |
| `loss_functions.py` | Hybrid VAE loss (MSE + CCE + KLD + Spectral), Flow Matching loss, SWD |
| `dataset.py` | Windowing, mixed-type normalization, categorical encoding/permutations |
| `metrics.py` | Full evaluation: MSE, MAE, R², JS, KL, MMD, F1, G-Mean, AUC-ROC/PR |
| `pipeline.py` | End-to-end training orchestrator |

---

## ✨ Features

- **Mixed-type input handling** — continuous, discrete, and categorical variables in a unified tensor
- **Two categorical encoding modes**: `duplicated` (soft-permutation encoding) and `embedded` (learned embeddings + cross-entropy head)
- **Flow Matching** with logit-normal timestep sampling and a **3rd-order Adams–Bashforth** non-uniform ODE solver
- **Classifier-Free Guidance** on history, metadata, and event streams independently
- **Event conditioning** (optional) via Perceiver-style latent query resampling
- **EMA** weights for the diffusion model
- **Comprehensive evaluation suite** with class-imbalance-aware metrics

---

## 📦 Installation

```bash
git clone https://github.com/<your-username>/LSDiff.git
cd LSDiff
conda env create -f environment.yml
conda activate lsdiff
```

---

## ⚙️ Configuration

All hyperparameters live in `config.yaml`. Example:

```yaml
dataset:
  json_path: "data/patients.json"
  target_len: 48
  hist_len: 24
  num_features: 8
  normalization: "standard"        # quantile | standard | minmax | robust
  cat_mode: "embedded"            # duplicated | embedded
  features_indices:
    - type: continuous
      index: [0, 1, 2, 3, 4, 5, 6]
    - type: categorical
      index: [7]
      cat_embed_dim: 4
      cat_seed: 42
  event_code_index: [7]
  meta_config:
    - name: age
      type: continuous
      num_classes: 1
    - name: sex
      type: categorical
      num_classes: 3

autoencoder:
  latent_channel: 16
  seq_len: 48
  stride: 2
  enc_hidden_dims: [128, 256]
  dec_hidden_dims: [256, 128, 64]
  num_groups: 8
  dropout: 0.1
  num_heads: 4
  # ...

diffusion:
  num_layers: 6
  embed_dim: 256
  num_heads: 8
  ff_mult: 4
  num_inference_steps: 15
  cfg_scale: 2.5
  cfg_dropout: 0.1
  # ...

training:
  batch_size: 64
  lr_vae: 1.0e-4
  lr_diffusion: 1.0e-4
  epochs_vae: 200
  epochs_diffusion: 500
  device: "cuda"
```

Configuration is validated at runtime via **Pydantic v2** (`config.py`), which enforces:
- `stride ∈ {2, 4, 8}`
- `seq_len % stride² == 0`

---

## 🚀 Usage

### Full Pipeline

```bash
python pipeline.py --vae 1 --hist_vae 1 --dit 1 --config config.yaml
```

Each flag can be set to `0` to skip a stage (e.g., resume from a pre-trained VAE).

### Individual Training

**Main VAE:**
```bash
python train_VAE.py --mode main --config config.yaml
```

**History VAE:**
```bash
python train_VAE.py --mode history --config config.yaml
```

**Diffusion Transformer:**
```bash
python train_DiT.py --config config.yaml
```

---

## 📊 Data Format

The dataset expects a JSON file with a list of patient records:

```json
[
  {
    "patient_id": "P0001",
   "metadata": {
      "patient_class": "19_arteritique",
      "surgery_type": "LMMC002_choc_ana_inguinal_hernia_laparoscopic_prothesis",
      "duree_totale_min": 83.0
    },
    "donnees": [
      [80.0, 120.0, 90.0, 70.0, 37.1, 98.0, 16.0, 0],
      [82.0, 118.0, 88.0, 72.0, 37.2, 97.0, 15.0, 0],
      ...
    ]
  }
]
```

- `donnees` is a 2D array `[seq_len, num_features]`
- Column indices for continuous / discrete / categorical variables are configured via `features_indices`

---

## 📈 Evaluation

The `DatasetEvaluator` class (`metrics.py`) produces a full report saved to `evaluation.txt`, including:

**Continuous variables:**
- MSE, MAE, MAPE, SMAPE, R²
- Jensen–Shannon divergence, KL divergence, MMD
- Distribution overlap histograms

**Categorical / event variables:**
- Global & minority-class accuracy
- Macro / weighted F1-score
- Geometric F-Score (FSG) — strict & weighted
- G-Mean (strict & smoothed)
- Macro / weighted AUC-ROC
- Macro AUC-PR

---

## 📁 Project Structure

```
LSDiff/
├── autoencoder.py          # VAE1D with residual + attention blocks
├── config.py               # Pydantic config loader
├── dataset.py              # PatientDataset, collate_fn, latent scaling
├── diffusion_engine.py     # DiT, CFG wrapper, Flow Matching scheduler
├── transformer_block.py    # DiTBlock, RoPE, time embeddings
├── loss_functions.py       # VAE loss, Flow Matching loss, SWD
├── metrics.py              # DatasetEvaluator
├── pipeline.py             # End-to-end orchestrator
├── train_VAE.py            # VAE training entry point
├── train_DiT.py            # Diffusion training entry point
└── config.yaml             # Hyperparameters
```

---

## 🙏 Acknowledgements

- Flow Matching formulation inspired by [Lipman et al., 2023](https://arxiv.org/abs/2210.02747)
- DiT architecture adapted from [Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748)