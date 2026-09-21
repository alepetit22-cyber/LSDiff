

# --- Fichier : ./autoencoder.py ---
```py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from torch.nn.utils.parametrizations import spectral_norm

class ResBlock1D(nn.Module):
    """
    Bloc résiduel pour stabiliser l'apprentissage des signaux complexes.
    Toutes les dimensions et paramètres (kernel_size, padding, num_groups, dropout) sont configurables.
    """
    def __init__(self, channels: int, dropout: float, kernel_size: int, padding: int, num_groups: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(num_groups, channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

class AttentionBlock(nn.Module):
    """
    Attention globale pour corréler les constantes vitales sur toute la séquence.
    """
    def __init__(self, channels: int, num_heads: int, num_groups: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.GroupNorm(num_groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L] -> transpose pour MultiheadAttention [B, L, C]
        h = x.transpose(1, 2)
        attn_out, _ = self.attn(h, h, h)
        h = h + attn_out
        # Retour en [B, C, L]
        return self.norm(h.transpose(1, 2))

class MixedInputProjection(nn.Module):
    """
    Projette un batch (flottant unifié) vers la dimension attendue.
    """
    def __init__(self, cat_mode: str = "duplicated",
                cat_vocab_sizes: Optional[int] = None,
                cat_embed_dim: int = 6):
        super().__init__()
        self.cat_mode = cat_mode
        if cat_mode == "embedded" and cat_vocab_sizes:
            self.embedding = nn.ModuleList([
                nn.Embedding(vocab_size, cat_embed_dim) for vocab_size in cat_vocab_sizes
            ])
            self.cat_embed_dim = cat_embed_dim


    def forward(self, x_float: torch.Tensor, x_cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.cat_mode == "duplicated" or x_cat is None or x_cat.shape[1] == 0:
            return x_float
        
        embedded_list = []
        for i, emb_layer in enumerate(self.embedding):
            emb = emb_layer(x_cat[:, i, :]).transpose(1, 2)
            embedded_list.append(emb)
        cat_features = torch.cat(embedded_list, dim=1) # [B, Num_Cat * Embed_Dim, L]
        return torch.cat([x_float, cat_features], dim=1)

class VAE1D(nn.Module):
    """
    Variational Autoencoder 1D pour la compression des séries temporelles (mixtes).
    """
    def __init__(
        self, 
        num_input_channels: int, 
        num_continuous: int, 
        num_discrete: int,
        latent_channel: int, 
        stride: int = 2, 
        seq_len: int = 48,
        enc_hidden_dims: Optional[List[int]] = None,
        dec_hidden_dims: Optional[List[int]] = None,
        num_groups: int = 8,
        dropout: float = 0.1,
        num_heads: int = 4,
        kernel_size_stride: int = 4,
        kernel_size_res: int = 3,
        padding: int = 1,
        logvar_clip_min: float = -30.0,
        logvar_clip_max: float = 20.0,
        cat_mode: str = "duplicated",
        cat_vocab_sizes: Optional[List[int]] = None,
        cat_embed_dim: int = 6,
    ):
        super(VAE1D, self).__init__()
        
        if enc_hidden_dims is None:
            enc_hidden_dims = [128, 256]
        if dec_hidden_dims is None:
            dec_hidden_dims = [256, 128, 64]
            
        self.latent_channel = latent_channel
        self.seq_len = seq_len
        self.num_input_channels = num_input_channels
        self.num_continuous = num_continuous
        self.num_discrete = num_discrete
        
        self.logvar_clip_min = logvar_clip_min
        self.logvar_clip_max = logvar_clip_max

        self.cat_mode = cat_mode
        self.cat_vocab_sizes = cat_vocab_sizes or []
        self.input_proj = MixedInputProjection(cat_mode, cat_vocab_sizes, cat_embed_dim)
        
        total_input_ch = num_input_channels
        if cat_mode == "embedded":
            total_input_ch += len(self.cat_vocab_sizes) * cat_embed_dim
        
        # --- ENCODER ---
        self.encoder = nn.Sequential(
            # Bloc 1
            nn.Conv1d(total_input_ch, enc_hidden_dims[0], kernel_size=kernel_size_stride, stride=stride, padding=padding),
            nn.GroupNorm(num_groups, enc_hidden_dims[0]),
            nn.SiLU(),
            ResBlock1D(enc_hidden_dims[0], dropout, kernel_size_res, padding, num_groups),
            
            # Bloc 2
            nn.Conv1d(enc_hidden_dims[0], enc_hidden_dims[1], kernel_size=kernel_size_stride, stride=stride, padding=padding),
            nn.GroupNorm(num_groups, enc_hidden_dims[1]),
            nn.SiLU(),
            AttentionBlock(enc_hidden_dims[1], num_heads, num_groups),
            ResBlock1D(enc_hidden_dims[1], dropout, kernel_size_res, padding, num_groups),
            
            # Projection latente finale
            nn.Conv1d(enc_hidden_dims[1], latent_channel * 2, kernel_size=kernel_size_res, padding=padding)
        )
        
        # --- DECODER ---
        self.decoder_input = nn.Conv1d(latent_channel, dec_hidden_dims[0], kernel_size=kernel_size_res, padding=padding)
        
        self.decoder_trunk = nn.Sequential(
            # Tronc initial
            ResBlock1D(dec_hidden_dims[0], dropout, kernel_size_res, padding, num_groups),
            AttentionBlock(dec_hidden_dims[0], num_heads, num_groups),
            nn.SiLU(),
            
            # Upsampling 1
            nn.ConvTranspose1d(dec_hidden_dims[0], dec_hidden_dims[1], kernel_size=kernel_size_stride, stride=stride, padding=padding),
            nn.GroupNorm(num_groups, dec_hidden_dims[1]),
            nn.SiLU(),
            ResBlock1D(dec_hidden_dims[1], dropout, kernel_size_res, padding, num_groups),
            ResBlock1D(dec_hidden_dims[1], dropout, kernel_size_res, padding, num_groups),
            
            # Upsampling 2
            nn.ConvTranspose1d(dec_hidden_dims[1], dec_hidden_dims[2], kernel_size=kernel_size_stride, stride=stride, padding=padding),
            nn.SiLU()
        )
        
        # Tête de sortie
        self.head_output_float = nn.Conv1d(dec_hidden_dims[2], num_input_channels, kernel_size=kernel_size_res, padding=padding)
        if cat_mode == "embedded" and len(self.cat_vocab_sizes) > 0:
            self.head_cat_logits = nn.ModuleList([
                nn.Conv1d(dec_hidden_dims[2], num_classes, kernel_size=kernel_size_res, padding=padding)
                for num_classes in self.cat_vocab_sizes
            ])
        else:
            self.head_cat_logits = None

        self._apply_spectral_norm(self)

    def _apply_spectral_norm(self, module):
        for name, child in list(module.named_children()):
            if isinstance(child, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                if not isinstance(child, nn.Embedding):
                    spectral_norm(child)
            else:
                self._apply_spectral_norm(child)

    def encode(self, x_float: torch.Tensor, x_cat: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h_in = self.input_proj(x_float, x_cat)
        h = self.encoder(h_in)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparamétrisation avec clamping dynamique.
        """
        logvar = torch.clamp(logvar, min=self.logvar_clip_min, max=self.logvar_clip_max)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_input(z)
        trunk = self.decoder_trunk(h)
        recon_float = self.head_output_float(trunk)
        
        if self.cat_mode == "embedded" and self.head_cat_logits is not None:
            # Liste de logits [B, Num_Classes_i, L] pour chaque variable catégorielle
            cat_logits = [head(trunk) for head in self.head_cat_logits]
            return recon_float, cat_logits
            
        return recon_float, None

    def forward(self, x_float: torch.Tensor, x_cat: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x_float, x_cat)
        z = self.reparameterize(mu, logvar)
        recon_float, recon_cat_logits = self.decode(z)
            
        return recon_float, recon_cat_logits, mu, logvar
```


# --- Fichier : ./config.py ---
```py
import yaml
import os
from pydantic import BaseModel, field_validator, model_validator
from typing import List, Optional

class FeatureIndexConfig(BaseModel):
    type: str
    index: List[int]
    cat_embed_dim: Optional[int] = 1
    cat_seed: Optional[int] = 42

class MetaConfigItem(BaseModel):
    name: str
    type: str
    num_classes: int

class DatasetConfig(BaseModel):
    json_path: str
    target_len: int
    num_features: int
    normalization: str
    features_indices: List[FeatureIndexConfig]
    event_code_index: Optional[List[int]] = None
    cat_mode: Optional[str] = "duplicated"
    hist_len: int
    meta_config: Optional[List[MetaConfigItem]] = None

    @property
    def continuous_indices(self) -> List[int]:
        for g in self.features_indices:
            if g.type == "continuous":
                return g.index
        return []

    @property
    def discrete_indices(self) -> List[int]:
        for g in self.features_indices:
            if g.type == "discrete":
                return g.index
        return []

    @property
    def categorical_indices(self) -> List[int]:
        for g in self.features_indices:
            if g.type == "categorical":
                return g.index
        return []

    @property
    def num_continuous(self) -> int:
        return len(self.continuous_indices)

    @property
    def num_discrete(self) -> int:
        return len(self.discrete_indices)

    @property
    def num_categorical(self) -> int:
        return len(self.categorical_indices)
    
    @property
    def cat_embed_dim(self) -> int:
        for g in self.features_indices:
            if g.type == "categorical":
                return getattr(g, "cat_embed_dim", 1)
        return 1

    @property
    def cat_seed(self) -> int:
        for g in self.features_indices:
            if g.type == "categorical":
                return getattr(g, "cat_seed", 42)
        return 42
    
    @property
    def num_effective_float_channels(self) -> int:
        """
        Calcul du nombre total de canaux après la transformation 
        [continus + discrets + (catégoriels * duplication)]
        """
        if self.cat_mode == "duplicated":
            return self.num_continuous + self.num_discrete + (self.num_categorical * self.cat_embed_dim)
        else:
            return self.num_continuous + self.num_discrete


class AutoencoderConfig(BaseModel):
    latent_channel: int
    seq_len: int
    input_channels: int
    stride: int
    kld_weight: float
    spectral_weight: float
    cce_weight: float
    scaler_path: str
    checkpoint_path: str
    best_model_path: str
    enc_hidden_dims: List[int]
    dec_hidden_dims: List[int]
    num_groups: int
    dropout: float
    num_heads: int
    kernel_size_stride: int
    kernel_size_res: int
    padding: int
    logvar_clip_min: float
    logvar_clip_max: float

    @field_validator('stride')
    @classmethod
    def stride_must_be_power_of_two(cls, v):
        assert v in (2, 4, 8), f"stride={v} invalide"
        return v

    @model_validator(mode='after')
    def seq_len_divisible_by_stride(self):
        assert self.seq_len % (self.stride ** 2) == 0, \
            f"seq_len={self.seq_len} doit être divisible par stride²={self.stride**2}"
        return self

class HistoryAutoencoderConfig(BaseModel):
    latent_channel: int
    seq_len: int
    input_channels: int
    stride: int
    kld_weight: float
    spectral_weight: float
    cce_weight: float
    hist_scaler_path: str
    checkpoint_path: str
    best_model_path: str
    enc_hidden_dims: List[int]
    dec_hidden_dims: List[int]
    num_groups: int
    dropout: float
    num_heads: int
    kernel_size_stride: int
    kernel_size_res: int
    padding: int
    logvar_clip_min: float
    logvar_clip_max: float

    @field_validator('stride')
    @classmethod
    def stride_must_be_power_of_two(cls, v):
        assert v in (2, 4, 8), f"stride={v} invalide"
        return v

    @model_validator(mode='after')
    def seq_len_divisible_by_stride(self):
        assert self.seq_len % (self.stride ** 2) == 0, \
            f"seq_len={self.seq_len} doit être divisible par stride²={self.stride**2}"
        return self

class DiffusionConfig(BaseModel):
    num_layers: int
    embed_dim: int
    num_heads: int
    ff_mult: int
    dropout: float
    num_inference_steps: int
    num_samples: int
    num_classes: int
    null_class: int
    cfg_scale: float
    cfg_dropout: float
    vae_stride: int
    latent_scale_path: str
    hist_latent_scale_path: str
    checkpoint_path: str
    best_model_path: str

class TrainingConfig(BaseModel):
    batch_size: int
    lr_vae: float
    lr_diffusion: float
    epochs_vae: int
    epochs_diffusion: int
    device: str

class InferenceConfig(BaseModel):
    vae_real_path: str
    vae_gen_path: str
    vae_evaluator_path: str
    dit_real_path: str
    dit_gen_path: str
    dit_evaluator_path: str

class AppConfig(BaseModel):
    dataset: DatasetConfig
    autoencoder: AutoencoderConfig
    history_autoencoder: HistoryAutoencoderConfig
    diffusion: DiffusionConfig
    training: TrainingConfig
    inference: InferenceConfig

    @classmethod
    def load(cls, path: str = "config.yaml") -> 'AppConfig':
        if not os.path.exists(path):
            parent_path = os.path.join("..", path)
            if os.path.exists(parent_path):
                path = parent_path
            else:
                raise FileNotFoundError(f"Configuration file {path} not found.")
                
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            
        return cls(
            dataset=DatasetConfig(**data.get('dataset', {})),
            autoencoder=AutoencoderConfig(**data.get('autoencoder', {})),
            history_autoencoder=HistoryAutoencoderConfig(**data.get('history_autoencoder', {})),
            diffusion=DiffusionConfig(**data.get('diffusion', {})),
            training=TrainingConfig(**data.get('training', {})),
            inference=InferenceConfig(**data.get('inference', {}))

        )

def load_config(path: Optional[str] = None) -> 'AppConfig':
    if path is None:
        path = os.environ.get("CONFIG_PATH", "config.yaml")
    if not os.path.exists(path):
        # tentative dans le dossier parent
        parent = os.path.join("..", path)
        if os.path.exists(parent):
            path = parent
        else:
            raise FileNotFoundError(f"Configuration file {path} not found.")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return AppConfig(
        dataset=DatasetConfig(**data.get('dataset', {})),
        autoencoder=AutoencoderConfig(**data.get('autoencoder', {})),
        history_autoencoder=HistoryAutoencoderConfig(**data.get('history_autoencoder', {})),
        diffusion=DiffusionConfig(**data.get('diffusion', {})),
        training=TrainingConfig(**data.get('training', {})),
        inference=InferenceConfig(**data.get('inference', {}))
    )

# Remplacer l'appel existant par :
try:
    config = load_config()
except Exception as e:
    print(f"Erreur de chargement : {e}")
    config = None
```


# --- Fichier : ./dataset.py ---
```py
import json
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler
from typing import List, Dict, Any, Optional, Tuple


class PatientDataset(Dataset):
    """
    Dataset optimisé pour les séries temporelles cliniques de LSDiff.
    Gère les variables continues, discrètes et catégorielles.
    """

    def __init__(
        self,
        raw_data: List[Dict],
        target_len: int,
        hist_len: int,
        continuous_indices: List[int],
        discrete_indices: List[int],
        categorical_indices: List[int],
        scaler_path: str,
        fit_stats: bool = False,
        meta_config: Optional[List[Dict[str, Any]]] = None,
        cat_embed_dim: int = 1,
        cat_seed: int = 42,
        cat_mode: str = "duplicated",
        mode: str = "DiT",
        normalization: str = "standard",
        event_code_index: Optional[List[int]] = None,
    ):
        # Paramètres de base
        self.target_len = target_len
        self.hist_len = hist_len
        self.continuous_cols = continuous_indices
        self.discrete_cols = discrete_indices
        self.categorical_cols = categorical_indices

        self.num_continuous = len(continuous_indices)
        self.num_discrete = len(discrete_indices)
        self.num_categorical = len(categorical_indices)
        self.cat_embed_dim = cat_embed_dim
        self.cat_seed = cat_seed
        self.cat_mode = cat_mode

        self.mode = mode.lower()
        self.normalization = normalization

        self.event_code_index = event_code_index

        # Index de début des variables catégorielles dans le tenseur complet
        self.cat_map_start_idx = self.num_continuous + self.num_discrete

        # Chemins de sauvegarde
        self.scaler_path = scaler_path
        self.vocab_path = scaler_path.replace('_scaler.pkl', '_vocab.json')
        self.cat_vocab_path = scaler_path.replace('_scaler.pkl', '_cat_vocab.json')
        self.cat_permutation_path = scaler_path.replace('_scaler.pkl', '_cat_permutations.json')

        self.meta_config = meta_config
        self.fit_stats = fit_stats

        # Chargement des données brutes
        self.raw_data = raw_data

        # Vocabulaires et mappings de permutation
        self.meta_vocabs = self._prepare_meta_vocabs()
        self.cat_vocabs = self._prepare_cat_vocabs()

        # Génération des fenêtres
        self.windowed_data = self._generate_windows()

        # Préparation des données (extraction, normalisation)
        self.data_float_list, self.meta_list = self._prepare_data()

    # ------------------------------------------------------------------
    # Méthodes de préparation des vocabulaires
    # ------------------------------------------------------------------

    def _prepare_meta_vocabs(self) -> Dict[str, Dict[str, int]]:
        """Construit ou charge le vocabulaire des métadonnées catégorielles."""
        if not self.meta_config:
            return {}

        vocabs = {feat.name: {} for feat in self.meta_config if feat.type == "categorical"}
        
        if self.fit_stats:
            for patient in self.raw_data:
                meta_raw = patient.get("metadata", {})
                for feat in self.meta_config:
                    if feat.type == "categorical":
                        name = feat.name
                        val = str(meta_raw.get(name, "UNKNOWN"))
                        if val not in vocabs[name]:
                            vocabs[name][val] = len(vocabs[name])
            
            os.makedirs(os.path.dirname(self.vocab_path), exist_ok=True)
            with open(self.vocab_path, 'w', encoding='utf-8') as f:
                json.dump(vocabs, f)
        else:
            if os.path.exists(self.vocab_path):
                with open(self.vocab_path, 'r', encoding='utf-8') as f:
                    vocabs = json.load(f)
        return vocabs

    def _prepare_cat_vocabs(self) -> Dict[int, Dict[str, int]]:
        """
        Construit ou charge :
        - le vocabulaire des valeurs catégorielles (par colonne)
        - les mappings de permutation pour les duplicats
        Retourne uniquement le vocabulaire.
        """
        # Vocabulaire initial avec token PAD_UNKNOWN (index 99)
        vocabs = {col: {"PAD_UNKNOWN": 99} for col in self.categorical_cols}

        if self.fit_stats:
            # Construction du vocabulaire à partir des données
            for patient in self.raw_data:
                seq_raw = np.array(patient["donnees"])
                for col in self.categorical_cols:
                    unique_vals = np.unique(seq_raw[:, col])
                    for val in unique_vals:
                        val_str = str(val)
                        if val_str not in vocabs[col]:
                            vocabs[col][val_str] = len(vocabs[col])

            os.makedirs(os.path.dirname(self.cat_vocab_path), exist_ok=True)
            with open(self.cat_vocab_path, 'w', encoding='utf-8') as f:
                json.dump(vocabs, f)

            if self.cat_mode == "duplicated":
                # Construction des mappings de permutation pour le soft-encoding
                self.cat_permutations = {}
                self.cat_inv_permutations = {}
                
                for col in self.categorical_cols:
                    self.cat_permutations[col] = []
                    self.cat_inv_permutations[col] = []
                    
                    encoded_vals = list(range(len(vocabs[col])))
                    for d in range(self.cat_embed_dim):
                        rng = np.random.RandomState(self.cat_seed + col + d * 1000)
                        permuted = rng.permutation(encoded_vals).tolist()
                        
                        self.cat_permutations[col].append(
                            {int(k): int(v) for k, v in zip(encoded_vals, permuted)}
                        )
                        self.cat_inv_permutations[col].append(
                            {int(v): int(k) for k, v in zip(encoded_vals, permuted)}
                        )

                with open(self.cat_permutation_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "cat_permutations": self.cat_permutations,
                        "cat_inv_permutations": self.cat_inv_permutations
                    }, f)
                
        else:
            # Chargement depuis les fichiers existants
            if os.path.exists(self.cat_vocab_path):
                with open(self.cat_vocab_path, 'r', encoding='utf-8') as f:
                    vocabs_str = json.load(f)
                    vocabs = {int(k): v for k, v in vocabs_str.items()}

            if self.cat_mode == "duplicated":
                if os.path.exists(self.cat_permutation_path):
                    with open(self.cat_permutation_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.cat_permutations = {int(k): v for k, v in data["cat_permutations"].items()}
                        self.cat_inv_permutations = {int(k): v for k, v in data["cat_inv_permutations"].items()}
                        
                        # Conversion des clés en int pour les sous-dictionnaires
                        for col in self.categorical_cols:
                            for d in range(self.cat_embed_dim):
                                self.cat_permutations[col][d] = {
                                    int(k): int(v) for k, v in self.cat_permutations[col][d].items()
                                }
                                self.cat_inv_permutations[col][d] = {
                                    int(k): int(v) for k, v in self.cat_inv_permutations[col][d].items()
                                }
                else:
                    raise FileNotFoundError(
                        f"Fichier de permutations catégorielles introuvable : {self.cat_permutation_path}"
                    )

        return vocabs

    # ------------------------------------------------------------------
    # Découpage en fenêtres
    # ------------------------------------------------------------------

    def _generate_windows(self) -> List[Dict]:
        """
        Génère des fenêtres glissantes adaptées au mode d'entraînement choisi.
        """
        if self.mode == "main":
            window_size = self.target_len
        elif self.mode == "history":
            window_size = self.hist_len
        else: # "DiT"
            window_size = self.hist_len + self.target_len

        windowed = []

        for patient in self.raw_data:
            data_np = np.array(patient['donnees'])
            length = data_np.shape[0]
            num_features = data_np.shape[1]

            begin_idx = 0 if self.mode == "main" else -self.hist_len
            end_idx = length - window_size

            for start_idx in range(begin_idx, end_idx + 1, self.target_len//4):
                window = np.zeros((window_size, num_features), dtype=data_np.dtype)

                if start_idx < 0:
                    nb_pads = abs(start_idx)
                    real_end = min(start_idx + window_size, length)
                    if real_end > 0:
                        real_seq = data_np[0:real_end, :]
                        window[nb_pads:nb_pads + len(real_seq), :] = real_seq
                else:
                    real_end = min(start_idx + window_size, length)
                    real_seq = data_np[start_idx:real_end, :]
                    window[0:len(real_seq), :] = real_seq

                patient_virtuel = {
                    "patient_id": patient.get("patient_id", "unknown"),
                    "metadata": patient.get("metadata", {}).copy(),
                    "donnees": window
                }
                windowed.append(patient_virtuel)

        return windowed

    # ------------------------------------------------------------------
    # Extraction et normalisation des données
    # ------------------------------------------------------------------

    def _prepare_data(self) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Extrait les features (continues, discrètes, catégorielles), applique la normalisation,
        et prépare les métadonnées. Retourne la liste des séquences normalisées et la liste des métadonnées.
        """
        all_features_float = []
        all_features_cat_idx = []
        all_meta = []

        # 1. Extraction brute pour chaque fenêtre
        for patient in self.windowed_data:
            seq_raw = np.array(patient["donnees"])

            # Features continues
            # Features continues & discrètes
            features_cont = seq_raw[:, self.continuous_cols].astype(np.float32) if self.num_continuous > 0 else np.zeros((len(seq_raw), 0), dtype=np.float32)
            features_disc = seq_raw[:, self.discrete_cols].astype(np.float32) if self.num_discrete > 0 else np.zeros((len(seq_raw), 0), dtype=np.float32)

            if self.cat_mode == "duplicated":
                # Mode duplicated: Duplication via permutation
                features_cat = np.zeros((len(seq_raw), self.num_categorical * self.cat_embed_dim), dtype=np.float32)
                for i, col in enumerate(self.categorical_cols):
                    for t in range(len(seq_raw)):
                        val_str = str(seq_raw[t, col])
                        encoded_val = self.cat_vocabs[col].get(val_str, 0)
                        for d in range(self.cat_embed_dim):
                            features_cat[t, i * self.cat_embed_dim + d] = self.cat_permutations[col][d].get(encoded_val, 0)
                
                all_features_float.append(np.concatenate([features_cont, features_disc, features_cat], axis=1))
                all_features_cat_idx.append(np.zeros((len(seq_raw), 0), dtype=np.int64))

            elif self.cat_mode == "embedded":
                # Mode embedded : x_float ne contient que cont + disc
                all_features_float.append(np.concatenate([features_cont, features_disc], axis=1))
                
                # Indices d'entiers pour nn.Embedding
                features_cat_idx = np.zeros((len(seq_raw), self.num_categorical), dtype=np.int64)
                for i, col in enumerate(self.categorical_cols):
                    for t in range(len(seq_raw)):
                        val_str = str(seq_raw[t, col])
                        features_cat_idx[t, i] = self.cat_vocabs[col].get(val_str, 0)
                all_features_cat_idx.append(features_cat_idx)
           
            # Métadonnées
            meta_processed = {}
            meta_raw = patient.get("metadata", {})
            if self.meta_config:
                for feat in self.meta_config:
                    name = feat.name
                    val = meta_raw.get(name)
                    if feat.type == "categorical":
                        val_str = str(val) if val is not None else "UNKNOWN"
                        meta_processed[name] = self.meta_vocabs[name].get(val_str, 0)
                    elif feat.type == "continuous":
                        is_missing = (val is None or val == -99.0)
                        meta_processed[name] = float(val) if not is_missing else 0.0
                        meta_processed[f"{name}_is_missing"] = 1.0 if is_missing else 0.0
            all_meta.append(meta_processed)

        # Normalisation sur l'ensemble des données flottantes
        normalized_list = self._normalize_features(all_features_float)
        self.data_cat_idx_list = all_features_cat_idx

        return normalized_list, all_meta

    # ------------------------------------------------------------------
    # Méthodes utilitaires
    # ------------------------------------------------------------------

    def _normalize_features(self, all_features: List[np.ndarray]) -> List[np.ndarray]:
        """
        Normalise les features via QuantileTransformer (normal distribution).
        Si fit_stats=True, ajuste le transformateur sur un échantillon.
        Retourne la liste des séquences normalisées.
        """
        if self.cat_mode == "duplicated":
            total_features = self.num_continuous + self.num_discrete + self.num_categorical * self.cat_embed_dim
        else:
            total_features = self.num_continuous + self.num_discrete

        if len(all_features) == 0 or total_features == 0:
            return all_features

        # Concaténer toutes les séquences pour l'analyse globale
        flat_all = np.concatenate(all_features, axis=0)

        # Imputation des NaN par la moyenne de chaque colonne
        col_means = np.nanmean(flat_all, axis=0)
        inds = np.where(np.isnan(flat_all))
        flat_all[inds] = np.take(col_means, inds[1])

        if self.fit_stats:
            if self.cat_mode == "duplicated":
                # En duplicated, les colonnes discrètes et catégorielles (soft-encoded) sont dans le flottant
                num_to_jitter = self.num_discrete + self.num_categorical * self.cat_embed_dim
            else:
                # En embedded, seules les colonnes discrètes sont dans le flottant
                num_to_jitter = self.num_discrete

            if num_to_jitter > 0:
                noise = np.random.uniform(-0.5, 0.5, size=(flat_all.shape[0], num_to_jitter))
                flat_all[:, self.num_continuous:] += noise

            # Sous-échantillonnage pour l'ajustement du QuantileTransformer
            num_samples = min(len(flat_all), 100000)
            indices = np.random.choice(len(flat_all), num_samples, replace=False)
            sample_for_fit = flat_all[indices]

            # Choix du scaler
            if self.normalization == "quantile":
                self.scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
            elif self.normalization == "standard":
                self.scaler = StandardScaler()
            elif self.normalization == "minmax":
                self.scaler = MinMaxScaler()
            elif self.normalization == "robust":
                self.scaler = RobustScaler(
                with_centering=True,
                with_scaling=True,
                quantile_range=(5.0, 95.0),
                unit_variance=False,
            )
            else:
                raise ValueError(f"Type de normalisation non supporté : {self.normalization}")
            
            self.scaler.fit(sample_for_fit)

            os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        else:
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

        # Transformer toutes les données
        flat_normalized = self.scaler.transform(flat_all)

        # Redécoupage en séquences individuelles
        normalized_list = []
        idx = 0
        for feat in all_features:
            length = len(feat)
            normalized_list.append(flat_normalized[idx:idx + length].astype(np.float32))
            idx += length

        return normalized_list

    def denormalize(self, batch: np.ndarray) -> np.ndarray:
        """
        Dénormalise un batch de données [B, L, C] ou [L, C].
        Retourne les données dans l'espace d'origine (les variables discrètes/catégorielles sont arrondies).
        """
        
        squeeze = False
        if batch.ndim == 2:
            batch = batch[np.newaxis, ...]
            squeeze = True

        if self.cat_mode == "duplicated":
            n_norm = self.num_continuous + self.num_discrete + self.num_categorical * self.cat_embed_dim
        else:
            n_norm = self.num_continuous + self.num_discrete

        if n_norm == 0:
            res = batch
        else:
            # Séparation des colonnes
            if self.cat_mode == "embedded" and batch.shape[-1] > n_norm:
                float_part = batch[..., :n_norm]
                cat_part = batch[..., n_norm:]
                # Dénormalisation de la partie flottante
                flat_float = float_part.reshape(-1, n_norm)
                restored_float = self.scaler.inverse_transform(flat_float).reshape(float_part.shape)
                # Arrondi des variables discrètes (si présentes)
                if self.num_discrete > 0:
                    restored_float[..., self.num_continuous:] = np.round(restored_float[..., self.num_continuous:])
                # Recomposition
                res = np.concatenate([restored_float, cat_part], axis=-1)
            else:
                # Cas standard (duplicated ou embedded sans catégories dans le batch)
                flat = batch.reshape(-1, n_norm)
                restored = self.scaler.inverse_transform(flat).reshape(batch.shape[0], -1, n_norm)
                # Arrondi pour les variables discrètes et catégorielles (duplicated)
                num_to_round = self.num_discrete
                if self.cat_mode == "duplicated":
                    num_to_round += self.num_categorical * self.cat_embed_dim
                if num_to_round > 0:
                    restored[..., self.num_continuous:self.num_continuous + num_to_round] = np.round(
                        restored[..., self.num_continuous:self.num_continuous + num_to_round]
                    )
                res = restored

        return res[0] if squeeze else res

    def aggregate_cat_duplicates(self, data: np.ndarray) -> np.ndarray:
        """
        Agrège les duplicats catégoriels pour retrouver la valeur la plus probable.
        data : array dénormalisé [B, L, C_total]
        Retourne : array [B, L, num_categorical] avec les indices des valeurs originales.
        """
        if self.cat_mode != "duplicated":
            raise ValueError("aggregate_cat_duplicates : Cette méthode ne s'applique qu'en mode 'duplicated'.")
                
        B, L, _ = data.shape
        out = np.zeros((B, L, self.num_categorical), dtype=np.int64)

        for b in range(B):
            for t in range(L):
                for i, col in enumerate(self.categorical_cols):
                    code_scores = {}
                    for d in range(self.cat_embed_dim):
                        idx = self.cat_map_start_idx + i * self.cat_embed_dim + d
                        val = data[b, t, idx]
                        rounded_val = int(np.round(val))
                        weight = 1.0 - 2.0 * abs(val - rounded_val)  # poids selon proximité

                        # Récupérer le code original via le mapping inverse
                        code = self.cat_inv_permutations[col][d].get(rounded_val, 0)
                        code_scores[code] = code_scores.get(code, 0.0) + weight

                    if code_scores:
                        out[b, t, i] = max(code_scores, key=code_scores.get)
        return out

    # ------------------------------------------------------------------
    # Méthodes du Dataset PyTorch
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data_float_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Retourne les segments requis selon le mode d'entraînement choisi.
        """
        float_seq = self.data_float_list[idx]
        cat_seq = self.data_cat_idx_list[idx]
        raw_window = self.windowed_data[idx]["donnees"]

        # Mode conditionné par les événements
        event_idx_col = self.event_code_index[0] if self.event_code_index is not None and len(self.event_code_index) > 0 else None
        
        if self.mode == "main":
            x_float_t, hist_float = float_seq, np.zeros((self.hist_len, float_seq.shape[1]), dtype=np.float32)
            x_cat_t, hist_cat = cat_seq, np.zeros((self.hist_len, cat_seq.shape[1]), dtype=np.int64)
            event_seq = raw_window[:, event_idx_col].astype(np.int64) if event_idx_col is not None else np.zeros(self.target_len, dtype=np.int64)
        elif self.mode == "history":
            x_float_t, hist_float = np.zeros((self.target_len, float_seq.shape[1]), dtype=np.float32), float_seq
            x_cat_t, hist_cat = np.zeros((self.target_len, cat_seq.shape[1]), dtype=np.int64), cat_seq
            event_seq = np.zeros(self.target_len, dtype=np.int64)
        else:  # "DiT"
            hist_float, x_float_t = float_seq[:self.hist_len], float_seq[self.hist_len:]
            hist_cat, x_cat_t = cat_seq[:self.hist_len], cat_seq[self.hist_len:]
            event_seq = raw_window[self.hist_len:, event_idx_col].astype(np.int64) if event_idx_col is not None else np.zeros(self.target_len, dtype=np.int64)

        # Métadonnées en tenseurs
        meta_tensors = {}
        for k, v in self.meta_list[idx].items():
            dtype = torch.float32 if isinstance(v, float) else torch.long
            meta_tensors[k] = torch.tensor(v, dtype=dtype)

        return (
            torch.tensor(x_float_t, dtype=torch.float32),
            torch.tensor(hist_float, dtype=torch.float32),
            meta_tensors,
            torch.tensor(x_cat_t, dtype=torch.long),
            torch.tensor(hist_cat, dtype=torch.long),
            torch.tensor(event_seq, dtype=torch.long)
        )


# ------------------------------------------------------------------
# Fonctions utilitaires hors classe
# ------------------------------------------------------------------

def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Regroupe un batch d'échantillons en les padant sur la longueur.
    Retourne : (x_float [B, C, L], hist_float [B, C, L_hist], meta {key: [B]})
    """
    x_floats, hist_floats, metas, x_cats, hist_cats, event_seqs = zip(*batch)

    max_len = max(x.shape[0] for x in x_floats)

    # Padding de x_float avec la dernière valeur
    padded_x = []
    for x_f in x_floats:
        curr_len = x_f.shape[0]
        if curr_len < max_len:
            pad_size = max_len - curr_len
            last_val = x_f[-1:].repeat(pad_size, 1)
            x_f_padded = torch.cat([x_f, last_val], dim=0)
        else:
            x_f_padded = x_f
        padded_x.append(x_f_padded)

    x_float_batch = torch.stack(padded_x, dim=0).transpose(1, 2).contiguous()  # [B, C, L]
    hist_f_batch = torch.stack(hist_floats, dim=0).transpose(1, 2).contiguous()  # [B, C, L_hist]

    x_cat_batch = torch.stack(x_cats, dim=0).transpose(1, 2).contiguous() # [B, Num_Cat, L]
    hist_cat_batch = torch.stack(hist_cats, dim=0).transpose(1, 2).contiguous() # [B, Num_Cat, L_hist]
    event_seq_batch = torch.stack(event_seqs, dim=0) # [B, L]

    meta_batch = {key: torch.stack([m[key] for m in metas]) for key in metas[0].keys()}

    return x_float_batch, hist_f_batch, meta_batch, x_cat_batch, hist_cat_batch, event_seq_batch


def compute_latent_scale(vae, dataloader, device, save_path, is_history=False, max_batches: Optional[int] = 50):
    """
    Calcule l'échelle du latent (1 / quantile 95% de ||mu||) pour stabiliser l'apprentissage.
    Sauvegarde la valeur dans un fichier numpy.
    """
    vae.eval()
    all_mu = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            
            x_float, hist_float, _, x_cat, hist_cat, _ = batch

            if is_history:
                h_float = hist_float.to(device)
                h_cat = hist_cat.to(device) if hist_cat is not None else None
                mu, _ = vae.encode(h_float, h_cat)
            else:
                x_f = x_float.to(device)
                x_c = x_cat.to(device) if x_cat is not None else None
                mu, _ = vae.encode(x_f, x_c)
                
            all_mu.append(mu.cpu())

    all_mu = torch.cat(all_mu, dim=0)
    scale = 1.0 / torch.quantile(all_mu.abs(), 0.95).item()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, np.array([scale]))
    return scale
```


# --- Fichier : ./diffusion_engine.py ---
```py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from transformer_block import SinusoidalTimeEmbedding, DiTBlock

class CFGWrapper(nn.Module):
    """
    Enveloppe le modèle de diffusion pour gérer le Classifier-Free Guidance.
    """
    def __init__(self, model: 'DiffusionEngine', cfg_scale: float = 2.5):
        super().__init__()
        self.model = model
        self.cfg_scale = cfg_scale

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, force_uncond: bool = False, **cond_kwargs) -> torch.Tensor:
        if force_uncond or self.cfg_scale <= 1.0:
            # On ne passe à None QUE la condition liée aux événements, on garde l'historique !
            kwargs_uncond = cond_kwargs.copy()
            kwargs_uncond['force_uncond_event'] = torch.ones(x_t.shape[0], dtype=torch.bool, device=x_t.device)
            return self.model(x_t, t, **kwargs_uncond)

        # Duplication du batch pour CFG
        x_in = torch.cat([x_t, x_t], dim=0)
        t_in = torch.cat([t, t], dim=0)
        
        # Duplication des conditionnements
        cond_combined = {}
        for k, v in cond_kwargs.items():
            if v is None:
                cond_combined[k] = None
            elif isinstance(v, dict):
                # Dupliquer les tenseurs si dictionnaire
                cond_combined[k] = {key: torch.cat([val, val], dim=0) for key, val in v.items()}
            elif isinstance(v, torch.Tensor):
                # Tenseurs classiques (z_hist, cond_idx)
                cond_combined[k] = torch.cat([v, v], dim=0)
            else:
                # Sécurité pour les booléens, ints, floats qui ne se concatènent pas
                cond_combined[k] = v
                
        # Masques de dropout pour la partie inconditionnelle
        b = x_t.shape[0]
        drop_mask = torch.cat([
            torch.zeros(b, dtype=torch.bool, device=x_t.device),
            torch.ones(b, dtype=torch.bool, device=x_t.device)
        ], dim=0)
        
        # Passe forward : On utilise preconditioned_forward pour avoir x_0
        model_to_call = self.model
        if not hasattr(model_to_call, 'preconditioned_forward') and hasattr(model_to_call, 'module'):
            model_to_call = model_to_call.module
        
        x_0_out, v_out = model_to_call.preconditioned_forward(
            x_in, t_in, 
            drop_meta=drop_mask, 
            drop_hist=drop_mask,
            force_uncond_event=drop_mask,
            **cond_combined
        )
        
        x_0_cond, x_0_uncond = torch.chunk(x_0_out, 2, dim=0)
        
        # CFG sur x_0
        x_0_pred = x_0_uncond + self.cfg_scale * (x_0_cond - x_0_uncond)
        
        # Re-calcul du vecteur vitesse final
        current_t = t.view(-1, 1, 1).clamp(min=1e-5)
        v_final = (x_t - x_0_pred) / current_t
        
        return v_final

class MetaEncoder(nn.Module):
    """
    Encodeur pour les métadonnées hétérogènes.
    """
    def __init__(self, meta_config: Optional[List[Dict]], embed_dim: int):
        super().__init__()
        self.meta_config = meta_config
        self.cat_encoders = nn.ModuleDict()
        self.cont_encoders = nn.ModuleDict()
        
        if meta_config:
            for feat in meta_config:
                name = feat.name
                if feat.type == "categorical":
                    self.cat_encoders[name] = nn.Embedding(feat.num_classes, embed_dim)
                elif feat.type == "continuous":
                    self.cont_encoders[name] = nn.Sequential(
                        nn.Linear(1, embed_dim),
                        nn.SiLU(),
                        nn.Linear(embed_dim, embed_dim)
                    )

    def forward(self, meta_dict: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.meta_config or not meta_dict:
            return None
            
        embeddings = []
        for feat in self.meta_config:
            name = feat.name
            if name in meta_dict:
                val = meta_dict[name]
                if feat.type == "categorical":
                    embeddings.append(self.cat_encoders[name](val))
                elif feat.type == "continuous":
                    if val.ndim == 1: val = val.unsqueeze(1)
                    embeddings.append(self.cont_encoders[name](val.float()))
        
        if not embeddings: return None
        return torch.stack(embeddings, dim=1).mean(dim=1)

class Transpose(nn.Module):
    def __init__(self, dim1: int, dim2: int):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim1, self.dim2)

class DiffusionEngine(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.latent_channel = config.latent_channel
        self.has_event = getattr(config, 'has_event', False)

        # Projections Entrée/Sortie
        self.input_proj = nn.Conv1d(config.latent_channel, config.embed_dim, kernel_size=1)
        self.output_proj = nn.Conv1d(config.embed_dim, config.latent_channel, kernel_size=1)
        
        # Embedding Temporel
        self.time_embed = SinusoidalTimeEmbedding(config.embed_dim)
        
        # Métadonnées
        self.meta_encoder = MetaEncoder(config.meta_config, config.embed_dim)
        self.meta_null_token = nn.Parameter(torch.randn(1, config.embed_dim))
        
        # Historique
        self.hist_proj = nn.Conv1d(config.latent_channel_hist, config.embed_dim, kernel_size=1)
        self.hist_null_token = nn.Parameter(torch.randn(1, config.embed_dim, 1))
        
        # Conditionnement par évènements
        if self.has_event:
            self.class_emb = nn.Embedding(config.num_classes, config.embed_dim)
            self.perceiver_cross_attn = nn.MultiheadAttention(
                config.embed_dim, config.num_heads, batch_first=True, dropout=config.dropout
            )
            self.latent_query = nn.Parameter(torch.randn(1, config.latent_seq_len, config.embed_dim))
            self.event_pos_enc = nn.Parameter(torch.randn(1, config.latent_seq_len, config.embed_dim))
        else:
            self.class_emb = None
            self.perceiver_cross_attn = None
            self.latent_query = None
            self.event_pos_enc = None

        # Core Transformer (DiT)
        self.blocks = nn.ModuleList([
            DiTBlock(config.embed_dim, config.num_heads, config.ff_mult, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # Positional Encoding (Séquence latente)
        self.pos_enc = nn.Parameter(torch.randn(1, config.embed_dim, config.latent_seq_len))

    def preconditioned_forward(self, x_t, t, **cond_kwargs):
        """
        Prédit x_0 pour stabiliser les extrêmes.
        """
        # Le réseau prédit la vitesse (v)
        v_pred = self.forward(x_t, t, **cond_kwargs)
        
        # Récupération de x_0
        x_0_pred = x_t - t.view(-1, 1, 1) * v_pred
        
        return x_0_pred, v_pred
        
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        cond_seq: Optional[torch.Tensor] = None, 
        meta_dict: Optional[Dict] = None, 
        z_hist: Optional[torch.Tensor] = None,
        drop_meta: Optional[torch.Tensor] = None,
        drop_hist: Optional[torch.Tensor] = None,
        force_uncond_event: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b, c, l = x.shape
        
        # 1. Input Projection
        h = self.input_proj(x) + self.pos_enc
        
        # 2. Conditionnement Global (Time + Meta)
        t_emb = self.time_embed(t)
        meta_emb = self.meta_encoder(meta_dict)
        
        if meta_emb is None:
            if self.meta_null_token is not None:
                meta_emb = self.meta_null_token.expand(b, -1)
            else:
                meta_emb = torch.zeros(b, self.embed_dim, device=x.device)
        if drop_meta is not None and self.meta_null_token is not None:
            meta_emb = torch.where(drop_meta.view(-1, 1), self.meta_null_token.expand(b, -1), meta_emb)
        
        global_cond = t_emb + meta_emb
        
        # 3. Conditionnement Séquentiel (Hist + Events)
        # Historique
        if z_hist is not None:
            hist_ctx = self.hist_proj(z_hist)
            if drop_hist is not None:
                null_h = self.hist_null_token.expand(b, -1, hist_ctx.shape[-1])
                hist_ctx = torch.where(drop_hist.view(-1, 1, 1), null_h, hist_ctx)
        else:
            hist_ctx = None
            
        # Évènements
        event_ctx = None
        if self.has_event and cond_seq is not None:
            e = self.class_emb(cond_seq) # [B, L_orig, D]
            
            # Perceiver Resampling
            query = self.latent_query.expand(b, -1, -1) + self.event_pos_enc
            event_ctx, _ = self.perceiver_cross_attn(query=query, key=e, value=e)
            event_ctx = event_ctx.transpose(1, 2).contiguous() # [B, D, L_lat]
            
            if force_uncond_event is not None:
                # Token nul d'événement
                event_ctx = torch.where(
                    force_uncond_event.view(-1, 1, 1),
                    torch.zeros_like(event_ctx),
                    event_ctx
                )
        
        # Fusion du contexte séquentiel
        if hist_ctx is not None and event_ctx is not None:
            context = torch.cat([hist_ctx, event_ctx], dim=2)
        elif hist_ctx is not None:
            context = hist_ctx
        elif event_ctx is not None:
            context = event_ctx
        else:
            context = None
            
        # Context needs to be [B, L_ctx, D] for DiTBlock
        if context is not None:
            context = context.transpose(1, 2).contiguous()
            
        # 4. DiT Blocks
        # x for DiTBlock should be [B, L, D]
        h = h.transpose(1, 2).contiguous()
        for block in self.blocks:
            h = block(h, global_cond, context)
        
        # 5. Output Projection
        h = h.transpose(1, 2).contiguous()
        return self.output_proj(h)

class FlowMatchingScheduler:
    def __init__(self, num_inference_steps: int = 15):
        self.num_inference_steps = num_inference_steps
        
    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x_0)
        x_t = (1.0 - t) * x_0 + t * noise
        v_target = noise - x_0
        return x_t, v_target

    @staticmethod
    def sample_logit_normal_t(batch_size: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(batch_size, device=device)
        t = torch.sigmoid(z).clamp(1e-4, 1.0 - 1e-4)
        return t.view(-1, 1, 1)

    @torch.no_grad()
    def sample(self, model_wrapper, shape, device,
           latent_scale=None, **cond_kwargs):
        """
        Solveur Adams-Bashforth ordre 3 non-uniforme avec amorçage Heun.

        Schedule cosinus conservé (pas non-uniformes).
        - Étapes 0-1 : Heun (ordre 2) pour constituer l'historique
        - Étapes 2+  : AB3 avec coefficients adaptés aux dt variables
        """
        x_t = torch.randn(shape, device=device)

        # t[0] ≈ 1.0 (bruit pur) → t[-1] ≈ 0.0 (signal pur)
        timesteps = torch.cos(
            torch.linspace(0, torch.pi / 2, self.num_inference_steps + 1, device=device)
        )

        # Historique circulaire : [v_{i-1}, v_{i-2}] et [dt_{i-1}, dt_{i-2}]
        v_hist  = []   # max 2 tenseurs (les deux vitesses passées)
        dt_hist = []   # max 2 scalaires float

        for i in range(self.num_inference_steps):
            t    = timesteps[i]
            dt   = (timesteps[i] - timesteps[i + 1]).item() 
            t_in = t.unsqueeze(0).expand(shape[0], 1)

            v_cur = model_wrapper(x_t, t_in, **cond_kwargs)

            if i < 2:
                # Amorçage Heun (ordre 2)
                x_pred = x_t - v_cur * dt
                t_next = timesteps[i + 1].expand(shape[0], 1)
                v_next = model_wrapper(x_pred, t_next, **cond_kwargs)
                x_t = x_t - 0.5 * (v_cur + v_next) * dt

            else:
                # AB3 non-uniforme
                h0 = dt           # pas courant        (i   → i+1)
                h1 = dt_hist[-1]  # pas précédent      (i-1 → i)
                h2 = dt_hist[-2]  # pas ante-précédent (i-2 → i-1)

                v0 = v_cur        # vitesse en t_i
                v1 = v_hist[-1]   # vitesse en t_{i-1}
                v2 = v_hist[-2]   # vitesse en t_{i-2}

                # Coefficients issus de l'intégration du polynôme de Newton-Gregory
                # Dégénèrent vers (23/12, -16/12, 5/12)·h quand h0=h1=h2=h
                c0 = (h0
                    + h0**2 / (2.0 * h1)
                    + h0**2 * (2.0*h1 + 3.0*h2) / (6.0 * h1 * (h1 + h2)))

                c1 = (-h0**2 / (2.0 * h1)
                    - h0**2 * (h0 + 2.0*h2) / (6.0 * h1 * h2))
                
                c2 = (h0**2 * (h0 + 2.0*h1) / (6.0 * h2 * (h1 + h2)))

                x_t = x_t - (c0 * v0 + c1 * v1 + c2 * v2)

            # Mise à jour de l'historique (fenêtre glissante de taille 2)
            v_hist.append(v_cur)
            dt_hist.append(dt)
            if len(v_hist)  > 2: v_hist.pop(0)
            if len(dt_hist) > 2: dt_hist.pop(0)

        if latent_scale is not None:
            x_t = x_t / latent_scale

        return x_t
```


# --- Fichier : ./loss_functions.py ---
```py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Any, Union

class SWDLoss(nn.Module):
    """
    Spectral Wasserstein Distance Loss optimisée.
    Utilise des projections pré-calculées stockées dans un buffer pour éviter les
    allocations GPU répétitives.
    Format attendu : [Batch, Channels, Length]
    """
    def __init__(self, num_channels: int, num_projections: int = 128, temporal_group: int = 8, p: int = 2):
        super().__init__()
        self.num_projections = num_projections
        self.temporal_group = temporal_group
        self.p = p
        
        # Pré-génération des projections aléatoires unitaires
        projections = torch.randn(num_channels, num_projections)
        projections = projections / torch.norm(projections, dim=0, keepdim=True)
        self.register_buffer('projections', projections)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        if C == 0:
            return torch.tensor(0.0, device=x.device)
            
        G = min(self.temporal_group, T)
        
        # Tronquage pour fenêtrage régulier
        T_trim = (T // G) * G
        if T_trim < T:
            x = x[..., :T_trim]
            y = y[..., :T_trim]
        
        # Découpage en fenêtres [Batch * Nb_Fenêtres, Channels, Taille_Fenêtre]
        x_windows = x.view(B, C, -1, G).permute(0, 2, 1, 3).reshape(-1, C, G)
        y_windows = y.view(B, C, -1, G).permute(0, 2, 1, 3).reshape(-1, C, G)
        
        # Projection : [Total_Windows, Taille_Fenêtre, Num_Projections]
        x_proj = torch.matmul(x_windows.transpose(1, 2), self.projections)
        y_proj = torch.matmul(y_windows.transpose(1, 2), self.projections)
        
        # Tri sur la dimension temporelle pour le calcul de la distance de Wasserstein 1D
        x_sorted, _ = torch.sort(x_proj, dim=1)
        y_sorted, _ = torch.sort(y_proj, dim=1)
        
        # Distance Lp moyenne
        loss = torch.pow(torch.abs(x_sorted - y_sorted), self.p).mean()
        
        return torch.pow(loss, 1.0/self.p) if self.p > 1 else loss

def evaluate_generation_swd(
    model: nn.Module, 
    vae: nn.Module, 
    vae_hist: nn.Module, 
    val_loader: torch.utils.data.DataLoader, 
    noise_scheduler: Any, 
    swd_criterion: SWDLoss,
    latent_scale: float,
    hist_latent_scale: float,
    latent_shape: Tuple[int, int],
    device: torch.device,
    dataset: Any,
    has_event: bool = True,
    max_batches: int = 1
) -> float:
    """
    Évaluation de la qualité de génération via SWD.
    """
    model.eval()
    all_swd = []
 
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i > max_batches:
                break

            if len(batch) == 6:
                real_vitals_float, history_feat_float, meta_dict, _, _, _ = batch
            else:
                real_vitals_float, history_feat_float, meta_dict = batch[:3]
                
            real_vitals_float = real_vitals_float.to(device)
            history_feat_float = history_feat_float.to(device)
            meta_dict = {k: v_meta.to(device) for k, v_meta in meta_dict.items()}
 
            b = real_vitals_float.shape[0]
            
            # Encodage de l'historique
            mu_hist, _ = vae_hist.encode(history_feat_float)
            z_hist = mu_hist * hist_latent_scale

            cond_idx = None
            if has_event and dataset is not None and dataset.num_categorical > 0:
                real_vitals_denorm = dataset.denormalize(real_vitals_float.cpu().numpy())
                cond_idx = torch.tensor(dataset.aggregate_cat_events(real_vitals_denorm)[..., 0]).to(device)

            # Génération
            x_0_gen = noise_scheduler.sample(
                model_wrapper=model,
                shape=(b, *latent_shape),
                device=device,
                latent_scale=latent_scale,
                cond_seq=cond_idx,
                z_hist=z_hist,
                meta_dict=meta_dict
            )
            
            # Décodage
            recon_out = vae.decode(x_0_gen)
            
            # Reconstitution d'un tenseur de flottants pour le SWD
            if recon_out is not None:
                swd = swd_criterion(recon_out, real_vitals_float)
                all_swd.append(swd.item())
 
    return float(np.mean(all_swd)) if all_swd else 0.0

class PIDControl:
    """
    Contrôleur PID pour stabiliser la perte KLD.
    """
    def __init__(self, target: float, kp: float = 1e-5, ki: float = 1e-6, max_weight: float = 0.0001, min_weight: float = 1e-7, start_weight: float = 1e-7):
        self.target = target
        self.kp = kp
        self.ki = ki
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.prev_error = 0.0
        self.current_weight = start_weight if start_weight is not None else min_weight

    def step(self, current_value: float) -> float:
        error = current_value - self.target
        delta_p = self.kp * (error - self.prev_error)
        delta_i = self.ki * error
        self.current_weight += (delta_p + delta_i)
        self.current_weight = max(self.min_weight, min(self.max_weight, self.current_weight))
        self.prev_error = error
        return self.current_weight
    
class HybridVAELoss(nn.Module):
    """
    Module calculant la perte hybride du VAE.
    """
    def __init__(self, spectral_weight: float = 0.1, cce_weight: float = 0.1, kld_weight: float = 0.1):
        super().__init__()
        self.spectral_weight = spectral_weight
        self.cce_weight = cce_weight
        self.kld_weight = kld_weight
        

    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        recon_float: torch.Tensor,
        recon_cat_logits: Optional[List[torch.Tensor]],
        target_float: torch.Tensor,
        target_cat: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        
        # MSE sur les variables flottantes
        MSE = F.mse_loss(recon_float, target_float, reduction='mean') if recon_float is not None else torch.tensor(0.0, device=mu.device)
        
        # Loss Catégorielle (CrossEntropy) si mode embedded
        CCE = torch.tensor(0.0, device=mu.device)
        if recon_cat_logits is not None and target_cat is not None:
            for i, logits in enumerate(recon_cat_logits):
                # logits: [B, Num_Classes, L], target: [B, L]
                CCE += F.cross_entropy(logits, target_cat[:, i, :])

        # Régularisation (KLD)
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2]))
        
        # 3. Cohérence Fréquentielle (SPEC sur tous les canaux)
        SPEC = torch.tensor(0.0, device=mu.device)
        if recon_float is not None:
            recon_fft = torch.fft.rfft(recon_float, dim=-1)
            target_fft = torch.fft.rfft(target_float, dim=-1)
            SPEC = F.l1_loss(torch.abs(recon_fft), torch.abs(target_fft), reduction='mean')
            
        # 4. Agrégation
        total_loss = (
            MSE + 
            (self.cce_weight * CCE) + 
            (self.kld_weight * KLD) + 
            (self.spectral_weight * SPEC)
        )
        
        return total_loss, MSE, CCE, KLD, SPEC

class FlowMatchingLoss(nn.Module):
    """
    Perte pour le Flow Matching avec similarité cosinus.
    """
    def __init__(self):
        super(FlowMatchingLoss, self).__init__()
        
    def forward(self, v_pred, v_target):
        # Perte sur la mse
        MSE = F.smooth_l1_loss(v_pred, v_target)
        
        # Similarité de direction
        COS = 1 - F.cosine_similarity(v_pred, v_target, dim=-1).mean()
        
        return MSE, COS

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num_losses: int =2):
        super().__init__()
        self.params = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        """
        Pondération des loss par incertitude homoscédastique
        """
        weighted_losses = []
        for i, loss in enumerate(losses):
            w = torch.exp(-self.params[i])
            weighted_losses.append(w * loss + self.params[i])
        
        return torch.sum(torch.stack(weighted_losses))


```


# --- Fichier : ./metrics.py ---
```py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, f1_score, roc_auc_score,
    classification_report, r2_score, precision_recall_curve, auc
)
from sklearn.preprocessing import label_binarize
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from imblearn.metrics import geometric_mean_score


class DatasetEvaluator:
    def __init__(self, real_data, gen_data, col_names, path_dir="checkpoints/"):
        """
        Initialise l'évaluateur.
        :param real_data: np.array contenant les données réelles (2D ou 3D).
        :param gen_data: np.array contenant les données générées (2D ou 3D).
        :param col_names: liste des noms de colonnes dans l'ordre.
        :param path_dir: dossier de sauvegarde.
        """
        self.real_data = real_data
        self.gen_data = gen_data
        self.col_names = col_names
        self.path = path_dir
        os.makedirs(path_dir, exist_ok=True)
        self.output_file = "evaluation.txt"
        self.event_idx = col_names.index('event_code')
        self.col_indices = [i for i in range(len(col_names))]
        self.continuous_indices = [i for i, col in enumerate(col_names) if col != 'event_code' and col != 'temps_sec']
        self.num_vars = len(self.continuous_indices)
        self.num_patients = self.real_data.shape[0] if self.real_data.ndim == 3 else "N/A"
        self.is_3d = (self.real_data.ndim == 3)

    def _write_and_print(self, text, file):
        """
        Écrit dans le fichier et dans la console.
        """
        print(text)
        file.write(text + "\n")

    def _get_column_vector(self, data, idx):
        """
        Extrait une variable sous forme de vecteur 1D (2D ou 3D).
        """
        if self.is_3d:
            return data[:, :, idx].flatten()
        else:
            return data[:, idx].flatten()

    def _calculate_mmd(self, x, y, gamma=1.0):
        """
        Calcule une approximation de la Maximum Mean Discrepancy (MMD).
        """
        if len(x) > 5000:
            idx = np.random.choice(len(x), 5000, replace=False)
            x, y = x[idx], y[idx]
        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        xx = np.exp(-gamma * (x - x.T)**2)
        yy = np.exp(-gamma * (y - y.T)**2)
        xy = np.exp(-gamma * (x - y.T)**2)
        return np.mean(xx) + np.mean(yy) - 2 * np.mean(xy)

    def _calculate_multiclass_auc_pr(self, y_true, y_pred_labels, classes):
        """
        Calcule l'AUC-PR macro en encodant One-vs-Rest et en ignorant les classes vides.
        """
        if len(classes) <= 1:
            return 0.0
        y_true_bin = label_binarize(y_true, classes=classes)
        y_pred_bin = label_binarize(y_pred_labels, classes=classes)
        
        if y_true_bin.shape[1] == 1:
            if np.sum(y_true_bin) == 0:
                return 0.0
            precision, recall, _ = precision_recall_curve(y_true_bin, y_pred_bin)
            return auc(recall, precision)
            
        auc_pr_list = []
        for i in range(len(classes)):
            if np.sum(y_true_bin[:, i]) == 0:
                continue
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
            auc_pr_list.append(auc(recall, precision))
            
        return np.mean(auc_pr_list) if len(auc_pr_list) > 0 else 0.0

    def _calculate_fsg(self, real_events, gen_events, classes_eval, weights=None, epsilon=1e-6):
        """
        Calcule le FSG (F-score, Weighted Geometric Mean).
        """
        if len(classes_eval) == 0:
            return 0.0

        f1_scores = f1_score(
            real_events, 
            gen_events, 
            labels=classes_eval, 
            average=None, 
            zero_division=0
        )

        if weights is None:
            w = np.ones(len(classes_eval), dtype=float)
        elif isinstance(weights, dict):
            w = np.array([weights.get(c, 1.0) for c in classes_eval], dtype=float)
        else:
            w = np.array(weights, dtype=float)

        sum_w = np.sum(w)
        if sum_w <= 0:
            return 0.0

        f1_clipped = np.clip(f1_scores, epsilon, 1.0)

        log_fsg = np.sum(w * np.log(f1_clipped)) / sum_w
        return float(np.exp(log_fsg))
    def _calculate_discrete_multiclass_auc_roc(self, real_events, gen_events, classes_eval):
            """
            Calcule l'AUC-ROC macro et pondérée One-vs-Rest sur des labels catégoriels discrets.
            """
            if len(classes_eval) <= 1:
                return 0.0, 0.0
    
            y_true_bin = label_binarize(real_events, classes=classes_eval)
            y_pred_bin = label_binarize(gen_events, classes=classes_eval)
    
            roc_scores = []
            weights = []
    
            for i, cls in enumerate(classes_eval):
                positives = np.sum(y_true_bin[:, i])
                negatives = len(y_true_bin) - positives
    
                if positives == 0 or negatives == 0:
                    continue
    
                score = roc_auc_score(y_true_bin[:, i], y_pred_bin[:, i])
                roc_scores.append(score)
                weights.append(positives)
    
            roc_scores = np.array(roc_scores)
            weights = np.array(weights)
    
            macro_auc_roc = np.mean(roc_scores) if len(roc_scores) > 0 else 0.0
            weighted_auc_roc = np.average(roc_scores, weights=weights) if len(roc_scores) > 0 else 0.0
    
            return float(macro_auc_roc), float(weighted_auc_roc)

    def evaluate_continuous_variables(self, file=None):
        """
        Calcule l'ensemble des métriques de régression et distributionnelles.
        Retourne :
          - "by_variable" : dict {nom_variable: {metrique: valeur}}
          - "records"     : list de dicts prête pour être convertie en pd.DataFrame
          - "global"      : dict des moyennes globales (avec correction de la coquille smape)
        """
        if file:
            self._write_and_print("=== ÉVALUATION DES VARIABLES CONTINUES ===\n", file)
            header = f"{'Variable':<10} | {'MSE':<9} | {'MAE':<9} | {'MAPE (%)':<9} | {'SMAPE (%)':<10} | {'R²':<7} | {'JS':<7} | {'KL':<7} | {'MMD':<7}"
            separator = "-" * len(header)
            self._write_and_print(separator, file)
            self._write_and_print(header, file)
            self._write_and_print(separator, file)

        epsilon = 1e-8
        by_variable = {}
        records = []
        
        mse_cum, mae_cum, mape_cum, smape_cum = 0.0, 0.0, 0.0, 0.0
        r2_cum, js_dist_cum, kl_div_cum, mmd_cum = 0.0, 0.0, 0.0, 0.0

        for idx in self.continuous_indices:
            col_name = self.col_names[idx]
            real_col = self._get_column_vector(self.real_data, idx)
            gen_col = self._get_column_vector(self.gen_data, idx)

            # 1. Métriques point à point
            mse = float(mean_squared_error(real_col, gen_col))
            mae = float(mean_absolute_error(real_col, gen_col))
            mape = float(np.mean(np.abs((real_col - gen_col) / (np.abs(real_col) + epsilon))) * 100)
            smape = float(np.mean(2 * np.abs(gen_col - real_col) / (np.abs(real_col) + np.abs(gen_col) + epsilon)) * 100)
            r2 = float(r2_score(real_col, gen_col))

            # 2. Histogrammes et lissage additif (pour éviter p=0 ou grid=0 menant à inf sur la KL)
            bins = np.histogram_bin_edges(np.concatenate([real_col, gen_col]), bins=50)
            p, _ = np.histogram(real_col, bins=bins, density=False)
            grid, _ = np.histogram(gen_col, bins=bins, density=False)

            p = (p + epsilon) / (np.sum(p) + epsilon * len(p))
            grid = (grid + epsilon) / (np.sum(grid) + epsilon * len(grid))

            js_dist = float(jensenshannon(p, grid))
            kl_div = float(entropy(p, grid))
            mmd = float(self._calculate_mmd(real_col, gen_col))

            # 3. Stockage des métriques de la variable
            metrics_dict = {
                "MSE": mse,
                "MAE": mae,
                "MAPE": mape,
                "SMAPE": smape,
                "R2": r2,
                "JS": js_dist,
                "KL": kl_div,
                "MMD": mmd
            }
            by_variable[col_name] = metrics_dict
            records.append({"variable": col_name, **metrics_dict})

            if file:
                row = f"{col_name:<10} | {mse:<9.4f} | {mae:<9.4f} | {mape:<9.2f} | {smape:<10.2f} | {r2:<7.3f} | {js_dist:<7.4f} | {kl_div:<7.4f} | {mmd:<7.4f}"
                self._write_and_print(row, file)

            mse_cum += mse
            mae_cum += mae
            mape_cum += mape
            smape_cum += smape
            r2_cum += r2
            js_dist_cum += js_dist
            kl_div_cum += kl_div
            mmd_cum += mmd

        if file:
            self._write_and_print(separator, file)

        n_vars = len(self.continuous_indices)
        global_summary = {
            "mse_glob": mse_cum / n_vars,
            "mae_glob": mae_cum / n_vars,
            "mape_glob": mape_cum / n_vars,
            "smape_glob": smape_cum / n_vars,  # Coquille 'smpae_glob' corrigée
            "r2_glob": r2_cum / n_vars,
            "js_glob": js_dist_cum / n_vars,
            "kl_glob": kl_div_cum / n_vars,
            "mmd_glob": mmd_cum / n_vars
        }

        return {
            "by_variable": by_variable,
            "records": records,
            "global": global_summary
        }

    def evaluate_categorical_event(self, file=None):
        """
        Calcule les métriques avancées pour la variable catégorielle (event_code)
        et retourne un dictionnaire structuré des résultats.
        """
        if file:
            self._write_and_print("\n=== ÉVALUATION DE LA VARIABLE CATÉGORIELLE (event_code) ===\n", file)
        
        real_events = self._get_column_vector(self.real_data, self.event_idx).astype(int)
        gen_events = self._get_column_vector(self.gen_data, self.event_idx).astype(int)
        classes = np.unique(np.concatenate([real_events, gen_events]))
        
        unique_elements, counts_elements = np.unique(real_events, return_counts=True)
        major_class = int(unique_elements[np.argmax(counts_elements)])
        major_count = int(np.max(counts_elements))
        classes_eval = unique_elements[counts_elements > 0]
        
        accuracy_globale = float(np.mean(real_events == gen_events))
        minority_mask = (real_events != major_class)
        
        if np.sum(minority_mask) > 0:
            accuracy_minoritaire = float(np.mean(real_events[minority_mask] == gen_events[minority_mask]))
        else:
            accuracy_minoritaire = float('nan')

        macro_f1_arith = float(f1_score(real_events, gen_events, average='macro'))
        weighted_f1 = float(f1_score(real_events, gen_events, average='weighted'))

        fsg_macro = float(self._calculate_fsg(real_events, gen_events, classes_eval, weights=None))
        support_weights = counts_elements[counts_elements > 0]
        fsg_weighted = float(self._calculate_fsg(real_events, gen_events, classes_eval, weights=support_weights))

        gmean_strict = float(geometric_mean_score(real_events, gen_events, labels=classes_eval, average='multiclass', correction=0))
        gmean_smoothed = float(geometric_mean_score(real_events, gen_events, labels=classes_eval, average='multiclass', correction=1e-3))
        macro_auc_roc, weighted_auc_roc = self._calculate_discrete_multiclass_auc_roc(real_events, gen_events, classes_eval)
        auc_pr_macro = float(self._calculate_multiclass_auc_pr(real_events, gen_events, classes))

        if file:
            self._write_and_print(f"Classe majoritaire identifiée                : {major_class} (Présente {major_count}/{len(real_events)})", file)
            self._write_and_print(f"Accuracy Globale                             : {accuracy_globale:.4f}", file)
            self._write_and_print(f"Accuracy Hors Classe Majoritaire             : {accuracy_minoritaire:.4f}", file)
            self._write_and_print(f"F1-Score Arithmétique (Macro Global)         : {macro_f1_arith:.4f}", file)
            self._write_and_print(f"F1-Score Arithmétique (Pondéré)              : {weighted_f1:.4f}", file)
            self._write_and_print(f"F1-Score géométrique (Macro Global)          : {fsg_macro:.4f}", file)
            self._write_and_print(f"F1-Score géométrique (Pondéré)               : {fsg_weighted:.4f}", file)
            self._write_and_print(f"G-Mean strict                                : {gmean_strict:.4f}", file)
            self._write_and_print(f"G-Mean smoothed (1e-3)                       : {gmean_smoothed:.4f}", file)
            self._write_and_print(f"AUC-ROC (Macro, One-vs-Rest)                 : {macro_auc_roc:.4f}", file)
            self._write_and_print(f"AUC-ROC (Pondéré)                            : {weighted_auc_roc:.4f}", file)
            self._write_and_print(f"AUC-PR  (Macro, One-vs-Rest)                 : {auc_pr_macro:.4f}", file)
            
            self._write_and_print("\nRapport détaillé par classe :", file)
            report = classification_report(real_events, gen_events, zero_division=0)
            self._write_and_print(report, file)

        # Retour structuré pour le calcul du ranking multi-configuration
        return {
            "accuracy_globale": accuracy_globale,
            "accuracy_minoritaire": accuracy_minoritaire,
            "macro_f1_arith": macro_f1_arith,
            "weighted_f1": weighted_f1,
            "fsg_macro": fsg_macro,
            "fsg_weighted": fsg_weighted,
            "gmean_strict": gmean_strict,
            "gmean_smoothed": gmean_smoothed,
            "macro_auc_roc": float(macro_auc_roc),
            "weighted_auc_roc": float(weighted_auc_roc),
            "auc_pr_macro": auc_pr_macro
        }

    def distribution_plots(self, file_img="population_distribution.png", file_txt="evaluation.txt", display_screen=True):
        """
        Génère les histogrammes de distribution pour chaque variable continue,
        calcule le pourcentage d'overlap et l'écrit dans le rapport final.
        """
        cols_grid = int(np.ceil(len(self.col_names) / 2))
        fig, axes = plt.subplots(2, cols_grid, figsize=(16, 9))
        fig.suptitle(f"Comparaison des Distributions (Population : {self.num_patients} patients)", fontsize=16)
        axes = axes.flatten()
        
        full_txt_path = f"{self.path}{file_txt}"
        
        with open(full_txt_path, "a", encoding="utf-8") as f:
            f.write("\n=== OVERLAP DES DISTRIBUTIONS (INTERSECTION DES DENSITÉS) ===\n\n")

        for plot_idx, idx in enumerate(self.col_indices):
            col_name = self.col_names[idx]
            real_flat = self._get_column_vector(self.real_data, idx)
            gen_flat = self._get_column_vector(self.gen_data, idx)
            
            min_val = min(real_flat.min(), gen_flat.min())
            max_val = max(real_flat.max(), gen_flat.max())
            bins = np.linspace(min_val, max_val, 50)
            
            hist_real, _ = np.histogram(real_flat, bins=bins)
            hist_gen, _ = np.histogram(gen_flat, bins=bins)
            
            prob_real = hist_real / len(real_flat)
            prob_gen = hist_gen / len(gen_flat)
            
            overlap = np.sum(np.minimum(prob_real, prob_gen))
            overlap_pct = overlap * 100

            weights_real = np.ones_like(real_flat) / len(real_flat)
            weights_gen = np.ones_like(gen_flat) / len(gen_flat)
            
            axes[plot_idx].hist(real_flat, bins=bins, alpha=0.5, weights=weights_real, color='blue', label='Réel')
            axes[plot_idx].hist(gen_flat, bins=bins, alpha=0.5, weights=weights_gen, color='red', label='Généré')
            
            axes[plot_idx].set_title(f"{col_name} (Overlap : {overlap_pct:.1f}%)")
            axes[plot_idx].legend()

            with open(full_txt_path, "a", encoding="utf-8") as f:
                f.write(f"   - {col_name:<10} : {overlap_pct:.1f}%\n")
        
        for i in range(len(self.col_names), len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        os.makedirs(self.path, exist_ok=True)
        plt.savefig(f"{self.path}{file_img}")
        print(f"[INFO] Graphique de distribution sauvegardé dans : {self.path}{file_img}")
        
        if display_screen and not os.environ.get("NO_PLOT"):
            plt.show()

    def run_full_analysis(self, plot=False):
        """
        Exécute l'ensemble du protocole et génère le fichier texte et l'image.
        """
        os.makedirs(self.path, exist_ok=True)
        full_txt_path = f"{self.path}{self.output_file}"
        
        with open(full_txt_path, "w", encoding="utf-8") as file:
            self._write_and_print("====================================================================================", file)
            self._write_and_print("                             RAPPORT D'ÉVALUATION                                   ", file)
            self._write_and_print("====================================================================================\n", file)
            
            cont_metrics = self.evaluate_continuous_variables(file)
            
            cat_metrics = self.evaluate_categorical_event(file)

        if plot:
            self.distribution_plots()

        print(f"[INFO] Analyse globale terminée. Résultats dans '{self.path}'.")

        return cont_metrics, cat_metrics
```


# --- Fichier : ./pipeline.py ---
```py
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
```


# --- Fichier : ./train_DiT.py ---
```py
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

    checkpoint_dir = os.path.dirname(config.autoencoder.scaler_path)
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
        "cat_mode": config.dataset.cat_mode,
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
        real_data = train_dataset.denormalize(real_np)
        gen_data  = train_dataset.denormalize(gen_np)

    fin_inf = time.time()
    logger.info("[Inférence terminée]")
    logger.info(f"Heure de fin de l'inférence : {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Clip de la SpO2
    gen_data[:,:,5] = np.clip(gen_data[:,:,5], a_min=0.0, a_max=100.0)
    # Arrondi des événements
    gen_data[:,:,7] = np.round(gen_data[:, :, 7])

    # Sauvegarde finale des fichiers NumPy
    real_path = f"{config.inference.dit_real_path}.npy"
    gen_path = f"{config.inference.dit_gen_path}.npy"

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
        path_dir= config.inference.dit_evaluator_path
    )

    evaluator.run_full_analysis()
    
if __name__ == "__main__":
    main()

```


# --- Fichier : ./train_VAE.py ---
```py
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

    if args.mode == "main":
        checkpoint_dir = os.path.dirname(config.autoencoder.scaler_path)
    else:
        checkpoint_dir = os.path.dirname(config.history_autoencoder.hist_scaler_path)
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
        spectral_weight=0,
        kld_weight=0, # 0.00025,
        cce_weight=vae_config.cce_weight,
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
            
            total_loss , mse, cce, kld, spec = loss_function(
                mu,
                logvar,
                recon_float,
                recon_cat_logits,
                inputs_float,
                target_cat=inputs_cat if config.dataset.cat_mode == "embedded" else None
            )
            
            loss = total_loss
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
                  f"| CCE: {train_metrics['cce']:.4f}/{val_metrics['cce']:.4f} "
                  f"| KLD: {train_metrics['kld']:.0f}/{val_metrics['kld']:.0f} "
                  f"| SPEC: {train_metrics['spec']:.4f}/{val_metrics['spec']:.6f} ")

        # Sauvegarde
        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
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

    real_path = f"{config.inference.vae_real_path}_{args.mode}.npy"
    gen_path = f"{config.inference.vae_gen_path}_{args.mode}.npy"
   
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
        path_dir=f"{config.inference.vae_evaluator_path}{args.mode}/"
    )

    evaluator.run_full_analysis()

if __name__ == "__main__":
    main()

```


# --- Fichier : ./transformer_block.py ---
```py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import math

class SinusoidalTimeEmbedding(nn.Module):
    """
    Embedding temporel sinusoidal pour le modèle de diffusion.
    Inclut un MLP et un scaling par 2*pi pour de meilleures performances.
    """
    def __init__(self, embed_dim: int, max_period: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B, 1]
        if t.ndim > 1:
            t = t.squeeze(-1)
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0) * (2 * math.pi)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.embed_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return self.mlp(emb)

class RoPE(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) optimisé avec cache.
    """
    def __init__(self, head_dim: int, max_len: int = 128):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim doit être pair pour RoPE"
        self.head_dim = head_dim
        
        # Précalcul des fréquences
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)
        
        # On stocke sin et cos
        self.register_buffer('cos', freqs.cos()) # [max_len, head_dim/2]
        self.register_buffer('sin', freqs.sin()) # [max_len, head_dim/2]

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, L, D]
        l = x.shape[2]
        cos = self.cos[:l]  # [L, D/2]
        sin = self.sin[:l]  # [L, D/2]
        cos = torch.cat([cos, cos], dim=-1).view(1, 1, l, -1)  # [1, 1, L, D]
        sin = torch.cat([sin, sin], dim=-1).view(1, 1, l, -1)  # [1, 1, L, D]
        return x * cos + self._rotate_half(x) * sin
    
class DiTBlock(nn.Module):
    """
    Bloc Diffusion Transformer (DiT) avec AdaLN-Zero et FlashAttention.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        
        # AdaLN-Zero (Scale, Shift, Gate)
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim, bias=True)
        )
        
        # Attention
        self.rope = RoPE(self.head_dim)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.cross_attn_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.cross_attn_kv = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.proj_self = nn.Linear(embed_dim, embed_dim)
        self.proj_cross = nn.Linear(embed_dim, embed_dim)
        
        # FeedForward
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_mult * embed_dim),
            nn.GELU(),
            nn.Linear(ff_mult * embed_dim, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. AdaLN Modulation
        mod = self.adaLN_modulation(t_emb).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod
        
        # 2. Self-Attention
        h = self.norm1(x) * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        b, l, d = h.shape
        qkv = self.qkv(h) # [B, L, 3 * D]
        q, k, v = qkv.chunk(3, dim=-1) # Sépare en 3 tenseurs [B, L, D]
        
        # Redimensionnement
        q = q.view(b, l, self.num_heads, self.head_dim).transpose(1, 2).contiguous() # [B, H, L, D]
        k = k.view(b, l, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(b, l, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        
        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)
        
        # FlashAttention (SDPA)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, l, d)
        x = x + gate_msa[:, None, :] * self.proj_self(attn_out)

        # 3. Cross-Attention
        if context is not None:
            h = self.norm2(x)
            q_cross = self.cross_attn_q(h).view(b, l, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
            
            # Même logique pour K et V du contexte
            kv_cross = self.cross_attn_kv(context) # [B, L_ctx, 2 * D]
            k_cross, v_cross = kv_cross.chunk(2, dim=-1)
            
            k_cross = k_cross.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
            v_cross = v_cross.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

            cross_out = F.scaled_dot_product_attention(q_cross, k_cross, v_cross)
            cross_out = cross_out.transpose(1, 2).contiguous().view(b, l, d)
            x = x + self.proj_cross(cross_out)            
            
        # 4. FeedForward
        h = self.norm3(x) * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        x = x + gate_mlp[:, None, :] * self.mlp(h)
        
        return x

```
