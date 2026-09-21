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
        Calcul of the total number of channels after transformation
        [continus + discrets + (categoriels * duplication)]
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
        assert v in (2, 4, 8), f"stride={v} invalid"
        return v

    @model_validator(mode='after')
    def seq_len_divisible_by_stride(self):
        assert self.seq_len % (self.stride ** 2) == 0, \
            f"seq_len={self.seq_len} should be divisible by stride²={self.stride**2}"
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
        assert v in (2, 4, 8), f"stride={v} invalid"
        return v

    @model_validator(mode='after')
    def seq_len_divisible_by_stride(self):
        assert self.seq_len % (self.stride ** 2) == 0, \
            f"seq_len={self.seq_len} should be divisible by stride²={self.stride**2}"
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

try:
    config = load_config()
except Exception as e:
    print(f"Error loading configuration: {e}")
    config = None