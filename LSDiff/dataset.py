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
    Optimized dataset for time series of LSDiff.
    Manages continuous, discrete and categorical variables.
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
        # Basic parameters
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

        # Index of start of categorical variables in the complete tensor
        self.cat_map_start_idx = self.num_continuous + self.num_discrete

        # Save paths
        self.scaler_path = scaler_path
        self.vocab_path = scaler_path.replace('_scaler.pkl', '_vocab.json')
        self.cat_vocab_path = scaler_path.replace('_scaler.pkl', '_cat_vocab.json')
        self.cat_permutation_path = scaler_path.replace('_scaler.pkl', '_cat_permutations.json')

        self.meta_config = meta_config
        self.fit_stats = fit_stats

        # Load raw data
        self.raw_data = raw_data

        # Vocabularies and permutation mappings
        self.meta_vocabs = self._prepare_meta_vocabs()
        self.cat_vocabs = self._prepare_cat_vocabs()

        # Generate windows
        self.windowed_data = self._generate_windows()

        # Prepare data (extraction, normalization)
        self.data_float_list, self.meta_list = self._prepare_data()

    # ------------------------------------------------------------------
    # Methods for preparing vocabularies
    # ------------------------------------------------------------------

    def _prepare_meta_vocabs(self) -> Dict[str, Dict[str, int]]:
        """Build or load the vocabulary of categorical metadata."""
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
        Build or load :
        - the vocabulary of categorical values (by column)
        - the permutation mappings for duplicates
        Returns only the vocabulary.
        """
        # Initial vocabulary with token PAD_UNKNOWN (index 99)
        vocabs = {col: {"PAD_UNKNOWN": 99} for col in self.categorical_cols}

        if self.fit_stats:
            # Build vocabulary from data
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
                # Build permutation mappings for soft-encoding
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
            # Load from existing files
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
    # Window splitting
    # ------------------------------------------------------------------

    def _generate_windows(self) -> List[Dict]:
        """
        Generate sliding windows adapted to the chosen training mode.
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
    # Data extraction and normalization
    # ------------------------------------------------------------------

    def _prepare_data(self) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Extract features (continuous, discrete, categorical), apply normalization,
        and prepare metadata. Returns the list of normalized sequences and the list of metadata.
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
    # Utility methods
    # ------------------------------------------------------------------

    def _normalize_features(self, all_features: List[np.ndarray]) -> List[np.ndarray]:
        """
        Normalize features via QuantileTransformer (normal distribution).
        If fit_stats=True, fit the transformer on a sample.
        Returns the list of normalized sequences.
        """
        if self.cat_mode == "duplicated":
            total_features = self.num_continuous + self.num_discrete + self.num_categorical * self.cat_embed_dim
        else:
            total_features = self.num_continuous + self.num_discrete

        if len(all_features) == 0 or total_features == 0:
            return all_features

        # Concatenate all sequences for global analysis
        flat_all = np.concatenate(all_features, axis=0)

        # Imputation of NaN by the mean of each column
        col_means = np.nanmean(flat_all, axis=0)
        inds = np.where(np.isnan(flat_all))
        flat_all[inds] = np.take(col_means, inds[1])

        if self.fit_stats:
            if self.cat_mode == "duplicated":
                # In the duplicated mode, the discrete and categorical (soft-encoded) columns are in the float part.
                num_to_jitter = self.num_discrete + self.num_categorical * self.cat_embed_dim
            else:
                # In embedded mode, only the discrete columns are in the float part
                num_to_jitter = self.num_discrete

            if num_to_jitter > 0:
                noise = np.random.uniform(-0.5, 0.5, size=(flat_all.shape[0], num_to_jitter))
                flat_all[:, self.num_continuous:] += noise

            # Subsampling for QuantileTransformer fitting
            num_samples = min(len(flat_all), 100000)
            indices = np.random.choice(len(flat_all), num_samples, replace=False)
            sample_for_fit = flat_all[indices]

            # Choice of scaler
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
                raise ValueError(f"Normalisation type not supported: {self.normalization}")
            
            self.scaler.fit(sample_for_fit)

            os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        else:
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

        # Transform all data
        flat_normalized = self.scaler.transform(flat_all)

        # Resplit into individual sequences
        normalized_list = []
        idx = 0
        for feat in all_features:
            length = len(feat)
            normalized_list.append(flat_normalized[idx:idx + length].astype(np.float32))
            idx += length

        return normalized_list

    def denormalize(self, batch: np.ndarray) -> np.ndarray:
        """
        Denormalize a batch of data [B, L, C] or [L, C].
        Returns the data in the original space (discrete/categorical variables are rounded).
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
            # Separation of columns
            if self.cat_mode == "embedded" and batch.shape[-1] > n_norm:
                float_part = batch[..., :n_norm]
                cat_part = batch[..., n_norm:]
                # Denormalization of the float part
                flat_float = float_part.reshape(-1, n_norm)
                restored_float = self.scaler.inverse_transform(flat_float).reshape(float_part.shape)
                # Rounding of discrete variables (if present)
                if self.num_discrete > 0:
                    restored_float[..., self.num_continuous:] = np.round(restored_float[..., self.num_continuous:])
                # Reconstruction
                res = np.concatenate([restored_float, cat_part], axis=-1)
            else:
                # Standard case (duplicated or embedded without categories in the batch)
                flat = batch.reshape(-1, n_norm)
                restored = self.scaler.inverse_transform(flat).reshape(batch.shape[0], -1, n_norm)
                # Rounding for discrete and categorical variables (duplicated)
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
        Aggregates categorical duplicates to find the most probable value.
        data : denormalized array [B, L, C_total]
        Returns : array [B, L, num_categorical] with indices of original values.
        """
        if self.cat_mode != "duplicated":
            raise ValueError("aggregate_cat_duplicates : This method only applies in 'duplicated' mode.")
                
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
                        weight = 1.0 - 2.0 * abs(val - rounded_val)  # weight according to proximity

                        # Retrieve the original code via the inverse mapping
                        code = self.cat_inv_permutations[col][d].get(rounded_val, 0)
                        code_scores[code] = code_scores.get(code, 0.0) + weight

                    if code_scores:
                        out[b, t, i] = max(code_scores, key=code_scores.get)
        return out

    # ------------------------------------------------------------------
    # PyTorch Dataset methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data_float_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Returns the required segments according to the chosen training mode.
        """
        float_seq = self.data_float_list[idx]
        cat_seq = self.data_cat_idx_list[idx]
        raw_window = self.windowed_data[idx]["donnees"]

        # Conditioned by events
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

        # Metadata as tensors
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
# Utility functions outside class
# ------------------------------------------------------------------

def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Groups a batch of samples by padding them on length.
    Returns : (x_float [B, C, L], hist_float [B, C, L_hist], meta {key: [B]})
    """
    x_floats, hist_floats, metas, x_cats, hist_cats, event_seqs = zip(*batch)

    max_len = max(x.shape[0] for x in x_floats)

    # Padding of x_float with the last value
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
    Calculates the latent scale (1 / 95th percentile of ||mu||) to stabilize learning.
    Saves the value in a numpy file.
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