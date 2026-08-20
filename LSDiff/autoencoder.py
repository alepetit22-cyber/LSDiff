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
    def __init__(self, cat_mode: str = "duplicated", cat_vocab_sizes: Optional[int] = None, cat_embed_dim: int = 6):
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