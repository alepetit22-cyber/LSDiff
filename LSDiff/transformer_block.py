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
