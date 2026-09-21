import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from transformer_block import SinusoidalTimeEmbedding, DiTBlock

class CFGWrapper(nn.Module):
    """
    Wraps the diffusion model to handle Classifier-Free Guidance.
    """
    def __init__(self, model: 'DiffusionEngine', cfg_scale: float = 2.5):
        super().__init__()
        self.model = model
        self.cfg_scale = cfg_scale

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, force_uncond: bool = False, **cond_kwargs) -> torch.Tensor:
        if force_uncond or self.cfg_scale <= 1.0:
            kwargs_uncond = cond_kwargs.copy()
            kwargs_uncond['force_uncond_event'] = torch.ones(x_t.shape[0], dtype=torch.bool, device=x_t.device)
            return self.model(x_t, t, **kwargs_uncond)

        # Batch duplication for CFG
        x_in = torch.cat([x_t, x_t], dim=0)
        t_in = torch.cat([t, t], dim=0)
        
        # Duplication of conditions
        cond_combined = {}
        for k, v in cond_kwargs.items():
            if v is None:
                cond_combined[k] = None
            elif isinstance(v, dict):
                # Duplicate tensors if dictionary
                cond_combined[k] = {key: torch.cat([val, val], dim=0) for key, val in v.items()}
            elif isinstance(v, torch.Tensor):
                # Classic tensors (z_hist, cond_idx)
                cond_combined[k] = torch.cat([v, v], dim=0)
            else:
                # Safety for booleans, ints, floats that don't concatenate
                cond_combined[k] = v
                
        # Dropout masks for the unconditional part
        b = x_t.shape[0]
        drop_mask = torch.cat([
            torch.zeros(b, dtype=torch.bool, device=x_t.device),
            torch.ones(b, dtype=torch.bool, device=x_t.device)
        ], dim=0)
        
        # Forward pass: Using preconditioned_forward to get x_0
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
        
        # CFG on x_0
        x_0_pred = x_0_uncond + self.cfg_scale * (x_0_cond - x_0_uncond)
        
        # Re-calculate the final velocity vector
        current_t = t.view(-1, 1, 1).clamp(min=1e-5)
        v_final = (x_t - x_0_pred) / current_t
        
        return v_final

class MetaEncoder(nn.Module):
    """
    Encoder for heterogeneous metadata.
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
    """
    Diffusion engine for latent space generation.
    """
    def __init__(self, config: Any):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.latent_channel = config.latent_channel
        self.has_event = getattr(config, 'has_event', False)

        # Input/Output Projections
        self.input_proj = nn.Conv1d(config.latent_channel, config.embed_dim, kernel_size=1)
        self.output_proj = nn.Conv1d(config.embed_dim, config.latent_channel, kernel_size=1)
        
        # Temporal Embedding
        self.time_embed = SinusoidalTimeEmbedding(config.embed_dim)
        
        # Metadata
        self.meta_encoder = MetaEncoder(config.meta_config, config.embed_dim)
        self.meta_null_token = nn.Parameter(torch.randn(1, config.embed_dim))
        
        # History
        self.hist_proj = nn.Conv1d(config.latent_channel_hist, config.embed_dim, kernel_size=1)
        self.hist_null_token = nn.Parameter(torch.randn(1, config.embed_dim, 1))
        
        # Conditionning by events
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
        
        # Positional Encoding
        self.pos_enc = nn.Parameter(torch.randn(1, config.embed_dim, config.latent_seq_len))

    def preconditioned_forward(self, x_t, t, **cond_kwargs):
        """
        Predict x_0 to stabilize extremes.
        """
        # The network predicts the velocity (v)
        v_pred = self.forward(x_t, t, **cond_kwargs)
        
        # Retrieval of x_0
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
        
        # 2. Global conditionning (Time + Meta)
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
        
        # 3. Sequential conditionning (Hist + Events)
        # History
        if z_hist is not None:
            hist_ctx = self.hist_proj(z_hist)
            if drop_hist is not None:
                null_h = self.hist_null_token.expand(b, -1, hist_ctx.shape[-1])
                hist_ctx = torch.where(drop_hist.view(-1, 1, 1), null_h, hist_ctx)
        else:
            hist_ctx = None
            
        # Events
        event_ctx = None
        if self.has_event and cond_seq is not None:
            e = self.class_emb(cond_seq) # [B, L_orig, D]
            
            # Perceiver Resampling
            query = self.latent_query.expand(b, -1, -1) + self.event_pos_enc
            event_ctx, _ = self.perceiver_cross_attn(query=query, key=e, value=e)
            event_ctx = event_ctx.transpose(1, 2).contiguous() # [B, D, L_lat]
            
            if force_uncond_event is not None:
                # Null event token
                event_ctx = torch.where(
                    force_uncond_event.view(-1, 1, 1),
                    torch.zeros_like(event_ctx),
                    event_ctx
                )
        
        # Fusion of sequential context
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
            
        h = h.transpose(1, 2).contiguous()
        for block in self.blocks:
            h = block(h, global_cond, context)
        
        # Output Projection
        h = h.transpose(1, 2).contiguous()
        return self.output_proj(h)

class FlowMatchingScheduler:
    """
    Flow Matching Scheduler for diffusion models.
    """
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
        Solver Adams-Bashforth order 3 non-uniform with Heun initialization.

        Cosine schedule preserved (non-uniform steps).
        - Steps 0-1 : Heun (order 2) to build history
        - Steps 2+  : AB3 with coefficients adapted to variable dt
        """
        x_t = torch.randn(shape, device=device)

        # t[0] ≈ 1.0 (noise) → t[-1] ≈ 0.0 (signal)
        timesteps = torch.cos(
            torch.linspace(0, torch.pi / 2, self.num_inference_steps + 1, device=device)
        )

        # Circular history: [v_{i-1}, v_{i-2}] and [dt_{i-1}, dt_{i-2}]
        v_hist  = []   # max 2 tenseurs (les deux vitesses passées)
        dt_hist = []   # max 2 scalaires float

        for i in range(self.num_inference_steps):
            t    = timesteps[i]
            dt   = (timesteps[i] - timesteps[i + 1]).item() 
            t_in = t.unsqueeze(0).expand(shape[0], 1)

            v_cur = model_wrapper(x_t, t_in, **cond_kwargs)

            if i < 2:
                # Heun initialization (order 2)
                x_pred = x_t - v_cur * dt
                t_next = timesteps[i + 1].expand(shape[0], 1)
                v_next = model_wrapper(x_pred, t_next, **cond_kwargs)
                x_t = x_t - 0.5 * (v_cur + v_next) * dt

            else:
                # AB3 non-uniform
                h0 = dt           # current step size        (i   → i+1)
                h1 = dt_hist[-1]  # previous step size      (i-1 → i)
                h2 = dt_hist[-2]  # anteprevious step size (i-2 → i-1)

                v0 = v_cur        # velocity at t_i
                v1 = v_hist[-1]   # velocity at t_{i-1}
                v2 = v_hist[-2]   # velocity at t_{i-2}

                # Coefficients derived from the integration of the Newton-Gregory polynomial
                # They converge to (23/12, -16/12, 5/12)·h when h0=h1=h2=h
                c0 = (h0
                    + h0**2 / (2.0 * h1)
                    + h0**2 * (2.0*h1 + 3.0*h2) / (6.0 * h1 * (h1 + h2)))

                c1 = (-h0**2 / (2.0 * h1)
                    - h0**2 * (h0 + 2.0*h2) / (6.0 * h1 * h2))
                
                c2 = (h0**2 * (h0 + 2.0*h1) / (6.0 * h2 * (h1 + h2)))

                x_t = x_t - (c0 * v0 + c1 * v1 + c2 * v2)

            # Update history (sliding window of size 2)
            v_hist.append(v_cur)
            dt_hist.append(dt)
            if len(v_hist)  > 2: v_hist.pop(0)
            if len(dt_hist) > 2: dt_hist.pop(0)

        if latent_scale is not None:
            x_t = x_t / latent_scale

        return x_t