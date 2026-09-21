import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Any, Union

class SWDLoss(nn.Module):
    """
    Spectral Wasserstein Distance Loss.
    """
    def __init__(self, num_channels: int, num_projections: int = 128, temporal_group: int = 8, p: int = 2):
        super().__init__()
        self.num_projections = num_projections
        self.temporal_group = temporal_group
        self.p = p
        
        # Pre-generation of randomly unitary projections
        projections = torch.randn(num_channels, num_projections)
        projections = projections / torch.norm(projections, dim=0, keepdim=True)
        self.register_buffer('projections', projections)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        if C == 0:
            return torch.tensor(0.0, device=x.device)
            
        G = min(self.temporal_group, T)
        
        # Truncate for regular windowing
        T_trim = (T // G) * G
        if T_trim < T:
            x = x[..., :T_trim]
            y = y[..., :T_trim]
        
        # Split into windows
        x_windows = x.view(B, C, -1, G).permute(0, 2, 1, 3).reshape(-1, C, G)
        y_windows = y.view(B, C, -1, G).permute(0, 2, 1, 3).reshape(-1, C, G)
        
        # Projection
        x_proj = torch.matmul(x_windows.transpose(1, 2), self.projections)
        y_proj = torch.matmul(y_windows.transpose(1, 2), self.projections)
        
        # Sort along temporal dimension for 1D Wasserstein distance calculation
        x_sorted, _ = torch.sort(x_proj, dim=1)
        y_sorted, _ = torch.sort(y_proj, dim=1)
        
        # Average distance
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
    Evaluation of generation quality via SWD.
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
            
            # Encode history
            mu_hist, _ = vae_hist.encode(history_feat_float)
            z_hist = mu_hist * hist_latent_scale

            cond_idx = None
            if has_event and dataset is not None and dataset.num_categorical > 0:
                real_vitals_denorm = dataset.denormalize(real_vitals_float.cpu().numpy())
                cond_idx = torch.tensor(dataset.aggregate_cat_events(real_vitals_denorm)[..., 0]).to(device)

            # Generation
            x_0_gen = noise_scheduler.sample(
                model_wrapper=model,
                shape=(b, *latent_shape),
                device=device,
                latent_scale=latent_scale,
                cond_seq=cond_idx,
                z_hist=z_hist,
                meta_dict=meta_dict
            )
            
            # Decode
            recon_out = vae.decode(x_0_gen)
            
            # Reconstitution of a float tensor for SWD
            if recon_out is not None:
                swd = swd_criterion(recon_out, real_vitals_float)
                all_swd.append(swd.item())
 
    return float(np.mean(all_swd)) if all_swd else 0.0

class PIDControl:
    """
    PID controller to stabilize the KLD loss.
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
    Calculate the loss of the VAE.
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

        
        # MSE for floats variables
        MSE = F.mse_loss(recon_float, target_float, reduction='mean') if recon_float is not None else torch.tensor(0.0, device=mu.device)
        
        # CCE (CrossEntropy for categorical variables)
        CCE = torch.tensor(0.0, device=mu.device)
        if recon_cat_logits is not None and target_cat is not None:
            for i, logits in enumerate(recon_cat_logits):
                # logits: [B, Num_Classes, L], target: [B, L]
                CCE += F.cross_entropy(logits, target_cat[:, i, :])

        # Regularization (KLD)
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2]))
        
        # Frequency Coherence
        SPEC = torch.tensor(0.0, device=mu.device)
        if recon_float is not None:
            recon_fft = torch.fft.rfft(recon_float, dim=-1)
            target_fft = torch.fft.rfft(target_float, dim=-1)
            SPEC = F.l1_loss(torch.abs(recon_fft), torch.abs(target_fft), reduction='mean')
            
        # Aggregation
        total_loss = (
            MSE + 
            (self.cce_weight * CCE) + 
            (self.kld_weight * KLD) + 
            (self.spectral_weight * SPEC)
        )
        
        return total_loss, MSE, CCE, KLD, SPEC

class FlowMatchingLoss(nn.Module):
    """
    Loss for Flow Matching with cosine similarity.
    """
    def __init__(self):
        super(FlowMatchingLoss, self).__init__()
        
    def forward(self, v_pred, v_target):
        # Loss on mse
        MSE = F.smooth_l1_loss(v_pred, v_target)
        
        # Direction similarity
        COS = 1 - F.cosine_similarity(v_pred, v_target, dim=-1).mean()
        
        return MSE, COS

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num_losses: int =2):
        super().__init__()
        self.params = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        """
        Weighting of losses by homoscedastic uncertainty
        """
        weighted_losses = []
        for i, loss in enumerate(losses):
            w = torch.exp(-self.params[i])
            weighted_losses.append(w * loss + self.params[i])
        
        return torch.sum(torch.stack(weighted_losses))

