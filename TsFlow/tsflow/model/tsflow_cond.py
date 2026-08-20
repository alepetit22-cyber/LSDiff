from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ema_pytorch import EMA
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.util import lagged_sequence_values
from gluonts.transform.split import InstanceSplitter
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from tsflow.arch import BackboneModel
# from tsflow.arch.backbones import BackboneModelMultivariate
from tsflow.model._base import PREDICTION_INPUT_NAMES, TSFlowBase
from tsflow.utils.gaussian_process import Q0Dist
from tsflow.utils.util import LongScaler
from tsflow.utils.variables import Prior, Setting

patch_typeguard()


class TSFlowCond(TSFlowBase):
    def __init__(
        self,
        setting: str,
        target_dim: int,
        event_dim: int,
        cce_weight: float,
        context_length: int,
        prediction_length: int,
        backbone_params: dict,
        prior_params: dict,
        optimizer_params: dict,
        ema_params: dict,
        frequency: str,
        normalization: str | None = None,
        use_lags: bool = True,
        use_ema: bool = False,
        num_steps: int = 16,
        solver: str = "euler",
        matching: str = "random",
    ):
        super().__init__(
            context_length=context_length,
            prediction_length=prediction_length,
            prior_params=prior_params,
            optimizer_params=optimizer_params,
            frequency=frequency,
            normalization=normalization,
            use_lags=use_lags,
            use_ema=use_ema,
            num_steps=num_steps,
            solver=solver,
            matching=matching,
        )
        num_features = 2 + (len(self.lags_seq) if use_lags else 0)

        # target_dim = target_dim if setting == Setting.MULTIVARIATE else 1

        # if setting == Setting.UNIVARIATE:
        #     self.backbone = BackboneModel(
        #         **backbone_params,
        #         num_features=num_features,
        #         target_dim=target_dim,
        #     )
        # else:
        #     self.backbone = BackboneModelMultivariate(
        #         **backbone_params,
        #         num_features=num_features,
        #         target_dim=target_dim,
        #     )
        
        ### Modification ###
        # Implémentation de la gestion de la variable catégorielle
        self.num_event_classes = 99             # Nombre max de codes d'événements possibles
        self.event_embedding_dim = event_dim    # Taille du vecteur d'embedding choisi
        self.target_dim_raw = target_dim        # Nombre de canaux bruts

        self.event_embedding = nn.Embedding(
            num_embeddings=self.num_event_classes, 
            embedding_dim=self.event_embedding_dim
        )

        # Correction de la gestion des données multivariées
        effective_target_dim = (target_dim - 1) + self.event_embedding_dim
        
        # Gestion de la CCE
        self.event_classifier = nn.Linear(self.event_embedding_dim, self.num_event_classes)
        self.cce_weight = cce_weight            # Pondération de la CCE

        backbone_params = backbone_params.copy()
        backbone_params["output_dim"] = effective_target_dim

        self.backbone = BackboneModel(
            **backbone_params,
            num_features=num_features,
            target_dim=effective_target_dim,
        )
        ###

        self.ema_backbone = EMA(self.backbone, **ema_params)
        self.setting = setting
        self.guidance_scale = 0
        self.sigmax = self.sigmin
        self.q0 = Q0Dist(
            **prior_params,
            prediction_length=prediction_length,
            freq=self.freq,
            iso=1e-1 if self.prior != Prior.ISO else 0,
        )


    def _embed_categorical(
        self, x_tensor: torch.Tensor, event_codes_raw: torch.Tensor
    ) -> torch.Tensor:
        """
        Transforme un tenseur [B, L, target_dim] en [B, L, K_ext]
        en remplaçant le canal d'index 7 (event_code) par son embedding.
        """
        x_cont = x_tensor[..., :7]
        x_evt = torch.clamp(event_codes_raw.long(), min=0, max=self.num_event_classes - 1)
        x_emb = self.event_embedding(x_evt)
        x_meta = x_tensor[..., 8:]
        return torch.cat([x_cont, x_emb, x_meta], dim=-1)             

        
    # @typechecked
    def _extract_features(
        self, data: dict
    ) -> Tuple[
        TensorType[float, "batch", "length", "num_series"],
        TensorType[float, "batch", "length", "num_series"],
        TensorType[float, "batch", "length", "num_series"],
        TensorType[float, "batch", 1, "num_series"],
        TensorType[float, "batch", 1, "num_series"],
        TensorType[float, "batch", "length", "num_series", "num_features"],
    ]:
        past             = data["past_target"]
        future           = data["future_target"]
        context_observed = data["past_observed_values"]
        mean             = data["mean"]

        context       = past[:, -self.context_length :]
        long_context  = past[:, : -self.context_length]
        prior_context = past[:, -self.prior_context_length :]

        # Définir le nombre de canaux continus (0 à 6)
        cont_dim = 7

        # --- Séparation des canaux continus et catégoriels ---
        context_cont = context[..., :cont_dim]
        context_cat  = context[..., cont_dim:]

        long_context_cont = long_context[..., :cont_dim]
        long_context_cat  = long_context[..., cont_dim:]

        prior_context_cont = prior_context[..., :cont_dim]
        prior_context_cat  = prior_context[..., cont_dim:]

        future_cont = future[..., :cont_dim]
        future_cat  = future[..., cont_dim:]

        # Pour le scaler, on utilise les parties continues du passé complet
        past_cont       = past[..., :cont_dim]
        observed_cont   = context_observed[..., :cont_dim]   # masque d'observation pour les continus

        # Appliquer le scaler sur les canaux continus seulement
        if isinstance(self.scaler, LongScaler):
            # Cas particulier LongScaler (si vous l'utilisez)
            scaled_context_cont, loc_cont, scale_cont = self.scaler(context_cont, scale=mean[..., :cont_dim] if mean is not None else None)
            # Pour les autres segments, on utilisera loc_cont et scale_cont
        else:
            # Scalers standards (MeanScaler, StdScaler, etc.)
            _, loc_cont, scale_cont = self.scaler(past_cont, observed_cont)
            # Normalisation : on suit la même logique que le code original
            scaled_context_cont = context_cont / scale_cont

        # Normaliser les autres segments selon la même logique
        scaled_long_context_cont  = (long_context_cont - loc_cont) / scale_cont
        scaled_prior_context_cont = (prior_context_cont - loc_cont) / scale_cont
        scaled_future_cont        = (future_cont - loc_cont) / scale_cont

        # --- Reconstruction des tenseurs complets avec les catégories non normalisées ---
        scaled_context = torch.cat([scaled_context_cont, context_cat], dim=-1)
        scaled_long_context = torch.cat([scaled_long_context_cont, long_context_cat], dim=-1)
        scaled_prior_context = torch.cat([scaled_prior_context_cont, prior_context_cat], dim=-1)
        scaled_future = torch.cat([scaled_future_cont, future_cat], dim=-1)

        # --- Construction de loc et scale complets (pour la dénormalisation ultérieure) ---
        # On suppose que loc_cont et scale_cont ont la forme (batch, cont_dim) ou (batch, 1, cont_dim)
        # On les concatène avec des zéros et des uns pour les canaux catégoriels.
        # Il faut adapter la dimension selon la forme exacte de loc_cont.
        if loc_cont.dim() == 2:  # (batch, cont_dim)
            loc_cat = torch.zeros(loc_cont.shape[0], context_cat.shape[-1], device=loc_cont.device)
            scale_cat = torch.ones_like(loc_cat)
            loc_full = torch.cat([loc_cont, loc_cat], dim=-1)
            scale_full = torch.cat([scale_cont, scale_cat], dim=-1)
        else:  # cas où loc_cont a une dimension temporelle? Normalement non, mais on vérifie
            # Si loc_cont a la forme (batch, 1, cont_dim) on fait de même
            loc_cat = torch.zeros(loc_cont.shape[0], 1, context_cat.shape[-1], device=loc_cont.device)
            scale_cat = torch.ones_like(loc_cat)
            loc_full = torch.cat([loc_cont, loc_cat], dim=-1)
            scale_full = torch.cat([scale_cont, scale_cat], dim=-1)

        # --- Suite du code original : calcul de x1, x0, etc. ---
        x1_raw = torch.cat([scaled_context, scaled_future], dim=-2)
        batch_size, length, c = x1_raw.shape

        # Régression GP sur le prior (inchangé)
        dist = self.q0.gp_regression(
            rearrange(scaled_prior_context, "b l c -> (b c) l"),
            self.prediction_length,
        )
        fut = rearrange(dist.sample(), "(b c) l -> b l c", c=c)
        fut_mean = rearrange(dist.mean, "(b c) l -> b l c", c=c)

        x0_raw = torch.cat([scaled_context, fut], dim=-2)

        # Récupération des event_codes bruts (non normalisés) – ils se trouvent dans les parties catégorielles
        # Attention, les event_codes sont maintenant dans context_cat et future_cat, mais ils étaient normalisés dans le code original.
        # Ici, comme on ne les a pas normalisés, ils sont bruts. On les récupère à partir des tenseurs catégoriels d'origine.
        # Pour cela, on doit conserver les event_codes bruts avant de les normaliser (on ne les normalise pas).
        # On peut les récupérer à partir de past et future sur le canal 7.
        evt_ctx_raw = past[:, -self.context_length:, 7]   # canal 7 brut (non normalisé)
        evt_fut_raw = future[:, :, 7]                     # canal 7 brut

        # Pour l'embedding, on utilise ces codes bruts (entiers)
        evt_full_raw_x1 = torch.cat([evt_ctx_raw, evt_fut_raw], dim=1)
        # Pour x0, on remplace future par du bruit aléatoire (comme avant)
        evt_fut_noise = torch.randint(0, self.num_event_classes, evt_fut_raw.shape, device=evt_fut_raw.device)
        evt_full_raw_x0 = torch.cat([evt_ctx_raw, evt_fut_noise], dim=1)

        # Application de l'embedding sur x1_raw et x0_raw
        x1 = self._embed_categorical(x1_raw, evt_full_raw_x1)
        x0 = self._embed_categorical(x0_raw, evt_full_raw_x0)

        # Masque d'observation : on le construit de la même manière mais en tenant compte des dimensions
        K_ext = x1.shape[-1]
        observation_mask = torch.zeros(batch_size, length, K_ext, device=x1.device)
        # On utilise context_observed pour les continus (les 7 premiers) et on étend pour les embeddings
        ctx_obs_cont = context_observed[:, -self.context_length:, :cont_dim]  # (B, ctx_len, cont_dim)
        # Pour l'event_code, on utilise le même masque mais étendu sur la dimension d'embedding
        ctx_obs_evt = context_observed[:, -self.context_length:, 7:8].expand(-1, -1, self.event_embedding_dim)
        # Métadonnées (canaux 8 et +) : on les considère comme toujours observées (ou selon leur masque)
        ctx_obs_meta = context_observed[:, -self.context_length:, cont_dim+1:]  # ou on pourrait tout mettre à 1
        # On concatène pour former le masque complet
        ctx_obs_ext = torch.cat([ctx_obs_cont, ctx_obs_evt, ctx_obs_meta], dim=-1)
        observation_mask[:, :-self.prediction_length] = ctx_obs_ext

        # Construction des features (lags, fut_mean, mask)
        features = []
        if self.use_lags:
            lags = lagged_sequence_values(
                self.lags_seq,
                scaled_long_context,
                x1_raw,
                dim=1,
            )
            features.append(lags)

        fut_mean_full = torch.cat([scaled_context, fut_mean], dim=-2)
        # Pour fut_mean, on utilise un event_code neutre (0) pour l'embedding
        evt_fut_neutral = torch.zeros_like(evt_fut_raw)
        evt_full_raw_meanfeat = torch.cat([evt_ctx_raw, evt_fut_neutral], dim=1)
        fut_mean_emb = self._embed_categorical(fut_mean_full, evt_full_raw_meanfeat)
        features.append(fut_mean_emb.unsqueeze(-1))
        features.append(observation_mask.unsqueeze(-1))

        features = torch.cat(features, dim=-1)

        # Récupération des event_codes bruts pour la CCE
        event_codes_ctx = past[:, -self.context_length:, 7].long()
        event_codes_fut = future[:, :, 7].long()
        event_codes = torch.cat([event_codes_ctx, event_codes_fut], dim=1)
        event_codes = torch.clamp(event_codes, 0, self.num_event_classes - 1)

        # Retourner x1, x0, observation_mask, loc_full, scale_full, features, event_codes
        return x1, x0, observation_mask, loc_full, scale_full, features, event_codes
    
    # Surcharge de la fonction de perte
    def p_losses(
        self,
        x1: TensorType[float, "batch", "length", "num_series"],
        x0: TensorType[float, "batch", "length", "num_series"],
        t: TensorType[float, "batch", 1],
        features: TensorType[float, "batch", "length", "num_series", "num_features"] | None = None,
        event_codes: TensorType[int, "batch", "length"] | None = None,
        ) -> TensorType[float]:

        # Broadcast de t
        t_3d = t.unsqueeze(-1)                  # [B, 1, 1] → broadcaste sur [B, L, K]
        t_broadcast = t_3d.expand_as(x1)       # [B, L, K] → pour x1_hat

        # forward_path reçoit t broadcastable, pas besoin de le modifier
        psi, dpsi = self.forward_path(x1, x0, t_3d)

        # Le backbone reçoit t en [B, 1] comme dans le code original
        predicted_flow = self.backbone(t, psi, features)   # ← t original [B, 1]

        # MSE sur canaux continus
        E = self.event_embedding_dim
        idx_cont = list(range(7))
        loss_mse = F.mse_loss(predicted_flow[..., idx_cont], dpsi[..., idx_cont])

        # CCE sur x1 estimé
        loss_cce = torch.tensor(0.0, device=self.device)
        if event_codes is not None:
            x1_hat   = psi + (1.0 - t_broadcast) * predicted_flow    # [B, L, K]
            pred_emb = x1_hat[..., 7:7 + E]           # [B, L, E]
            logits   = self.event_classifier(pred_emb)               # [B, L, num_classes]
            B, L, C  = logits.shape
            loss_cce = F.cross_entropy(
                logits.reshape(B * L, C),
                event_codes.reshape(B * L).long(),
            )

        loss = loss_mse + self.cce_weight * loss_cce
        return loss, loss_mse, loss_cce
    
    @typechecked
    def training_step(
        self, 
        data: dict, 
        idx: int
        ) -> dict:
        assert self.training is True
        x1, x0, _, _, _, features, event_codes = self._extract_features(data)
        t = torch.rand((x1.shape[0], 1), device=self.device)
        
        loss, loss_mse, loss_ce = self.p_losses(x1, x0, t, features, event_codes)
        
        self.log("train_loss",     loss,     on_step=False, batch_size=x1.shape[0], on_epoch=True)
        self.log("train_mse", loss_mse, on_step=False, batch_size=x1.shape[0], on_epoch=True)
        self.log("train_cce",  loss_ce,  on_step=False, batch_size=x1.shape[0], on_epoch=True)
        
        return {"loss": loss}
    
    # @typechecked
    def forward(
        self,
        past_target: TensorType[float, "batch", "length"] | TensorType[float, "batch", "length", "num_series"],
        past_observed_values: TensorType[float, "batch", "length"] | TensorType[float, "batch", "length", "num_series"],
        mean: TensorType[float, "batch", 1] | TensorType[float, "batch", 1, "num_series"] = None,
    ) -> (
        TensorType[float, "batch", "num_samples", "prediction_length"]
        | TensorType[float, "batch", "num_samples", "prediction_length", "num_series"]
    ):
        # This is only used during prediction
        past_target          = past_target.to(self.device).repeat_interleave(self.num_samples, dim=0)
        past_observed_values = past_observed_values.to(self.device).repeat_interleave(self.num_samples, dim=0)
        mean                 = mean.to(self.device).repeat_interleave(self.num_samples, dim=0)
        future_target        = torch.zeros_like(past_target[:, -self.prediction_length :])
        
        data = dict(
            past_target=past_target,
            past_observed_values=past_observed_values,
            mean=mean,
            future_target=future_target,
        )

        observation, x0, observation_mask, loc, scale, features, _ = self._extract_features(data)
        x0 = x0 + self.sigmax * torch.randn_like(x0)
        pred = self.sample(
            x0.to(self.device),
            features=features,
            observation=observation,
            observation_mask=observation_mask,
            guidance_scale=self.guidance_scale,
        )

        pred_continuous = pred[..., :7]                               # Les 7 signes vitaux débruités
        pred_embedding = pred[..., 7:7+self.event_embedding_dim]      # Les dimensions de l'embedding débruitées
        pred_meta = pred[..., 7+self.event_embedding_dim:]            # Les 2 métadonnées
        
        # Projection des dimensions du tenseur
        logits = self.event_classifier(pred_embedding)                # [B, L, num_classes]
        pred_event_class = torch.argmax(logits, dim=-1).float().unsqueeze(-1)  # [B, L, 1]
        
        # Concaténation du tenseur
        pred = torch.cat([pred_continuous, pred_event_class, pred_meta], dim=-1)

        # Neutralisation du scaler pour la variable catégorielle
        scale_eff = scale.clone()
        loc_eff   = loc.clone()
        scale_eff[..., 7] = 1.0
        loc_eff[..., 7]   = 0.0

        if self.setting == Setting.UNIVARIATE:
            pred = rearrange(pred * scale_eff + loc_eff, "(b n) l 1 -> b n l", n=self.num_samples)
        else:
            pred = rearrange(pred * scale_eff + loc_eff, "(b n) l k -> b n l k", n=self.num_samples)
            
        return pred[:, :, observation.shape[1] - self.prediction_length :]

    @typechecked
    def get_predictor(
        self, 
        input_transform: InstanceSplitter, 
        batch_size: int = 40, 
        device: str | torch.device = None
    ):
        return PyTorchPredictor(
            prediction_length=self.prediction_length,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=self,
            batch_size=batch_size,
            input_transform=input_transform,
            device=device,
        )
