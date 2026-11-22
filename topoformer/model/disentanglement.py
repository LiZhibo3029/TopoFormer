import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    import timm  # type: ignore
except ImportError:
    timm = None
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class Adapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1, out_dim: int = 1, style: str = "mlp") -> None:
        super().__init__()
        self.style = style
        if self.style == "linear" or hidden_dim <= 0:
            self.linear = nn.Linear(input_dim, out_dim)
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
            self.fc1 = None  # type: ignore[assignment]
            self.fc2 = None  # type: ignore[assignment]
            self.act = None  # type: ignore[assignment]
            self.drop = None  # type: ignore[assignment]
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_dim, out_dim)
            self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.style == "linear":
            return
        assert self.fc1 is not None and self.fc2 is not None
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.style == "linear":
            assert hasattr(self, "linear") and self.linear is not None
            return self.linear(x)
        assert self.fc1 is not None and self.fc2 is not None and self.act is not None and self.drop is not None
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class OrthogonalDisentangler(nn.Module):
    def __init__(
        self,
        encoder_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        feature_dim: Optional[int] = None,
        frozen_blocks: int = 8,
        encoder_source: str = "huggingface",
        pretrained_path: Optional[str] = None,
        hf_local_files_only: bool = False,
        c_mode: str = "cls",
        c_mask_ratio: float = 0.15,
        c_attn_temp: float = 1.0,
        out_dim: int = 1,
        head_style: str = "mlp",
        debug_shapes: bool = False,
    ) -> None:
        super().__init__()
        self.encoder_source = encoder_source.lower()
        self.encoder_name = encoder_name
        self.hf_local_files_only = hf_local_files_only
        self.logger = logging.getLogger("puredino")
        self.c_mode = c_mode
        self.c_mask_ratio = c_mask_ratio
        self.c_attn_temp = c_attn_temp
        self.out_dim = max(1, int(out_dim))
        # default head style; may be overridden by args (e.g., linear_probe)
        self._head_style = head_style
        self.debug_shapes = bool(debug_shapes)
        encoder_dim: Optional[int] = None

        if self.encoder_source == "dinov3":
            try:
                from dinov3.hub import backbones  # type: ignore
            except ModuleNotFoundError:
                import importlib
                import types as _types
                import sys as _sys
                local_pkg_dir = Path(__file__).resolve().parent / "dinov3" / "dinov3"
                if not local_pkg_dir.exists():
                    raise
                # Create/override a 'dinov3' package pointing to local repo
                pkg_name = "dinov3"
                pkg = _types.ModuleType(pkg_name)
                pkg.__path__ = [str(local_pkg_dir)]  # type: ignore[attr-defined]
                _sys.modules[pkg_name] = pkg
                backbones = importlib.import_module("dinov3.hub.backbones")  # type: ignore[assignment]

            build_fn = getattr(backbones, encoder_name, None)
            if build_fn is None:
                raise ValueError(f"Unknown DINOv3 backbone: {encoder_name}")
            weights_arg = pretrained_path if pretrained_path else backbones.Weights.LVD1689M
            self.logger.info(
                f"Loading DINOv3 backbone '{encoder_name}' with weights '{weights_arg}'."
            )
            self.encoder = build_fn(pretrained=True, weights=weights_arg)
            encoder_dim = getattr(self.encoder, "num_features", None) or getattr(self.encoder, "embed_dim", None)
        elif self.encoder_source == "timm":
            if timm is None:
                raise ImportError("timm is not installed; cannot use timm encoder source.")
            self.logger.info(f"Loading timm backbone '{encoder_name}'.")
            self.encoder = timm.create_model(
                encoder_name,
                pretrained=True,
                num_classes=0,
                global_pool="avg",
            )
            encoder_dim = getattr(self.encoder, "num_features", None)
        elif self.encoder_source == "huggingface":
            try:
                from transformers import AutoModel  # type: ignore
            except ImportError as exc:
                raise ImportError("transformers must be installed for huggingface encoder source") from exc

            model_ref = pretrained_path or encoder_name
            self.logger.info(f"Loading Hugging Face backbone '{model_ref}'.")
            self.encoder = AutoModel.from_pretrained(model_ref, local_files_only=self.hf_local_files_only)
            encoder_dim = getattr(self.encoder.config, "hidden_size", None)
        else:
            raise ValueError(f"Unsupported encoder source: {encoder_source}")

        if encoder_dim is None:
            raise ValueError("Unable to determine encoder feature dimension; please specify --feature_dim explicitly.")

        if feature_dim is None or feature_dim <= 0:
            feature_dim = encoder_dim

        self.feature_dim = feature_dim
        if encoder_dim != feature_dim:
            self.logger.info(f"Projecting encoder features from {encoder_dim} to {feature_dim} dimensions.")
            self.proj = nn.Linear(encoder_dim, feature_dim)
        else:
            self.proj = nn.Identity()

        self._freeze_blocks(frozen_blocks)
        self.use_adapter_default = True
        self.adapter_disc = Adapter(feature_dim)
        self.adapter_indisc = Adapter(feature_dim)
        # Fallback simple projections if adapters are disabled at runtime
        self.proj_disc_simple = nn.Linear(feature_dim, 256)
        self.proj_indisc_simple = nn.Linear(feature_dim, 256)
        # head style may be overridden based on finetune_mode (set later via kwargs)
        head_style = self._head_style
        self.decoderA = Decoder(256, out_dim=self.out_dim, style=head_style)
        self.decoderA_fused = Decoder(512, out_dim=self.out_dim, style=head_style)
        self.decoderB = Decoder(256, out_dim=self.out_dim, style=head_style)
        # optional adaptive gating: per-sample soft routing of features between A/B
        self.use_gating: bool = False
        self.gate_layer = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 256),
        )
        # Optional feature prompt (used in prompt-tuning mode)
        self.feat_prompt = nn.Parameter(torch.zeros(feature_dim), requires_grad=False)
        self._use_feat_prompt = False
        # attention pooling for C2/C4
        self.pool_attn = nn.Linear(feature_dim, 1)

    def _freeze_blocks(self, frozen_blocks: int) -> None:
        if frozen_blocks <= 0:
            return

        blocks = None
        if self.encoder_source in {"dinov3", "timm"}:
            blocks = getattr(self.encoder, "blocks", None)
        elif self.encoder_source == "huggingface":
            candidate = None
            if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
                candidate = self.encoder.encoder.layer
            elif hasattr(self.encoder, "vit") and hasattr(self.encoder.vit, "encoder"):
                candidate = getattr(self.encoder.vit.encoder, "layer", None)
            blocks = candidate

        if blocks is None:
            self.logger.warning("Encoder does not expose transformer blocks for freezing; skipping freeze.")
            return

        blocks_iterable = list(blocks)
        frozen_blocks = min(frozen_blocks, len(blocks_iterable))
        for block in blocks_iterable[:frozen_blocks]:
            for param in block.parameters():
                param.requires_grad = False

    def _encode(self, x: torch.Tensor, topo_levels: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        if self.encoder_source == "huggingface":
            outputs = self.encoder(pixel_values=x, output_hidden_states=False)
            if self.debug_shapes and hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                try:
                    b, n, d = outputs.last_hidden_state.shape
                    self.logger.info(f"[DEBUG SHAPE] encoder=huggingface tokens BxNxD={b}x{n}x{d}")
                except Exception:
                    pass
            if getattr(outputs, "pooler_output", None) is not None:
                feats = outputs.pooler_output
            else:
                feats = outputs.last_hidden_state[:, 0]
            feats = self.proj(feats)
            if self._use_feat_prompt:
                feats = feats + self.feat_prompt
            return feats
        elif self.encoder_source == "timm":
            feats = self.encoder.forward_features(x)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            if feats.dim() == 3:
                if self.debug_shapes:
                    try:
                        b, n, d = feats.shape
                        self.logger.info(f"[DEBUG SHAPE] encoder=timm tokens BxNxD={b}x{n}x{d}")
                    except Exception:
                        pass
                cls = feats[:, 0]
                patches = feats[:, 1:]
                # default to cls for timm unless mean/attn/mask requested
                if self.c_mode == "mean" and patches is not None:
                    v = patches.mean(dim=1)
                elif self.c_mode in {"attn_pool", "attn_mask"} and patches is not None:
                    # project patches then attn pool
                    B, P, Denc = patches.shape
                    patches_proj = self.proj(patches.reshape(B * P, Denc)).reshape(B, P, -1)
                    scores = self.pool_attn(patches_proj).squeeze(-1) / max(1e-6, self.c_attn_temp)
                    weights = torch.softmax(scores, dim=1)
                    if self.c_mode == "attn_mask":
                        k = max(1, int(P * self.c_mask_ratio))
                        top_idx = torch.topk(weights, k=k, dim=1).indices
                        mask = torch.ones_like(weights)
                        mask.scatter_(1, top_idx, 0.0)
                        weights = weights * mask
                        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
                    v = (weights.unsqueeze(-1) * patches_proj).sum(dim=1)
                    feats = v
                    if self._use_feat_prompt:
                        feats = feats + self.feat_prompt
                    return feats
                else:
                    feats = cls
        else:  # dinov3 repo backbone
            if topo_levels is None:
                outputs = self.encoder.forward_features(x)
            else:
                try:
                    outputs = self.encoder.forward_features(x, topo_levels=topo_levels)
                except TypeError:
                    # Backward-compat: older backbones may not accept topo_levels
                    outputs = self.encoder.forward_features(x)
            if isinstance(outputs, dict):
                cls = outputs.get("x_norm_clstoken")
                patches = outputs.get("x_norm_patchtokens", None)
                if self.debug_shapes and (cls is not None):
                    try:
                        B = cls.shape[0]
                        if patches is not None and patches.dim() == 3:
                            P = patches.shape[1]
                            Denc = patches.shape[2]
                            self.logger.info(f"[DEBUG SHAPE] encoder=dinov3 tokens BxNxD={B}x{1+P}x{Denc} (P={P})")
                        else:
                            # Only CLS visible; Denc from CLS
                            Denc = cls.shape[-1]
                            self.logger.info(f"[DEBUG SHAPE] encoder=dinov3 tokens BxNxD={B}x1x{Denc} (no patch tokens exposed)")
                    except Exception:
                        pass
                if cls is None:
                    raise ValueError("DINOv3 backbone did not return 'x_norm_clstoken'.")
                # C-modes on dinov3
                if self.c_mode == "mean" and patches is not None:
                    B, P, Denc = patches.shape
                    patches_proj = self.proj(patches.reshape(B * P, Denc)).reshape(B, P, -1)
                    v = patches_proj.mean(dim=1)
                    feats = v
                elif self.c_mode in {"attn_pool", "attn_mask"} and patches is not None:
                    B, P, Denc = patches.shape
                    patches_proj = self.proj(patches.reshape(B * P, Denc)).reshape(B, P, -1)
                    scores = self.pool_attn(patches_proj).squeeze(-1) / max(1e-6, self.c_attn_temp)
                    weights = torch.softmax(scores, dim=1)
                    if self.c_mode == "attn_mask":
                        k = max(1, int(P * self.c_mask_ratio))
                        top_idx = torch.topk(weights, k=k, dim=1).indices
                        mask = torch.ones_like(weights)
                        mask.scatter_(1, top_idx, 0.0)
                        weights = weights * mask
                        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
                    v = (weights.unsqueeze(-1) * patches_proj).sum(dim=1)
                    feats = v
                elif self.c_mode == "token_mask" and patches is not None:
                    B, P, Denc = patches.shape
                    patches_proj = self.proj(patches.reshape(B * P, Denc)).reshape(B, P, -1)
                    k = max(1, int(P * self.c_mask_ratio))
                    # random mask per sample
                    device = patches_proj.device
                    mask = torch.ones((B, P), device=device)
                    idx = torch.rand((B, P), device=device).argsort(dim=1)[:, :k]
                    mask.scatter_(1, idx, 0.0)
                    denom = mask.sum(dim=1, keepdim=True) + 1e-6
                    v = (patches_proj * mask.unsqueeze(-1)).sum(dim=1) / denom
                    feats = v
                else:
                    feats = cls
            elif isinstance(outputs, list):
                primary = outputs[0]
                if isinstance(primary, dict):
                    cls = primary.get("x_norm_clstoken")
                    patches = primary.get("x_norm_patchtokens", None)
                    if cls is None:
                        raise ValueError("Unexpected DINOv3 backbone output structure.")
                    feats = cls
                else:
                    feats = primary
            else:
                feats = outputs
        # if feats still in encoder dim, project
        if feats.dim() == 2 and feats.size(1) != self.feature_dim:
            feats = self.proj(feats)
        if self._use_feat_prompt:
            feats = feats + self.feat_prompt
        return feats

    def forward(
        self,
        x: torch.Tensor,
        grl_lambda: float = 1.0,
        use_adapter: bool = True,
        topo_levels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        feats = self._encode(x, topo_levels=topo_levels)
        if use_adapter:
            f_disc = self.adapter_disc(feats)
            f_indisc = self.adapter_indisc(feats)
        else:
            f_disc = self.proj_disc_simple(feats)
            f_indisc = self.proj_indisc_simple(feats)
        gate = None
        if self.use_gating:
            gate = torch.sigmoid(self.gate_layer(feats))
            f_disc = f_disc * gate
            f_indisc = f_indisc * (1.0 - gate)
        logitsA = self.decoderA(f_disc)
        logitsB = self.decoderB(grad_reverse(f_indisc, grl_lambda))
        return {
            "logitsA": logitsA,
            "logitsB": logitsB,
            "f_disc": f_disc,
            "f_indisc": f_indisc,
            "feat": feats,
            "gate": gate,
        }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def covariance_loss(f_disc: torch.Tensor, f_indisc: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    b = f_disc.size(0)
    if b <= 1:
        return torch.tensor(0.0, device=f_disc.device)
    x = f_disc
    y = f_indisc
    if normalize:
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        y = torch.nn.functional.normalize(y, p=2, dim=1)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cov = (x.transpose(0, 1) @ y) / (b - 1)
    return torch.norm(cov, p="fro") ** 2


def hsic_loss(
    f_disc: torch.Tensor,
    f_indisc: torch.Tensor,
    sigma_x: Optional[float] = None,
    sigma_y: Optional[float] = None,
) -> torch.Tensor:
    # Biased HSIC with RBF kernels
    def rbf(x: torch.Tensor, sigma: Optional[float]) -> torch.Tensor:
        x = x.float()  # stabilize under AMP
        x2 = (x * x).sum(dim=1, keepdim=True)
        dist2 = x2 + x2.t() - 2.0 * (x @ x.t())
        dist2 = torch.clamp(dist2, min=0.0)
        if sigma is None:
            # median heuristic
            vals = dist2.detach().flatten()
            # exclude zeros for median; if all zero, fallback to 1.0
            nz = vals[vals > 0]
            med = torch.median(nz) if nz.numel() > 0 else torch.tensor(1.0, device=x.device)
            med = torch.clamp(med, min=1e-6)
            sigma_val = torch.sqrt(0.5 * med)
        else:
            sigma_val = torch.tensor(sigma, device=x.device).clamp(min=1e-6)
        denom = 2.0 * sigma_val**2 + 1e-8
        k = torch.exp(-dist2 / denom)
        return k

    n = f_disc.size(0)
    if n <= 1:
        return torch.tensor(0.0, device=f_disc.device)
    K = rbf(f_disc, sigma_x)
    L = rbf(f_indisc, sigma_y)
    H = torch.eye(n, device=f_disc.device) - (1.0 / n) * torch.ones((n, n), device=f_disc.device)
    KH = (K @ H)
    LH = (L @ H)
    hsic = torch.trace(KH @ LH) / ((n - 1) ** 2 + 1e-8)
    return hsic


def gram_loss(f_disc: torch.Tensor, f_indisc: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    b = f_disc.size(0)
    if b <= 1:
        return torch.tensor(0.0, device=f_disc.device)
    x = f_disc
    y = f_indisc
    if normalize:
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        y = torch.nn.functional.normalize(y, p=2, dim=1)
    gram = (x.transpose(0, 1) @ y) / b
    return torch.norm(gram, p="fro") ** 2


def _select_topk_dims_by_corr(f1: torch.Tensor, f2: torch.Tensor, topk_ratio: float) -> torch.Tensor:
    D = f1.size(1)
    k = max(1, int(D * float(topk_ratio)))
    f1c = f1 - f1.mean(dim=0, keepdim=True)
    f2c = f2 - f2.mean(dim=0, keepdim=True)
    num = (f1c * f2c).sum(dim=0)
    den = (f1c.pow(2).sum(dim=0).sqrt() * f2c.pow(2).sum(dim=0).sqrt() + 1e-6)
    corr = (num / den).abs()
    _, idx = torch.topk(corr, k=min(k, D))
    return idx


def decouple_loss_selective(
    f_disc: torch.Tensor,
    f_indisc: torch.Tensor,
    mode: str,
    normalize: bool,
    topk_ratio: float,
) -> torch.Tensor:
    if topk_ratio > 0.0 and mode in {"gram", "cov"}:
        idx = _select_topk_dims_by_corr(f_disc, f_indisc, topk_ratio)
        f_d = f_disc[:, idx]
        f_i = f_indisc[:, idx]
    else:
        f_d, f_i = f_disc, f_indisc
    if mode == "cov":
        return covariance_loss(f_d, f_i, normalize=normalize)
    if mode == "hsic":
        return hsic_loss(f_d, f_i)
    return gram_loss(f_d, f_i, normalize=normalize)


def decouple_loss_weighted(
    f_disc: torch.Tensor,
    f_indisc: torch.Tensor,
    mode: str,
    normalize: bool,
    topk_ratio: float,
    importance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Variant of decouple_loss_selective that can reweight feature dimensions
    based on a per-dim importance signal (e.g., |dL_cls/df_disc| averaged over
    the batch). Lower-importance dims receive relatively larger weights so that
    disentanglement focuses more on dimensions that are less critical for A.
    """
    if importance is not None:
        # importance expected shape [D]; emphasize lower-importance dims
        imp = importance.detach()
        if imp.dim() == 1 and imp.numel() == f_disc.size(1):
            # small trick: higher weight for smaller importance
            w = (imp.max() - imp + 1e-6)
            w = w / (w.mean() + 1e-6)
            f_disc = f_disc * w
            f_indisc = f_indisc * w
    return decouple_loss_selective(f_disc, f_indisc, mode=mode, normalize=normalize, topk_ratio=topk_ratio)


def supervised_contrastive_loss(z: torch.Tensor, y: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    z = torch.nn.functional.normalize(z, dim=1)
    sim = torch.matmul(z, z.t()) / max(temperature, 1e-6)
    B = z.size(0)
    eye = torch.eye(B, device=z.device, dtype=torch.bool)
    y = y.view(-1)
    pos_mask = (y.unsqueeze(0) == y.unsqueeze(1)) & (~eye)
    sim_exp = torch.exp(sim) * (~eye).float()
    denom = sim_exp.sum(dim=1, keepdim=True) + 1e-6
    num = (sim_exp * pos_mask.float()).sum(dim=1, keepdim=True) + 1e-12
    valid = pos_mask.any(dim=1).float()
    loss = -(valid * torch.log(num / denom)).sum() / (valid.sum() + 1e-6)
    return loss


class MINE(nn.Module):
    def __init__(self, in_dim: int = 512, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def estimate_mi(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Pos pairs
        pos = self.forward(torch.cat([x, y], dim=1))
        # Neg pairs (shuffle y)
        idx = torch.randperm(y.size(0), device=y.device)
        y_shuf = y[idx]
        neg = self.forward(torch.cat([x, y_shuf], dim=1))
        mi = pos.mean() - torch.log(torch.exp(neg).mean() + 1e-12)
        return mi


def entropy_regularizer(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    term = probs * torch.log(probs + eps) + (1.0 - probs) * torch.log(1.0 - probs + eps)
    return -term.mean()


def softmax_entropy_regularizer(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    ent = -(probs * (probs + eps).log()).sum(dim=1)
    return ent.mean()


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.collected: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert name in self.shadow
                new_avg = (1.0 - self.decay) * p.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.collected[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.collected:
                p.data.copy_(self.collected[name])
        self.collected = {}


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "max") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: Optional[float] = None
        self.num_bad = 0

    def step(self, metric: Optional[float]) -> bool:
        if metric is None or math.isnan(metric):
            # treat NaN as no improvement
            self.num_bad += 1
            return self.num_bad > self.patience
        if self.best is None:
            self.best = metric
            self.num_bad = 0
            return False
        improve = (metric - self.best) > self.min_delta if self.mode == "max" else (self.best - metric) > self.min_delta
        if improve:
            self.best = metric
            self.num_bad = 0
        else:
            self.num_bad += 1
        return self.num_bad > self.patience


@torch.no_grad()
def evaluate(
    model: OrthogonalDisentangler,
    loader: DataLoader,
    device: torch.device,
    lambda_grl: float = 0.0,
) -> Dict[str, float]:
    def _unpack_batch(batch):
        if not isinstance(batch, (tuple, list)):
            raise ValueError(f"Unexpected batch type: {type(batch)}")
        if len(batch) < 2:
            raise ValueError(f"Unexpected batch len={len(batch)}")
        images = batch[0]
        targets = batch[1]
        topo_levels = batch[2] if len(batch) >= 3 else None
        return images, targets, topo_levels

    model.eval()
    all_logits_a: List[torch.Tensor] = []
    all_logits_b: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    for batch in loader:
        images, targets, topo_levels = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        topo_dev = None
        if isinstance(topo_levels, dict):
            topo_dev = {k: v.to(device, non_blocking=True) for k, v in topo_levels.items()}
        try:
            outputs = model(images, grl_lambda=lambda_grl, topo_levels=topo_dev)
        except TypeError:
            outputs = model(images, grl_lambda=lambda_grl)
        all_logits_a.append(outputs["logitsA"].detach().cpu())
        all_logits_b.append(outputs["logitsB"].detach().cpu())
        all_targets.append(targets.detach().cpu())
    logits_a = torch.cat(all_logits_a, dim=0)
    logits_b = torch.cat(all_logits_b, dim=0)
    targets_t = torch.cat(all_targets, dim=0)
    if logits_a.dim() == 1 or logits_a.size(1) == 1:
        la = logits_a.numpy().ravel(); lb = logits_b.numpy().ravel(); ty = targets_t.numpy().ravel()
        pa = 1.0/(1.0+np.exp(-la)); pb = 1.0/(1.0+np.exp(-lb))
        pred_a = (pa>=0.5).astype(int); pred_b=(pb>=0.5).astype(int)
        bal_acc_a = balanced_accuracy_score(ty, pred_a)
        bal_acc_b = balanced_accuracy_score(ty, pred_b)
        prec_a, rec_a, f1_a, _ = precision_recall_fscore_support(ty, pred_a, average='binary', zero_division=0)
        prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(ty, pred_b, average='binary', zero_division=0)
        metrics = {
            "acc_A": float(accuracy_score(ty, pred_a)),
            "acc_B": float(accuracy_score(ty, pred_b)),
            "bal_acc_A": float(bal_acc_a),
            "bal_acc_B": float(bal_acc_b),
            "prec_A": float(prec_a),
            "rec_A": float(rec_a),
            "f1_A": float(f1_a),
            "prec_B": float(prec_b),
            "rec_B": float(rec_b),
            "f1_B": float(f1_b),
            "auc_A": 0.5,
            "auc_B": 0.5,
        }
        # compute ROC AUC when possible
        try:
            metrics["auc_A"] = float(roc_auc_score(ty, pa))
        except ValueError:
            metrics["auc_A"] = 0.5
        try:
            metrics["auc_B"] = float(roc_auc_score(ty, pb))
        except ValueError:
            metrics["auc_B"] = 0.5
        return metrics
    else:
        # multiclass: top-1 accuracy, balanced accuracy, macro OVR AUROC
        ty = targets_t.numpy()
        pred_a = logits_a.argmax(dim=1).numpy(); pred_b = logits_b.argmax(dim=1).numpy()
        bal_acc_a = balanced_accuracy_score(ty, pred_a)
        bal_acc_b = balanced_accuracy_score(ty, pred_b)
        # macro precision/recall/f1
        prec_a, rec_a, f1_a, _ = precision_recall_fscore_support(ty, pred_a, average='macro', zero_division=0)
        prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(ty, pred_b, average='macro', zero_division=0)
        # OVR macro AUROC
        pa = torch.softmax(logits_a, dim=1).numpy(); pb = torch.softmax(logits_b, dim=1).numpy()
        try:
            ovr_a = float(roc_auc_score(ty, pa, multi_class='ovr', average='macro'))
        except ValueError:
            ovr_a = 0.5
        try:
            ovr_b = float(roc_auc_score(ty, pb, multi_class='ovr', average='macro'))
        except ValueError:
            ovr_b = 0.5
        return {
            "acc_A": float(accuracy_score(ty, pred_a)),
            "acc_B": float(accuracy_score(ty, pred_b)),
            "bal_acc_A": float(bal_acc_a),
            "bal_acc_B": float(bal_acc_b),
            "prec_A": float(prec_a),
            "rec_A": float(rec_a),
            "f1_A": float(f1_a),
            "prec_B": float(prec_b),
            "rec_B": float(rec_b),
            "f1_B": float(f1_b),
            "auc_A": ovr_a,
            "auc_B": ovr_b,
        }


@torch.no_grad()
def evaluate_quick(
    model: OrthogonalDisentangler,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 512,
    lambda_grl: float = 0.0,
) -> Dict[str, float]:
    def _unpack_batch(batch):
        if not isinstance(batch, (tuple, list)):
            raise ValueError(f"Unexpected batch type: {type(batch)}")
        if len(batch) < 2:
            raise ValueError(f"Unexpected batch len={len(batch)}")
        images = batch[0]
        targets = batch[1]
        topo_levels = batch[2] if len(batch) >= 3 else None
        return images, targets, topo_levels

    model.eval()
    seen = 0
    all_logits_a: List[torch.Tensor] = []
    all_logits_b: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    for batch in loader:
        images, targets, topo_levels = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        topo_dev = None
        if isinstance(topo_levels, dict):
            topo_dev = {k: v.to(device, non_blocking=True) for k, v in topo_levels.items()}
        try:
            outputs = model(images, grl_lambda=lambda_grl, topo_levels=topo_dev)
        except TypeError:
            outputs = model(images, grl_lambda=lambda_grl)
        all_logits_a.append(outputs["logitsA"].detach().cpu())
        all_logits_b.append(outputs["logitsB"].detach().cpu())
        all_targets.append(targets.detach().cpu())
        seen += images.size(0)
        if seen >= max_samples:
            break
    if len(all_targets) == 0:
        return {"acc_A": 0.0, "auc_A": 0.5, "acc_B": 0.0, "auc_B": 0.5}
    logits_a = torch.cat(all_logits_a, dim=0)
    logits_b = torch.cat(all_logits_b, dim=0)
    targets_t = torch.cat(all_targets, dim=0)
    if logits_a.dim() == 1 or logits_a.size(1) == 1:
        la = logits_a.numpy().ravel(); lb = logits_b.numpy().ravel(); ty = targets_t.numpy().ravel()
        pa = 1.0/(1.0+np.exp(-la)); pb = 1.0/(1.0+np.exp(-lb))
        pred_a=(pa>=0.5).astype(int); pred_b=(pb>=0.5).astype(int)
        metrics = {"acc_A": float(accuracy_score(ty, pred_a)), "acc_B": float(accuracy_score(ty, pred_b)), "auc_A": 0.5, "auc_B": 0.5}
        try: metrics["auc_A"] = float(roc_auc_score(ty, pa))
        except ValueError: metrics["auc_A"] = 0.5
        try: metrics["auc_B"] = float(roc_auc_score(ty, pb))
        except ValueError: metrics["auc_B"] = 0.5
        return metrics
    else:
        pred_a = logits_a.argmax(dim=1).numpy(); pred_b = logits_b.argmax(dim=1).numpy(); ty = targets_t.numpy()
        return {"acc_A": float(accuracy_score(ty, pred_a)), "acc_B": float(accuracy_score(ty, pred_b)), "auc_A": 0.5, "auc_B": 0.5}


def train(
    model: OrthogonalDisentangler,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, float]], Optional[Dict[str, torch.Tensor]]]:
    def _unpack_batch(batch):
        if not isinstance(batch, (tuple, list)):
            raise ValueError(f"Unexpected batch type: {type(batch)}")
        if len(batch) < 2:
            raise ValueError(f"Unexpected batch len={len(batch)}")
        images = batch[0]
        targets = batch[1]
        topo_levels = batch[2] if len(batch) >= 3 else None
        return images, targets, topo_levels

    # Build optimizer params, include external adapter params if not registered under model
    base_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    # LyCORIS wrapper returns a separate network whose params may not be part of model.parameters()
    extra_params = []
    enc = getattr(model, 'encoder', None)
    if enc is not None and hasattr(enc, '_lycoris'):
        try:
            for p in enc._lycoris.parameters():  # type: ignore[attr-defined]
                if p.requires_grad and (p not in base_params):
                    extra_params.append(p)
        except Exception:
            pass
    optimizer = torch.optim.AdamW(
        [{'params': base_params}, {'params': extra_params}] if extra_params else [{'params': base_params}],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp, init_scale=16384.0, growth_interval=1000000)
    logger.info(f"CTI_AMP_POLICY enabled={args.amp} scale={scaler.get_scale():.0f} growth_interval=1000000")

    is_binary = (model.out_dim == 1)
    if is_binary:
        # BCE with optional pos_weight
        if args.use_pos_weight and hasattr(train_loader.dataset, "targets"):
            targets_np = np.array(train_loader.dataset.targets)
            if targets_np.ndim == 1 and np.unique(targets_np).size == 2:
                pos = (targets_np == 1).sum()
                neg = (targets_np == 0).sum()
                pos_weight_value = float(neg / max(pos, 1))
                bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
            else:
                bce = nn.BCEWithLogitsLoss()
        else:
            bce = nn.BCEWithLogitsLoss()
        ce = None
    else:
        ce = nn.CrossEntropyLoss(label_smoothing=max(0.0, float(args.label_smoothing)))
        bce = None

    # EMA
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None

    # MINE head and optimizer (if needed)
    mine_head: Optional[MINE] = None
    mine_opt: Optional[torch.optim.Optimizer] = None
    if args.decouple_mode == "mine":
        mine_head = MINE(in_dim=512, hidden=256).to(device)
        mine_opt = torch.optim.Adam(mine_head.parameters(), lr=args.mine_lr)

    # Scheduler: warmup + cosine
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=args.min_lr
    )

    # Monitor metric: AUC for binary, ACC for multiclass
    best_score = -math.inf
    best_state: Dict[str, torch.Tensor] = {}
    best_ema_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, float]] = []
    early = EarlyStopping(patience=args.early_stop_patience, min_delta=args.early_stop_min_delta, mode="max")

    do_val = val_loader is not None

    global_step = 0
    cti = getattr(args, "cti_audit", None)
    checkpoint_dir = getattr(args, "cti_output", None)
    if cti is not None:
        cti_probes = {f"L{i}.out": cti.stages[i].out.weight for i in cti.levels}
        cti_probes["L3.CLS"] = cti.stages[3].cls_out.weight
        cti_before = {name: param.detach().clone() for name, param in cti_probes.items()}
    # Optional KD teacher on A head
    kd_teacher: Optional[OrthogonalDisentangler] = None
    kd_weight = float(getattr(args, "kd_weight", 0.0))
    if bool(getattr(args, "kd_on_A", False)) and kd_weight > 0.0 and getattr(args, "kd_teacher_path", None):
        import copy
        teacher_path = args.kd_teacher_path
        try:
            kd_teacher = copy.deepcopy(model).to(device)
            raw = torch.load(teacher_path, map_location="cpu")
            if isinstance(raw, dict):
                if "state_dict" in raw and isinstance(raw["state_dict"], dict):
                    sd = raw["state_dict"]
                elif "model" in raw and isinstance(raw["model"], dict):
                    sd = raw["model"]
                else:
                    sd = raw
            else:
                sd = raw  # type: ignore[assignment]
            missing, unexpected = kd_teacher.load_state_dict(sd, strict=False)
            logger.info(
                f"[KD] Loaded teacher from {teacher_path} (missing={len(missing)}, unexpected={len(unexpected)})"
            )
            for p in kd_teacher.parameters():
                p.requires_grad = False
            kd_teacher.eval()
            if hasattr(kd_teacher, "use_gating"):
                kd_teacher.use_gating = False  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning(f"[KD] Failed to load teacher from {teacher_path}: {e}")
            kd_teacher = None

    # Optional SMOKE: print adapter stats before training
    if args.smoke_one_batch:
        try:
            from adapters_vit import assert_adapter_smoke
            assert_adapter_smoke(model.encoder)
        except Exception as e:
            logger.warning(f"[SMOKE] adapter stats check failed: {e}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        # Adjust learning rate for warmup (epoch-level)
        if epoch <= args.warmup_epochs:
            warmup_ratio = epoch / max(1, args.warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * warmup_ratio
        else:
            scheduler.step()

        # Adversarial warmup + optional delay + dynamic lambda
        steps_warm = max(1, int(args.adv_warmup_epochs) * max(1, len(train_loader)))
        adv_ratio = min(1.0, float(global_step) / float(steps_warm)) if args.adv_warmup_epochs > 0 else 1.0
        if epoch < int(getattr(args, "adv_delay_epochs", 0)):
            adv_ratio = 0.0
        # decay towards lambda_adv_final in the last adv_decay_epochs
        adv_decay_epochs = max(0, int(getattr(args, "adv_decay_epochs", 0)))
        adv_decay_prog = 0.0
        if adv_decay_epochs > 0 and epoch >= max(0, args.epochs - adv_decay_epochs):
            adv_decay_prog = min(1.0, (epoch - (args.epochs - adv_decay_epochs)) / float(adv_decay_epochs))
        adv_final = args.lambda_adv if args.lambda_adv_final is None else float(args.lambda_adv_final)
        adv_decay_scale = (1.0 - adv_decay_prog) + adv_decay_prog * (adv_final / max(args.lambda_adv, 1e-8))
        # dynamic scale kept in closure via nonlocal-like list
        if 'dyn_scale' not in locals():
            dyn_scale = 1.0
            dyn_bad = 0
            last_best_accA = -1.0
        curr_grl_lambda = args.grl_lambda * adv_ratio
        curr_lambda_adv = (
            args.lambda_adv
            * adv_ratio
            * adv_decay_scale
            * (dyn_scale if bool(getattr(args, "dynamic_lambda_adv", False)) else 1.0)
        )

        # Orthogonal dynamic weight (delay + decay)
        ortho_ratio = 1.0
        if epoch < int(getattr(args, "ortho_delay_epochs", 0)):
            ortho_ratio = 0.0
        ortho_decay_epochs = max(0, int(getattr(args, "ortho_decay_epochs", 0)))
        ortho_decay_prog = 0.0
        if ortho_decay_epochs > 0 and epoch >= max(0, args.epochs - ortho_decay_epochs):
            ortho_decay_prog = min(1.0, (epoch - (args.epochs - ortho_decay_epochs)) / float(ortho_decay_epochs))
        ortho_final = args.lambda_ortho if args.lambda_ortho_final is None else float(args.lambda_ortho_final)
        ortho_decay_scale = (1.0 - ortho_decay_prog) + ortho_decay_prog * (ortho_final / max(args.lambda_ortho, 1e-8))
        curr_lambda_ortho = args.lambda_ortho * ortho_ratio * ortho_decay_scale

        optimizer.zero_grad(set_to_none=True)
        for batch in iterator:
            images, targets, topo_levels = _unpack_batch(batch)
            images = images.to(device, non_blocking=True)
            if is_binary:
                targets_f = targets.float().unsqueeze(1).to(device, non_blocking=True)
            else:
                targets_l = targets.long().to(device, non_blocking=True)

            topo_dev = None
            if isinstance(topo_levels, dict):
                topo_dev = {k: v.to(device, non_blocking=True) for k, v in topo_levels.items()}

            with torch.cuda.amp.autocast(enabled=args.amp):
                try:
                    outputs = model(
                        images,
                        grl_lambda=curr_grl_lambda,
                        use_adapter=(not args.no_adapter),
                        topo_levels=topo_dev,
                    )
                except TypeError:
                    outputs = model(images, grl_lambda=curr_grl_lambda, use_adapter=(not args.no_adapter))
                logits_a = outputs["logitsA"]
                logits_b = outputs["logitsB"]
                f_disc = outputs["f_disc"]
                f_indisc = outputs["f_indisc"]
                gate = outputs.get("gate", None)

                # Optionally use fusion head for A
                if args.fusion_concat:
                    logits_a = model.decoderA_fused(torch.cat([f_disc, f_indisc], dim=1))

                if is_binary:
                    loss_cls = bce(logits_a, targets_f)  # type: ignore[arg-type]
                else:
                    loss_cls = ce(logits_a, targets_l)  # type: ignore[arg-type]

                # Disentanglement regularizer
                if args.decouple_mode == "mine":
                    assert mine_head is not None
                    # Step T (MINE) to maximize MI using detached features
                    with torch.no_grad():
                        fd_det = f_disc.detach()
                        fi_det = f_indisc.detach()
                    mine_opt.zero_grad()  # type: ignore[union-attr]
                    mi_pos = mine_head.estimate_mi(fd_det, fi_det)  # type: ignore[union-attr]
                    loss_mine_T = -mi_pos
                    loss_mine_T.backward()
                    mine_opt.step()  # type: ignore[union-attr]
                    # Feature step: minimize MI with current MINE
                    mi_est = mine_head.estimate_mi(f_disc, f_indisc)  # type: ignore[union-attr]
                    loss_decouple = mi_est
                elif args.decouple_mode == "none":
                    loss_decouple = loss_cls.new_zeros(())
                else:
                    # cov/hsic/gram via selective wrapper, optionally reweighted by per-dim importance
                    imp_vec: Optional[torch.Tensor] = None
                    if bool(getattr(args, "kd_on_A", False)):
                        try:
                            grad_fd = torch.autograd.grad(
                                outputs=loss_cls,
                                inputs=f_disc,
                                retain_graph=True,
                                create_graph=False,
                                allow_unused=True,
                            )[0]
                            if grad_fd is not None:
                                imp_vec = grad_fd.abs().mean(dim=0)
                        except Exception:
                            imp_vec = None
                    loss_decouple = decouple_loss_weighted(
                        f_disc,
                        f_indisc,
                        mode=args.decouple_mode,
                        normalize=args.normalize_ortho,
                        topk_ratio=float(getattr(args, "ortho_topk_ratio", 0.0)),
                        importance=imp_vec,
                    )
                # decouple loss defined above; no-op branch removed (decouple_mode always one of cov/hsic/mine/gram)

                # Adversarial branch selection
                if args.adv_mode == "grl":
                    logits_b_eff = logits_b
                elif args.adv_mode in {"cam", "cam_grl"}:
                    # compute gradient wrt f_indisc to build a mask; if A head does not depend on f_indisc, fallback to zeros
                    grad = torch.autograd.grad(
                        outputs=logits_a.sum(), inputs=f_indisc, retain_graph=True, create_graph=True, allow_unused=True
                    )[0]
                    if grad is None:
                        grad = torch.zeros_like(f_indisc)
                    sal = grad.abs().detach()
                    k = max(1, int(sal.size(1) * float(args.cam_topk)))
                    topk_vals, topk_idx = torch.topk(sal, k=k, dim=1)
                    mask = torch.zeros_like(sal)
                    mask.scatter_(1, topk_idx, 1.0)
                    masked = f_indisc * (1.0 - mask)
                    if args.adv_mode == "cam_grl":
                        masked = grad_reverse(masked, curr_grl_lambda)
                    logits_b_eff = model.decoderB(masked)
                elif args.adv_mode == "kl_uniform":
                    # Encourage B head to be uniform (task-insensitive): minimize KL(p||uniform)
                    logits_b_eff = model.decoderB(f_indisc)
                else:  # none
                    logits_b_eff = None

                if logits_b_eff is not None:
                    if args.adv_mode == "kl_uniform":
                        # softmax over classes (for binary: construct 2-class prob)
                        if logits_b_eff.dim() == 2 and logits_b_eff.size(1) > 1:
                            probs = torch.softmax(logits_b_eff, dim=1)
                            C = float(logits_b_eff.size(1))
                        else:
                            s = torch.sigmoid(logits_b_eff)
                            probs = torch.cat([s, 1.0 - s], dim=1)
                            C = 2.0
                        ent = -(probs * (probs + 1e-6).log()).sum(dim=1).mean()
                        loss_adv = -ent + math.log(C)  # KL(p||U) = -H(p) + log C
                    else:
                        if is_binary:
                            if args.adv_loss_type == "focal":
                                # binary focal with logits
                                prob = torch.sigmoid(logits_b_eff)
                                pt = prob * targets_f + (1 - prob) * (1 - targets_f)
                                alpha = float(getattr(args, "adv_focal_alpha", 0.25))
                                gamma = float(getattr(args, "adv_focal_gamma", 2.0))
                                loss_adv_term = (
                                    -alpha * (1 - pt).pow(gamma) * targets_f * torch.log(prob + 1e-8)
                                    - (1 - alpha) * (pt).pow(gamma) * (1 - targets_f) * torch.log(1 - prob + 1e-8)
                                ).mean()
                            else:
                                loss_adv_term = bce(logits_b_eff, targets_f)  # type: ignore[arg-type]
                            loss_entropy = entropy_regularizer(logits_b_eff)
                        else:
                            loss_adv_term = ce(logits_b_eff, targets_l)  # type: ignore[arg-type]
                            loss_entropy = softmax_entropy_regularizer(logits_b_eff)
                        loss_adv = loss_adv_term - args.alpha_entropy * loss_entropy
                else:
                    loss_adv = torch.tensor(0.0, device=device)

                # Optional SupCon on A branch
                loss_supcon = torch.tensor(0.0, device=device)
                if bool(getattr(args, "supcon_on_A", False)):
                    # Build labels for SupCon: binary vs multiclass
                    sup_y = targets_l if not is_binary else targets_f.view(-1).long()
                    loss_supcon = supervised_contrastive_loss(
                        f_disc, sup_y, temperature=float(getattr(args, "supcon_temp", 0.07))
                    )
                loss_gate = torch.tensor(0.0, device=device)
                gate_l1_w = float(getattr(args, "gate_l1", 0.0))
                if bool(getattr(args, "use_gate", False)) and gate_l1_w > 0.0 and gate is not None:
                    loss_gate = gate.abs().mean()
                # Optional KD on A head
                loss_kd = torch.tensor(0.0, device=device)
                if kd_teacher is not None and kd_weight > 0.0:
                    with torch.no_grad():
                        t_out = kd_teacher(images, grl_lambda=0.0, use_adapter=(not args.no_adapter))
                        t_logits = t_out["logitsA"]
                    T = float(getattr(args, "kd_temperature", 1.0))
                    eps = 1e-6
                    if is_binary:
                        s_logit = logits_a / max(T, 1e-6)
                        t_logit = t_logits / max(T, 1e-6)
                        s_prob = torch.sigmoid(s_logit)
                        t_prob = torch.sigmoid(t_logit)
                        s_log = torch.stack(
                            [torch.log(1.0 - s_prob + eps), torch.log(s_prob + eps)],
                            dim=1,
                        )
                        t_p = torch.stack([1.0 - t_prob, t_prob], dim=1)
                        loss_kd = torch.nn.functional.kl_div(s_log, t_p, reduction="batchmean")
                    else:
                        s_log = torch.log_softmax(logits_a / max(T, 1e-6), dim=1)
                        t_p = torch.softmax(t_logits / max(T, 1e-6), dim=1)
                        loss_kd = torch.nn.functional.kl_div(s_log, t_p, reduction="batchmean") * (T * T)

                loss_total = (
                    loss_cls
                    + curr_lambda_ortho * loss_decouple
                    + curr_lambda_adv * loss_adv
                    + float(getattr(args, "lambda_supcon", 0.0)) * loss_supcon
                    + gate_l1_w * loss_gate
                    + kd_weight * loss_kd
                )

                # gradient accumulation
                loss_total = loss_total / max(1, args.accum_steps)

            # Track one trainable param before step (for SMOKE)
            tracked_name, tracked_before = None, None
            if args.smoke_one_batch:
                for n, p in model.named_parameters():
                    if p.requires_grad and p.data.numel() > 0:
                        tracked_name, tracked_before = n, p.data.detach().clone()
                        break

            if cti is not None and not math.isfinite(loss_total.item()):
                bad = [f"L{i}" for i, stage in enumerate(cti.stages)
                       if not torch.isfinite(stage.last_amax)]
                raise FloatingPointError(
                    f"CTI_NONFINITE epoch={epoch} step={global_step + 1} first_cti={next(iter(bad), None)} "
                    f"loss={loss_total.item()} grad_norm=unavailable_before_backward"
                )
            scaler.scale(loss_total).backward()

            if (global_step + 1) % args.accum_steps == 0:
                # optional gradient clipping (needs unscale before clipping)
                if args.clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    try:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            max_norm=args.clip_grad_norm, error_if_nonfinite=cti is not None,
                        )
                    except RuntimeError as error:
                        if cti is None or "non-finite" not in str(error):
                            raise
                        bad = [f"L{i}" for i, stage in enumerate(cti.stages)
                               if any(p.grad is not None and not torch.isfinite(p.grad).all()
                                      for p in stage.parameters())]
                        raise FloatingPointError(
                            f"CTI_NONFINITE epoch={epoch} step={global_step + 1} first_cti={next(iter(bad), None)} "
                            f"loss={loss_total.item():.9g} grad_norm=nonfinite_pre_clip scale={scaler.get_scale()}"
                        ) from error
                if cti is not None and global_step == 0:
                    gradients = {name: param.grad.detach().abs().amax().item() if param.grad is not None else 0
                                 for name, param in cti_probes.items()}
                    assert all(math.isfinite(value) and value > 0 for value in gradients.values()), gradients
                scaler.step(optimizer)
                scaler.update()
                if cti is not None and global_step == 0:
                    deltas = {name: (param.detach() - cti_before[name]).abs().amax().item()
                              for name, param in cti_probes.items()}
                    assert all(math.isfinite(value) and value > 0 for value in deltas.values()), deltas
                    logger.info(f"CTI_OPTIMIZER_PASS gradients={gradients} changes={deltas}")
                    del cti_before
                if cti is not None and (global_step + 1) % 100 == 0:
                    logger.info(f"CTI_STEP epoch={epoch} step={global_step + 1} loss={loss_total.item():.7g} grad_norm={grad_norm.item():.7g}")
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                if args.smoke_one_batch:
                    # Verify an update happened
                    if tracked_name is not None:
                        cur = dict(model.named_parameters())[tracked_name].data
                        try:
                            updated = not torch.allclose(tracked_before, cur)
                        except Exception:
                            updated = True
                        logger.info(f"[SMOKE] one-step done. updated={updated} on {tracked_name}")
                    # Exit after first optimizer step
                    return {"smoke": 1.0}

            epoch_loss += loss_total.item() * images.size(0)
            iterator.set_postfix({
                "loss": float(loss_total.item()),
                "cls": float(loss_cls.item()),
                "dec": float(loss_decouple.item() if torch.is_tensor(loss_decouple) else float(loss_decouple)),
                "adv": float(loss_adv.item()),
            })
            global_step += 1

        # Flush remaining grads if accumulation steps not aligned
        if (global_step % max(1, args.accum_steps)) != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        epoch_loss /= len(train_loader.dataset)

        if do_val:
            # Quick validation on subset for fast feedback
            quick_metrics = evaluate_quick(model, val_loader, device, max_samples=args.val_quick_samples, lambda_grl=0.0)  # type: ignore[arg-type]

            # Full validation possibly at interval; use EMA weights if enabled
            run_full = (epoch % args.val_full_interval == 0)
            if ema is not None and run_full:
                ema.apply_to(model)
            val_metrics = evaluate(model, val_loader, device, lambda_grl=0.0) if run_full else quick_metrics  # type: ignore[arg-type]
            if ema is not None and run_full:
                ema.restore(model)
        else:
            # Refit / training-only mode: no validation set.
            quick_metrics = {"acc_A": float("nan"), "auc_A": float("nan"), "acc_B": float("nan"), "auc_B": float("nan")}
            run_full = False
            val_metrics = {}

        record = {
            "epoch": epoch,
            "loss": epoch_loss,
            "loss_cls": float(loss_cls.detach().item()) if 'loss_cls' in locals() else None,
            "loss_decouple": float(loss_decouple.detach().item()) if 'loss_decouple' in locals() and torch.is_tensor(loss_decouple) else None,
            "loss_adv": float(loss_adv.detach().item()) if 'loss_adv' in locals() else None,
            "q_acc_A": quick_metrics["acc_A"],
            "q_auc_A": quick_metrics["auc_A"],
            "q_acc_B": quick_metrics["acc_B"],
            "q_auc_B": quick_metrics["auc_B"],
            "acc_A": val_metrics.get("acc_A", float("nan")),
            "auc_A": val_metrics.get("auc_A", float("nan")),
            "acc_B": val_metrics.get("acc_B", float("nan")),
            "auc_B": val_metrics.get("auc_B", float("nan")),
            "bal_acc_A": val_metrics.get("bal_acc_A", float("nan")),
            "bal_acc_B": val_metrics.get("bal_acc_B", float("nan")),
            "prec_A": val_metrics.get("prec_A", float("nan")),
            "rec_A": val_metrics.get("rec_A", float("nan")),
            "f1_A": val_metrics.get("f1_A", float("nan")),
            "prec_B": val_metrics.get("prec_B", float("nan")),
            "rec_B": val_metrics.get("rec_B", float("nan")),
            "f1_B": val_metrics.get("f1_B", float("nan")),
        }
        history.append(record)
        logger.info(json.dumps(record))

        # Update best on full validation only.
        # Prefer Balanced Accuracy over plain accuracy when available
        # so that epoch selection respects class imbalance.
        if do_val and run_full:
            if "bal_acc_A" in val_metrics and isinstance(val_metrics["bal_acc_A"], float) and not math.isnan(val_metrics["bal_acc_A"]):
                current_score = val_metrics["bal_acc_A"]
            elif is_binary:
                current_score = val_metrics["auc_A"]
            else:
                current_score = val_metrics["acc_A"]
            if isinstance(current_score, float) and (not math.isnan(current_score)):
                improved = current_score > best_score
            else:
                improved = False
            if improved:
                best_score = current_score
                fingerprint = getattr(args, "cti_dino_sha256", None)
                if fingerprint:
                    from .checkpoints import compact_state

                    best_payload = compact_state(model, fingerprint)
                    best_state = best_payload["weights"]
                else:
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    best_payload = best_state
                if checkpoint_dir is not None:
                    temporary = checkpoint_dir / "best_val.tmp"
                    torch.save(best_payload, temporary)
                    temporary.replace(checkpoint_dir / "best_val.pt")
                    label = "CTI_CHECKPOINT" if cti is not None else "NO_CTI_CHECKPOINT"
                    logger.info(f"{label} epoch={epoch} val_BAcc_EMA={best_score:.9f}")
                if ema is not None and checkpoint_dir is None and cti is None:
                    # store EMA weights too
                    ema.apply_to(model)
                    best_ema_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    ema.restore(model)
            # dynamic lambda_adv scheduling based on acc_A progression
            if bool(getattr(args, 'dynamic_lambda_adv', False)):
                accA_cur = val_metrics.get('acc_A', None)
                if isinstance(accA_cur, float):
                    if accA_cur <= (last_best_accA + float(getattr(args,'dyn_delta',5e-4))):
                        dyn_bad += 1
                    else:
                        last_best_accA = accA_cur
                        dyn_bad = 0
                    if dyn_bad >= int(getattr(args,'dyn_patience',2)):
                        dyn_bad = 0
                        dyn_scale = max(float(getattr(args,'dyn_lambda_min',0.01)), (dyn_scale * 0.7))
                        logger.info(f"[DYNAMIC] Reducing lambda_adv scale to {dyn_scale:.4f}")

            # Early stopping check
            if early.step(current_score):
                logger.info("Early stopping triggered")
                break

    # Load best weights (allow non-strict for adapter/quant buffers)
    # Default behavior keeps backward compatibility: load best@val at end.
    # Caller can disable this (e.g. to keep last-epoch weights) via args.load_best_at_end=False.
    if bool(getattr(args, "load_best_at_end", True)) and best_state:
        model.load_state_dict(best_state, strict=False)
    # Return also best_ema in case caller wants to save it
    if args.ema and best_ema_state is not None:
        # Attach for saving downstream
        model._best_ema_state = best_ema_state  # type: ignore[attr-defined]

    return history, best_state if best_state else None
