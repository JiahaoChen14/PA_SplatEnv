# -- coding: utf-8 --

"""HTGS/Loss.py: Loss function."""

import torch
import torchmetrics

from Framework import ConfigParameterList
from Optim.Losses.Base import BaseLoss
from Optim.Losses.DSSIM import fused_dssim
from Methods.GaussianSplatting.utils import rgb_to_sh0

def bce_loss(input: torch.Tensor, symmetrical=False) -> torch.Tensor:
    x = input.clamp(min=1e-6, max=1.0 - 1e-6)
    return -(x * torch.log(x) + (1 - x) * torch.log(1 - x)).mean() if symmetrical else (-x * torch.log(x)).mean()

def fg_in_sky_suppression(alpha: torch.Tensor,
                          mask: torch.Tensor,
                          tau: float = 1e-3) -> torch.Tensor:
    """
    Penalize foreground leakage into sky regions.
    Applies ReLU(alpha - tau) only where sky_mask == 1.
    """
    m = mask.float()
    num = torch.relu(alpha - tau) * m
    den = m.sum().clamp(min=1.0)
    return num.sum() / den


def background_in_sky_supervision(background: torch.Tensor,
                                  target: torch.Tensor,
                                  mask: torch.Tensor) -> torch.Tensor:
    """Directly supervise background RGB at pixels classified as sky."""
    m = mask.to(device=background.device, dtype=background.dtype)
    if m.ndim == 2:
        m = m.unsqueeze(0)
    target = target.to(device=background.device, dtype=background.dtype)
    channel_count = background.shape[0]
    denominator = (m.sum() * channel_count).clamp(min=1.0)
    return (torch.abs(background - target) * m).sum() / denominator


# per-image sky mode (top 5% band) → dataset-level aggregate
# Keep SH0 close to a dataset-level color target.
# The target (RGB in [0,255]) is computed per image as the mode color
# within the sky mask’s top 5% vertical band, then aggregated across images.

def background_color_regularization(bg_sh0: torch.Tensor, target):
    """L2 penalty: align SH0 with a dataset-level color target."""
    target01 = torch.as_tensor(target, dtype=bg_sh0.dtype, device=bg_sh0.device) / 255.0
    target_sh0 = rgb_to_sh0(target01)   # (3,)
    values = bg_sh0.reshape(-1, 3)      # (N, 3)
    return torch.nn.functional.mse_loss(values, target_sh0.expand_as(values))


@torch.no_grad()
def sample_sky_cone_directions(K, cone_deg, device, dtype):
    """Sample K directions within a cone_deg cap around up=(0,0,-1) in z-down coords."""
    theta = torch.rand(K, device=device) * (cone_deg * torch.pi / 180)
    phi = torch.rand(K, device=device) * (2 * torch.pi)
    dirs = torch.stack([torch.cos(phi) * torch.sin(theta),
                        torch.sin(phi) * torch.sin(theta),
                        -torch.cos(theta)], dim=-1)
    return torch.nn.functional.normalize(dirs, dim=-1)

def sky_cone_floor_loss(bgm, target, margin=0.05, cone_deg=35, K=1024):
    """
    Penalize only when rgb is darker than target by more than margin (per channel).
    penalty = ReLU((target - rgb) - margin)
    """
    parameter = next(bgm.parameters())
    dirs = sample_sky_cone_directions(K, cone_deg, parameter.device, parameter.dtype)
    # Use the model's normal evaluation path. The tiny-cuda-nn SH encoding
    # expects directions mapped from [-1, 1] to [0, 1]. Calling encoding()
    # directly here previously constrained a different region than forward().
    rgb = bgm.evaluate_directions(dirs)
    target = torch.as_tensor(target, device=rgb.device, dtype=rgb.dtype).view(1, 3)
    return torch.nn.functional.relu((target - rgb) - margin).mean()


class HTGSLoss(BaseLoss):
    def __init__(self, loss_config: ConfigParameterList) -> None:
        super().__init__()
        self.add_loss_metric('L1_Color', torch.nn.functional.l1_loss, loss_config.LAMBDA_L1)
        self.add_loss_metric('DSSIM_Color', fused_dssim, loss_config.LAMBDA_DSSIM)
        self.add_loss_metric('BCE_Alpha', bce_loss, loss_config.LAMBDA_BCE)
        self.add_loss_metric('FG_in_Sky', fg_in_sky_suppression, loss_config.LAMBDA_FG_IN_SKY)
        self.add_quality_metric('PSNR', torchmetrics.functional.image.peak_signal_noise_ratio)

    def forward(self, input: torch.Tensor, target: torch.Tensor, alpha: torch.Tensor,sky_mask: torch.Tensor | None = None) -> torch.Tensor:
        return super().forward({
            'L1_Color': {'input': input, 'target': target},
            'DSSIM_Color': {'input': input, 'target': target},
            'BCE_Alpha': {'input': alpha},
            'FG_in_Sky': {'alpha': alpha, 'mask': sky_mask},
            'PSNR': {'preds': input, 'target': target, 'data_range': 1.0}
        })
