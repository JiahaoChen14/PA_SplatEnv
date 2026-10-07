# -- coding: utf-8 --

"""SPaGS/Loss.py: Loss function."""

import torch
import torchmetrics

from Framework import ConfigParameterList
from Optim.Losses.Base import BaseLoss
from Optim.Losses.DSSIM import fused_dssim

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
    denominator = (m.sum() * background.shape[0]).clamp(min=1.0)
    return (torch.abs(background - target) * m).sum() / denominator


class SPaGSLoss(BaseLoss):
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
