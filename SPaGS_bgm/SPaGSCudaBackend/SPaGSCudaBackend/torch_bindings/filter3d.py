import torch

from Datasets.utils import View

from SPaGSCudaBackend import _C

def update_3d_filter(
        view: View,
        positions: torch.Tensor,
        filter_3d: torch.Tensor,
        visibility_mask: torch.Tensor,
        distance2filter: float,
) -> None:
    return _C.update_3d_filter_cuda(
        positions,
        view.position.to(device=positions.device, dtype=positions.dtype),
        filter_3d,
        visibility_mask,
        view.camera.near_plane,
        distance2filter,
    )
