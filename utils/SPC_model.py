import numpy as np
import torch
import torch.nn as nn

from utils.hadamard import hadamard_matrix
import numpy as np


def forward_spc(x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    r"""

    Forward propagation through the Single Pixel Camera (SPC) model.

    For more information refer to: Optimized Sensing Matrix for Single Pixel Multi-Resolution Compressive Spectral Imaging 10.1109/TIP.2020.2971150

    Args:
        x (torch.Tensor): Input image tensor of size (B, L, M, N).
        H (torch.Tensor): Measurement matrix of size (S, M*N).

    Returns:
        torch.Tensor: Output measurement tensor of size (B, S, L).
    """
    B, L, M, N = x.size()
    x = x.contiguous().view(B, L, M * N)
    x = x.permute(0, 2, 1)

    # measurement
    H = H.unsqueeze(0).repeat(B, 1, 1)
    y = torch.bmm(H, x)
    return y


def backward_spc(y: torch.Tensor, H: torch.Tensor, pinv=False) -> torch.Tensor:
    r"""

    Inverse operation to reconstruct the image from measurements.

    For more information refer to: Optimized Sensing Matrix for Single Pixel Multi-Resolution Compressive Spectral Imaging  10.1109/TIP.2020.2971150

    Args:
        y (torch.Tensor): Measurement tensor of size (B, S, L).
        H (torch.Tensor): Measurement matrix of size (S, M*N).
        pinv (bool): Boolean, if True the pseudo-inverse of H is used, otherwise the transpose of H is used, defaults to False.
    Returns:
        torch.Tensor: Reconstructed image tensor of size (B, L, M, N).
    """

    Hinv = torch.pinverse(H) if pinv else torch.transpose(H, 0, 1)
    Hinv = Hinv.unsqueeze(0).repeat(y.shape[0], 1, 1)

    x = torch.bmm(Hinv, y)
    x = x.permute(0, 2, 1)
    b, c, hw = x.size()
    h = int(np.sqrt(hw))
    x = x.reshape(b, c, h, h)
    return x


class SPCModel(nn.Module):
    """SPC forward and adjoint operator using Hadamard patterns."""

    def __init__(
        self,
        im_size: int,
        compression_ratio: float,
    ) -> None:
        super().__init__()

        self.im_size = im_size
        self.n = im_size * im_size

        self.compression_ratio = compression_ratio
        self.m = int(self.n * compression_ratio)

        H = hadamard_matrix(self.n)
        H = H[: self.m, :]
        self.H = torch.tensor(H).float()

    def forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        return forward_spc(x, self.H.to(x.device))

    def transpose_pass(self, y: torch.Tensor) -> torch.Tensor:
        return backward_spc(y, self.H.to(y.device), pinv=False)

    def pseudo_inverse(self, y: torch.Tensor) -> torch.Tensor:
        return backward_spc(y, self.H.to(y.device), pinv=True)
