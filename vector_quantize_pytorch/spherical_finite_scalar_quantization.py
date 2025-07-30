"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""

from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor, int32
from torch.cuda.amp import autocast
import math
from .finite_scalar_quantization import FSQ

from einops import rearrange, pack, unpack

# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# tensor helpers

def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

def round_ste_dynamic(z: Tensor, dynamic_levels: list) -> Tensor:
    """
    Round with straight through gradients.
    
    dynamic_levels = [
        {# channel 0
            'depth': 2
            'levels': torch.tensor([-4, -3, -2.5, -2.25, -2, -1, 0, 1])
        },
        {
            ...
        }
    ]
    """
    # quantization
    assert len(dynamic_levels) == z.shape[-1]
    with torch.no_grad():
        zhat = torch.zeros_like(z)
        for i, levels_dict in enumerate(dynamic_levels): 
            depth = levels_dict['depth']
            levels = levels_dict['levels']
            for d in range(depth):
                coef = 2 ** d
                z_i_d = z[:, i] * coef
                zhat_i_d = z_i_d.round() / coef
                valid = (zhat_i_d[:, None] - levels[None] == 0).sum(dim=-1).float()
                zhat[:, i] = valid * zhat_i_d + (1 - valid) * zhat[:, i]
    return z + (zhat - z).detach()

# main class
class SphericalFSQ(FSQ):
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None
    ):
        super().__init__(levels, dim, num_codebooks, keep_num_codebooks_dim, scale)

    def quantize(self, z: Tensor) -> Tensor:
        z_fsq = self.quantize_fsq(z)
        indices = self.codes_to_indices_fsq(z_fsq)
        codes = self.quantize_parametric(z_fsq)
        return codes, indices

    def quantize_parametric(self, quantized):
        # theta_0, ..., theta_{n-2} to [0, pi]
        # theta_{n-1} to [-pi, pi]
        quantized = torch.cat([(math.pi / 2) * quantized[..., :-1] + math.pi / 2, math.pi * quantized[..., -1:] + math.pi], dim=-1)
        sines = torch.sin(quantized) # (..., D)
        cum_sines = torch.cumprod(sines, dim=-1) # (..., D)
        cosines = torch.cos(quantized) # (..., D)
        # expand
        cum_sines_expand = torch.cat([torch.ones_like(sines[..., :1]), cum_sines], dim=-1)
        cosines_expand = torch.cat([cosines, torch.ones_like(sines[..., :1])], dim=-1)
        parametric = cum_sines_expand * cosines_expand
        return parametric
        
    def quantize_fsq(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2 # Renormalize to [-1, 1].
        return quantized / half_width
    
    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat: Tensor, eps: float = 1e-3) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / (half_width + eps)
    
    def codes_to_indices_fsq(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim # added - parametric
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    @torch.no_grad()
    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        dist = (zhat[..., None, :] - self.implicit_codebook[None, None, None]).abs().mean(dim=-1)
        return dist.min(dim=-1).indices

    def indices_to_codes(
        self,
        indices: Tensor,
        project_out = True
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        # added - parametric
        codes = self.quantize_parametric(codes)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        if project_out:
            codes = self.project_out(codes)

        if is_img_or_video:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    @autocast(enabled = False)
    def forward(self, z: Tensor) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        if is_img_or_video:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)

        codes, indices = self.quantize(z)

        # test
        # indices_x = self.codes_to_indices(codes)

        codes = rearrange(codes, 'b n c d -> b n (c d)')

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if is_img_or_video:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        return out, indices

class SphericalFSNQ(SphericalFSQ):
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None
    ):
        super().__init__(levels, dim, num_codebooks, keep_num_codebooks_dim, scale)

    def quantize_fsq(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = self.bound(z)
        half_width = self._levels // 2 # Renormalize to [-1, 1].
        return quantized / half_width

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    NUM = 4096
    z_0 = torch.rand(NUM) * 4 - 2
    z_1 = torch.rand(NUM) * 4 - 2
    z = torch.stack([z_0, z_1], dim=-1)

    fsq = SphericalFSQ(levels=[10, 9])

    with torch.no_grad():
        quant, _ = fsq(z[:, None])
        quant = quant[:, 0]

    point = quant.numpy()

    # point = fsq.implicit_codebook.numpy()

    figure = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(point[:, 0], point[:, 1], point[:, 2])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.show()