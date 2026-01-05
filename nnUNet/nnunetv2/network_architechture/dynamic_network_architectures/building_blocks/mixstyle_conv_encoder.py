import torch
from torch import nn
import numpy as np
from typing import Union, Type, List, Tuple

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op

class MixStyle(nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6, training: bool = False):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.training = training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only apply during training, and with probability p
        if (not self.training) or (torch.rand(1, device=x.device).item() > self.p):
            return x

        if x.dim() not in (4, 5):
            raise ValueError(f"MixStyle expects 4D or 5D tensor, got shape {tuple(x.shape)}")

        B = x.size(0)
        if B < 2:
            return x

        # Spatial dims: (H, W) or (D, H, W)
        # Reduce over spatial dims, keep (B, C, 1, 1) or (B, C, 1, 1, 1)
        reduce_dims = tuple(range(2, x.dim()))

        mu = x.mean(dim=reduce_dims, keepdim=True)
        var = x.var(dim=reduce_dims, keepdim=True, unbiased=False)
        sigma = (var + self.eps).sqrt()

        # Normalize to remove current style
        x_normed = (x - mu) / sigma

        # Shuffle across the batch
        perm = torch.randperm(B, device=x.device)
        mu2, sigma2 = mu[perm], sigma[perm]

        # Sample mixing coefficient lambda ~ Beta(alpha, alpha), per-sample
        # Shape: (B, 1, 1, 1) or (B, 1, 1, 1, 1)
        # Use Gamma trick to avoid torch.distributions overhead
        g1 = torch._standard_gamma(torch.full((B, 1), self.alpha, device=x.device))
        g2 = torch._standard_gamma(torch.full((B, 1), self.alpha, device=x.device))
        lam = g1 / (g1 + g2 + self.eps)  # (B,1)

        # Expand lambda to broadcast on feature map
        # target shape: (B, 1, 1, 1) or (B, 1, 1, 1, 1)
        view_shape = [B, 1] + [1] * (x.dim() - 2)
        lam = lam.view(*view_shape)

        mu_mix = lam * mu + (1.0 - lam) * mu2
        sigma_mix = lam * sigma + (1.0 - lam) * sigma2

        # Re-apply mixed style
        return x_normed * sigma_mix + mu_mix

class MixstyleConvEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 nonlin_first: bool = False,
                 pool: str = 'conv'
                 ):

        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                             "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        stages = []
        for s in range(n_stages):
            stage_modules = []
            if pool == 'max' or pool == 'avg':
                if (isinstance(strides[s], int) and strides[s] != 1) or \
                        isinstance(strides[s], (tuple, list)) and any([i != 1 for i in strides[s]]):
                    stage_modules.append(get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides[s], stride=strides[s]))
                conv_stride = 1
            elif pool == 'conv':
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            stage_modules.append(StackedConvBlocks(
                n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ))
            stages.append(nn.Sequential(*stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes
        self.mixstyle = MixStyle(p=0.5, alpha=0.1, eps=1e-6, training=False)

    def set_mixstyle_training(self, training: bool):
        self.mixstyle.training = training

    def forward(self, x):
        ret = []
        for i, s in enumerate(self.stages):
            x = s(x)
            ret.append(x)
            # add mixstyle except for the first and last stage
            if i < len(self.stages) - 1 and i > 0:
                x = self.mixstyle(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        output += self.stages[s][-1].compute_conv_feature_map_size(input_size)
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output


