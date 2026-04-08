"""
Vendored minimal VideoPose3D temporal-convolution model.

Source: https://github.com/facebookresearch/VideoPose3D
License: CC-BY-NC 4.0 (Facebook Research)

Only the TemporalModel class is included — the minimum needed to load the
pretrained checkpoint and run 2D→3D inference.
"""

import torch
import torch.nn as nn


class TemporalModelBase(nn.Module):
    """
    Base class — do not instantiate directly.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal, dropout, channels):
        super().__init__()

        for fw in filter_widths:
            assert fw % 2 != 0, f"Only odd filter widths are supported, got {fw}"

        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths

        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self.pad = [filter_widths[0] // 2]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)

    def receptive_field(self) -> int:
        """Return the total receptive field in number of frames."""
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2 * frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, frames, num_joints_in, in_features)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, out_frames, num_joints_out, 3)``.
        """
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features

        sz = x.shape[:3]
        x = x.view(x.shape[0], x.shape[1], -1)   # (B, T, J*F)
        x = x.permute(0, 2, 1)                    # (B, J*F, T)

        x = self._forward_blocks(x)

        x = x.permute(0, 2, 1)                    # (B, T_out, J_out*3)
        x = x.view(sz[0], -1, self.num_joints_out, 3)
        return x


class TemporalModel(TemporalModelBase):
    """
    Non-causal temporal convolutional model for 3D pose estimation.

    Uses dilated convolutions WITHOUT internal padding — the temporal
    dimension naturally shrinks by ``receptive_field - 1`` frames.
    To get full-length output, pad the input externally.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024):
        super().__init__(num_joints_in, in_features, num_joints_out,
                         filter_widths, causal, dropout, channels)

        self.expand_conv = nn.Conv1d(
            num_joints_in * in_features, channels,
            filter_widths[0], bias=False,
        )

        layers_conv = []
        layers_bn = []

        self.causal_shift = [(filter_widths[0]) // 2 if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append(
                (filter_widths[i] // 2 * next_dilation) if causal else 0
            )

            layers_conv.append(nn.Conv1d(
                channels, channels,
                filter_widths[i], dilation=next_dilation, bias=False,
            ))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def _forward_blocks(self, x):
        # Expand: (B, J*F, T) → (B, channels, T-2)  (kernel=3, no padding)
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        for i in range(len(self.pad) - 1):
            pad = self.pad[i + 1]
            shift = self.causal_shift[i + 1]

            # Crop residual to match the smaller output of the dilated conv
            res = x[:, :, pad + shift: x.shape[2] - pad + shift]

            # Dilated conv WITHOUT padding → shrinks temporal dim by 2*pad
            x = self.drop(self.relu(
                self.layers_bn[2 * i](self.layers_conv[2 * i](x))
            ))
            # 1×1 conv (same temporal dim) + residual
            x = res + self.drop(self.relu(
                self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))
            ))

        x = self.shrink(x)
        return x
