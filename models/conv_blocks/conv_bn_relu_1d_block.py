

import torch
from typing import Union, List

class ConvBnRelu1dBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

        self.ReLU = torch.nn.ReLU(inplace=True)
        self.norm = torch.nn.BatchNorm1d(out_channels, track_running_stats=True)

    def forward(self, x:List[torch.Tensor]):

        x = torch.cat(x, dim=-1)
        x = self.conv(x)
        x = self.norm(x)
        x = self.ReLU(x)

        return x

