

import torch
from typing import Union, List

class ConvLnRelu1dBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        ln_num_feats: int,
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
        self.norm = torch.nn.LayerNorm(ln_num_feats)

    def forward(self, x:List[torch.Tensor]):
        
        x = self.conv(x)
        x = self.norm(x)
        x = self.ReLU(x)

        return x

