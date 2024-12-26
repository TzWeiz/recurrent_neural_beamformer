from stft_module import STFT_module


from typing import *

import torch
from torch.nn.common_types import _size_2_t
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


class BaseMemoryEfficientModule(torch.nn.Module):
    def __init__(self, memory_efficient=False):
        super().__init__()
        self.memory_efficient = memory_efficient
        if memory_efficient:
            # see https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/11
            # for why dummy tensor is needed
            self.dummy_tensor = torch.ones(1, requires_grad=True)

    def forward_func(self, inputs):
        raise NotImplementedError("Implement the forward_func like a forward")

    def call_checkpoint_apply(self, input: torch.Tensor):
        def closure(dummy_tensor, _input):
            return self.forward_func(_input)

        return checkpoint(closure, self.dummy_tensor, input)

    def forward(self, x):
        # print(x)

        if hasattr(self, "memory_efficient"):
            if self.memory_efficient:
                if torch.jit.is_scripting():
                    raise Exception("Memory Efficient not supported in JIT")
                x = self.call_checkpoint_apply(x)
            else:
                x = self.forward_func(x)
        else:
            x = self.forward_func(x)
        # x = self.base_conv.forward(x)
        return x


class Conv_BN_ReLU_2dBlock(BaseMemoryEfficientModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        memory_efficient=False,
    ):
        super().__init__(memory_efficient)
        # super().__init__()
        self.conv = torch.nn.Conv2d(
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
        self.norm = torch.nn.BatchNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        x = self.ReLU(x)

        return x

class Conv_BN_ReLU_1dBlock(BaseMemoryEfficientModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        memory_efficient=False,
    ):
        super().__init__(memory_efficient)
        # super().__init__()
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

    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        x = self.ReLU(x)

        return x


class Conv_freq_norm_ReLU_1dBlock(BaseMemoryEfficientModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        memory_efficient=False,
    ):
        super().__init__(memory_efficient)
        # super().__init__()
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
        # self.norm = torch.nn.BatchNorm1d(out_channels, track_running_stats=True)
        self.norm = torch.nn.LayerNorm(1025, elementwise_affine=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        x = self.ReLU(x)

        return x

class Concat_DeConv_freq_norm_ReLU_1dBlock(BaseMemoryEfficientModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_2_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        memory_efficient=True,
    ):
        super().__init__(memory_efficient)
        # super().__init__()
        self.deconv = torch.nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.ReLU = torch.nn.ReLU(inplace=True)
        self.norm = torch.nn.LayerNorm(1025, elementwise_affine=True)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = self.deconv(x)
        x = self.norm(x)
        x = self.ReLU(x)

        return x



class Conv_BN_LeakyReLU_2dBlock(BaseMemoryEfficientModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        memory_efficient=False,
    ):
        super().__init__(memory_efficient)
        # super().__init__()
        self.conv = torch.nn.Conv2d(
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

        self.ReLU = torch.nn.LeakyReLU(inplace=True)
        self.norm = torch.nn.BatchNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        x = self.ReLU(x)

        return x


class Concat_DeConv_BN_ReLU_2dBlock(BaseMemoryEfficientModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_2_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        memory_efficient=True,
    ):
        super().__init__(memory_efficient)
        # super().__init__()
        self.deconv = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.ReLU = torch.nn.ReLU(inplace=True)
        self.norm = torch.nn.BatchNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = self.deconv(x)
        x = self.norm(x)
        x = self.ReLU(x)

        return x


class Concat_DeConv_BN_ReLU_1dBlock(BaseMemoryEfficientModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_2_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        memory_efficient=True,
    ):
        super().__init__(memory_efficient)
        # super().__init__()
        self.deconv = torch.nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.ReLU = torch.nn.ReLU(inplace=True)
        self.norm = torch.nn.BatchNorm1d(out_channels, track_running_stats=True)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = self.deconv(x)
        x = self.norm(x)
        x = self.ReLU(x)

        return x




class Concat_DeConv_BN_LeakyReLU_2dBlock(BaseMemoryEfficientModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_2_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        memory_efficient=True,
    ):
        super().__init__(memory_efficient)
        # super().__init__()
        self.deconv = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.ReLU = torch.nn.LeakyReLU(inplace=True)
        self.norm = torch.nn.BatchNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = self.deconv(x)
        x = self.norm(x)
        x = self.ReLU(x)

        return x


class MultiChannel_SMoLNet_Conv_Deconv_Unet_BN(torch.nn.Module):
    def __init__(
        self,
        STFT: STFT_module,
        n_mics: int,
        n_kernels=64,
        kernel_size=3,
        memory_efficient=False,
        **kwargs,
    ):
        super().__init__()

        self.STFT = STFT

        in_channels = n_mics * 2
        out_channels = n_kernels
        n_dilated_layers = 10

        self.n_dilated_layers = n_dilated_layers

        self.frequency_feature_extractor = torch.nn.Sequential()
        self.frequency_feature_deconv_extractor = torch.nn.Sequential()

        for i in range(0, n_dilated_layers):
            d = 2 ** i
            # to make it work for non-kernel 3
            self.frequency_feature_extractor.add_module(
                "Conv:{}".format(i),
                Conv_BN_ReLU_2dBlock(
                    in_channels,
                    out_channels,
                    (1, kernel_size),
                    padding=(0, d),
                    dilation=(1, d),
                    memory_efficient=memory_efficient,
                ),
            )

            in_channels = out_channels

        for i in range(n_dilated_layers - 1, -1, -1):
            d = 2 ** i
            # to make it work for non-kernel 3
            if i != n_dilated_layers - 1:
                in_channels = 2 * n_kernels

            self.frequency_feature_deconv_extractor.add_module(
                "DeConv:{}".format(i),
                Concat_DeConv_BN_ReLU_2dBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, kernel_size),
                    padding=(0, d),
                    # output_padding=(0, d),
                    dilation=(1, d),
                    memory_efficient=memory_efficient,
                ),
            )

        self.filter_selector = torch.nn.Conv2d(n_kernels, 2, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: batch_size, n_chns, n_samples
        x = self.STFT(x)  # x: batch_size, n_chns, 2, n_frames, n_freqs

        N, n_chns, _, n_frames, n_freq = x.shape
        x = x.reshape(N, n_chns * 2, n_frames, n_freq)
        x = self.apply_network(x)
        x = x.reshape(N, 1, 2, n_frames, n_freq)
        x = self.STFT.backward(x)
        x = x.squeeze(1)
        return x

    def apply_network(self, x: torch.Tensor):
        device = x.device
        out = []
        for i, layer in enumerate(self.frequency_feature_extractor):
            x = layer(x)
            if i < self.n_dilated_layers - 1:
                out.append(x.to("cpu"))

        for i, layer in enumerate(self.frequency_feature_deconv_extractor):

            if i == 0:
                x = layer([x])
            else:
                x = layer([out.pop().to(device), x])

        x = self.filter_selector(x)

        return x


class MIMO_DU_Net_BN(torch.nn.Module):
    def __init__(
        self,
        STFT: STFT_module,
        n_mics: int,
        ref_mic_idx: int,
        n_kernels=64,
        kernel_size=3,
        memory_efficient=False,
        **kwargs,
    ):
        super().__init__()

        self.STFT = STFT

        in_channels = n_mics * 2
        out_channels = n_kernels
        n_dilated_layers = 10

        self.n_dilated_layers = n_dilated_layers
        self.ref_mic_idx = ref_mic_idx
        self.frequency_feature_extractor = torch.nn.Sequential()
        self.frequency_feature_deconv_extractor = torch.nn.Sequential()

        for i in range(0, n_dilated_layers):
            d = 2 ** i
            # to make it work for non-kernel 3
            self.frequency_feature_extractor.add_module(
                "Conv:{}".format(i),
                Conv_BN_ReLU_2dBlock(
                    in_channels,
                    out_channels,
                    (1, kernel_size),
                    padding=(0, d),
                    dilation=(1, d),
                    memory_efficient=memory_efficient,
                ),
            )

            in_channels = out_channels

        for i in range(n_dilated_layers - 1, -1, -1):
            d = 2 ** i
            # to make it work for non-kernel 3
            if i != n_dilated_layers - 1:
                in_channels = 2 * n_kernels

            self.frequency_feature_deconv_extractor.add_module(
                "DeConv:{}".format(i),
                Concat_DeConv_BN_ReLU_2dBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, kernel_size),
                    padding=(0, d),
                    # output_padding=(0, d),
                    dilation=(1, d),
                    memory_efficient=memory_efficient,
                ),
            )

        self.filter_selector = torch.nn.Conv2d(n_kernels, 2 * n_mics, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: batch_size, n_chns, n_samples
        x = x["mixture"]
        x = self.STFT(x)  # x: batch_size, n_chns, 2, n_frames, n_freqs

        N, n_chns, _, n_frames, n_freq = x.shape
        x = x.reshape(N, n_chns * 2, n_frames, n_freq)
        x = self.apply_network(x)
        x = x.reshape(N, n_chns, 2, n_frames, n_freq)
        x = self.STFT.backward(x)
        x = x.squeeze(1)
        # x = x
        output = {
            "est_target_multi_channel": x,
            "est_target": x[:, self.ref_mic_idx],
        }
        return output

    def apply_network(self, x: torch.Tensor):
        out = []
        for i, layer in enumerate(self.frequency_feature_extractor):
            x = layer(x)
            if i < self.n_dilated_layers - 1:
                out.append(x)

        for i, layer in enumerate(self.frequency_feature_deconv_extractor):

            if i == 0:
                x = layer([x])
            else:
                x = layer([out.pop(), x])

        x = self.filter_selector(x)

        return x


class MIMO_DU_Net_BN_per_frame(torch.nn.Module):
    def __init__(
        self,
        STFT: STFT_module,
        n_mics: int,
        ref_mic_idx: int,
        n_kernels=64,
        kernel_size=3,
        memory_efficient=False,
        **kwargs,
    ):
        super().__init__()

        self.STFT = STFT

        in_channels = n_mics * 2
        out_channels = n_kernels
        n_dilated_layers = 10

        self.n_dilated_layers = n_dilated_layers
        self.ref_mic_idx = ref_mic_idx
        self.frequency_feature_extractor = torch.nn.Sequential()
        self.frequency_feature_deconv_extractor = torch.nn.Sequential()

        for i in range(0, n_dilated_layers):
            d = 2 ** i
            # to make it work for non-kernel 3
            self.frequency_feature_extractor.add_module(
                "Conv:{}".format(i),
                Conv_BN_ReLU_2dBlock(
                    in_channels,
                    out_channels,
                    (1, kernel_size),
                    padding=(0, d),
                    dilation=(1, d),
                    memory_efficient=memory_efficient,
                ),
            )

            in_channels = out_channels

        for i in range(n_dilated_layers - 1, -1, -1):
            d = 2 ** i
            # to make it work for non-kernel 3
            if i != n_dilated_layers - 1:
                in_channels = 2 * n_kernels

            self.frequency_feature_deconv_extractor.add_module(
                "DeConv:{}".format(i),
                Concat_DeConv_BN_ReLU_2dBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, kernel_size),
                    padding=(0, d),
                    # output_padding=(0, d),
                    dilation=(1, d),
                    memory_efficient=memory_efficient,
                ),
            )

        self.filter_selector = torch.nn.Conv2d(n_kernels, 2 * n_mics, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: batch_size, n_chns, n_samples
        x = x["mixture"]
        x = self.STFT(x)  # x: batch_size, n_chns, 2, n_frames, n_freqs

        N, n_chns, _, n_frames, n_freq = x.shape
        x = x.reshape(N, n_chns * 2, n_frames, n_freq)
        x = self.apply_network(x)

        x = torch.cat([x[:, :, t : t + 1] for t in range(n_frames)], dim=2)

        x = x.reshape(N, n_chns, 2, n_frames, n_freq)
        x = self.STFT.backward(x)
        x = x.squeeze(1)
        x = x
        output = {
            "est_target_multi_channel": x,
            "est_target": x[:, self.ref_mic_idx],
        }
        return output

    def apply_network(self, x: torch.Tensor):
        out = []
        for i, layer in enumerate(self.frequency_feature_extractor):
            x = layer(x)
            if i < self.n_dilated_layers - 1:
                out.append(x)

        for i, layer in enumerate(self.frequency_feature_deconv_extractor):

            if i == 0:
                x = layer([x])
            else:
                x = layer([out.pop(), x])

        x = self.filter_selector(x)

        return x


class MIMO_DU_Net_BN_per_frame_corrected(torch.nn.Module):
    def __init__(
        self,
        STFT: STFT_module,
        n_mics: int,
        ref_mic_idx: int,
        n_kernels=64,
        kernel_size=3,
        memory_efficient=False,
        n_input_stfts=1,
        **kwargs,
    ):
        super().__init__()

        self.STFT = STFT

        in_channels = n_mics * 2 * n_input_stfts
        out_channels = n_kernels
        n_dilated_layers = 10

        self.n_dilated_layers = n_dilated_layers
        self.ref_mic_idx = ref_mic_idx
        self.frequency_feature_extractor = torch.nn.Sequential()
        self.frequency_feature_deconv_extractor = torch.nn.Sequential()

        for i in range(0, n_dilated_layers):
            d = 2 ** i
            # to make it work for non-kernel 3
            self.frequency_feature_extractor.add_module(
                "Conv:{}".format(i),
                Conv_BN_ReLU_2dBlock(
                    in_channels,
                    out_channels,
                    (1, kernel_size),
                    padding=(0, d),
                    dilation=(1, d),
                    memory_efficient=memory_efficient,
                ),
            )

            in_channels = out_channels

        for i in range(n_dilated_layers - 1, -1, -1):
            d = 2 ** i
            # to make it work for non-kernel 3
            if i != n_dilated_layers - 1:
                in_channels = 2 * n_kernels

            self.frequency_feature_deconv_extractor.add_module(
                "DeConv:{}".format(i),
                Concat_DeConv_BN_ReLU_2dBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, kernel_size),
                    padding=(0, d),
                    # output_padding=(0, d),
                    dilation=(1, d),
                    memory_efficient=memory_efficient,
                ),
            )

        self.filter_selector = torch.nn.Conv2d(n_kernels, 2 * n_mics, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: batch_size, n_chns, n_samples
        x = x["mixture"]
        x = self.STFT(x)  # x: batch_size, n_chns, 2, n_frames, n_freqs

        N, n_chns, _, n_frames, n_freq = x.shape
        x = x.reshape(N, n_chns * 2, n_frames, n_freq)

        x = torch.cat(
            [self.apply_network(x[:, :, t : t + 1]) for t in range(n_frames)], dim=2
        )

        x = x.reshape(N, n_chns, 2, n_frames, n_freq)
        x = self.STFT.backward(x)
        x = x.squeeze(1)
        x = x
        output = {
            "est_target_multi_channel": x,
            "est_target": x[:, self.ref_mic_idx],
        }
        return output

    def apply_network(self, x: torch.Tensor):

        out = []
        for i, layer in enumerate(self.frequency_feature_extractor):
            x = layer(x)
            if i < self.n_dilated_layers - 1:
                out.append(x)

        for i, layer in enumerate(self.frequency_feature_deconv_extractor):

            if i == 0:
                x = layer([x])
            else:
                x = layer([out.pop(), x])

        x = self.filter_selector(x)

        return x
