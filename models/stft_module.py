#%%
import torch
import soundfile as sf
from functools import partial
from typing import *
import pytest


class STFT(torch.nn.Module):
    def __init__(
        self,
        n_fft: int,
        window: torch.Tensor,
        hop_length: Optional[int] = None,
        pad_mode: str = "constant",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        return_complex: bool = False,
    ):
        super().__init__()
        # self.window = window
        self.register_buffer("window", window)
        self.win_length = len(window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        self.return_complex = return_complex

    def forward(self, x: torch.Tensor):
        return self.stft(x)

    def backward(self, X: torch.Tensor, length: Optional[int]):
        return self.istft(X, length)

    def stft(self, x: torch.Tensor):
        """
        Take a time-series signal of size (...,\tau), and outputs the freq-time representation of the signal (..., T, F)

        Parameters
        ----------
        x : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """

        #

        X = torch.stft(
            x,
            window=self.window,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            pad_mode=self.pad_mode,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=self.return_complex,
        )

        return X

    # @torch.jit.export
    def istft(self, X, length: Optional[int]):

        x = torch.istft(
            X,
            length=length,
            n_fft=self.n_fft,
            window=self.window,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=self.return_complex,
        )

        return x


class STFT_module(STFT):
    def __init__(
        self,
        n_fft: int,
        window_func: str or Callable[int],
        window_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        pad_mode: str = "constant",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        return_complex: bool = False,
    ):

        if isinstance(window_func, str):
            window_func = getattr(torch, window_func)
        elif not callable(window_func):
            raise ValueError("window_func must be a str or callable")

        if window_length is None:
            window_length = n_fft

        window = window_func(window_length)
        win_length = len(window)

        if hop_length is None:
            hop_length = n_fft // 4
        n_overlap = window_length - hop_length
        ola_const = self.get_ola_const(window, window_length, n_overlap)

        window = window / ola_const

        super().__init__(
            n_fft,
            window,
            hop_length,
            pad_mode,
            center,
            normalized,
            onesided,
            return_complex,
        )
        self.prev_length = 0

    def forward(self, x: torch.Tensor):
        """
        Perform short-time fourier transform

        Parameters
        ----------
        x : torch.Tensor (*, n_samples)


        Returns
        -------
        STFT(x) torch.Tensor (*, (2 if not complex), n_frames, n_freq)
            The freq-time representation of x.
        """
        ndim = x.ndim
        shape = x.shape
        if ndim > 2:
            x = x.flatten(start_dim=0, end_dim=ndim - 2)
            # KIV: get the first few dims names

        assert x.ndim == 2
        self.prev_length = x.shape[-1]

        X = self.stft(x)  # *, n_freq, n_frames, (2 if not complex)

        if ndim > 2:
            prev_shape = shape[0 : ndim - 1]
            new_shape = tuple((*prev_shape, *X.shape[1:]))
            X = X.reshape(new_shape)

        if not self.return_complex:
            X = X.transpose(-3, -1)
        else:
            X = X.transpose(-2, -1)

        return X

    @staticmethod
    def get_ola_const(window: torch.Tensor, win_len: int, n_overlap: int):
        """Get Over-lap added constant of given STFT params

        Parameters
        ----------
        win_func: callable
            Window function
        win_len: int
            Window length
        n_overlap: int
            Number of overlapping segment.

        Returns
        -------
        ola: float
            COLA constant == sqrt(sum of squared window)

        """
        # window = win_func(win_len)
        hop_size = win_len - n_overlap
        swin = torch.zeros(size=(n_overlap + win_len,))
        for i in range((win_len) // hop_size):
            swin[i * hop_size : i * hop_size + win_len] = (
                swin[i * hop_size : i * hop_size + win_len] + window * window
            )

        ola = swin[n_overlap:-n_overlap]

        if not torch.allclose(ola, ola[0]):
            raise ValueError("The current STFT params do not satisfy COLA condition.")

        return torch.sqrt(ola[0])

    def backward(self, X, length=None):

        if not self.return_complex:
            X = X.transpose(-3, -1)
        else:
            X = X.transpose(-2, -1)

        ndim = X.ndim
        shape = X.shape
        if ndim > 3 and self.return_complex or ndim > 4 and not self.return_complex:

            if self.return_complex:
                X = X.flatten(start_dim=0, end_dim=ndim - 3)  # *, T, F or *, T, F, 2
            else:
                X = X.flatten(start_dim=0, end_dim=ndim - 4)  # *, T, F or *, T, F, 2
            # KIV: get the first few dims names
        assert (X.ndim == 3 and self.return_complex) or (
            X.ndim == 4 and not self.return_complex
        )

        if length is None:
            length = self.prev_length

        x = self.istft(X, length)
        if (ndim > 3 and self.return_complex) or (ndim > 4 and not self.return_complex):

            if self.return_complex:
                new_shape = (*shape[0 : ndim - 2], x.shape[-1])
            else:
                new_shape = (*shape[0 : ndim - 3], x.shape[-1])

            x = x.reshape(new_shape)

        return x


def test_stft_module(wav_file):

    audio, fs = sf.read(wav_file)

    stft = STFT_module(
        window_name="hamming_window", n_fft=2048, hop_length=2048 // 4, center=True
    )

    audio = torch.Tensor(audio)
    if torch.cuda.is_available():
        audio = audio.cuda()
    audio = audio.transpose(-1, -2)  # C, tau
    X = stft.forward(audio)  # C, F, T, 2

    x = stft.backward(X)
    x = x.rename(None)
    assert torch.allclose(x, audio)


def test_stft_batch_module(wav_file):
    batch_size = 8
    audio, fs = sf.read(wav_file)

    stft = STFT_module(
        window_name="hamming_window",
        n_fft=2048,
        hop_length=2048 // 4,
        center=True,
    ).cuda()

    audio = torch.Tensor(audio)

    audio = torch.stack(
        [audio] * batch_size,
    )
    N, tau, C = audio.shape
    if torch.cuda.is_available():
        audio = audio.cuda()
    audio = audio.transpose(-1, -2)  # N, C, tau
    assert audio.shape == (N, C, tau)
    X = stft.forward(audio)  # N, C, F, T, 2

    assert X.shape[0:2] == (
        N,
        C,
    )
    x = stft.backward(X, tau)
    x = x.rename(None)
    assert torch.allclose(x, audio)


# def test_stft_jit_batch_module(wav_file):
#     batch_size = 8
#     audio, fs = sf.read(wav_file)

#     stft = STFT_module(
#         window_name="hamming_window",
#         n_fft=2048,
#         hop_length=2048 // 4,
#         center=True,
#         use_jit=True,
#     )

#     audio = torch.Tensor(audio)

#     audio = torch.stack(
#         [audio] * batch_size,
#     )
#     N, tau, C = audio.shape
#     if torch.cuda.is_available():
#         audio = audio.cuda()
#     audio = audio.transpose(-1, -2)  # N, C, tau
#     assert audio.shape == (N, C, tau)
#     X = stft.forward(audio)  # N, C, F, T, 2

#     assert X.shape[0:2] == (
#         N,
#         C,
#     )
#     x = stft.backward(X, tau)
#     x = x.rename(None)
#     assert torch.allclose(x, audio)


def profile_module_speed(wav_file, stft_cfgs, n=30):
    batch_size = 8
    audio, fs = sf.read(wav_file)

    stft = STFT_module(**stft_cfgs)

    audio = torch.Tensor(audio)

    audio = torch.stack(
        [audio] * batch_size,
    )
    N, tau, C = audio.shape
    if torch.cuda.is_available():
        stft.cuda()
        audio = audio.cuda()
    audio = audio.transpose(-1, -2)  # N, C, tau

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=n),
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_flops=True,
        # on_trace_ready=torch.profiler.tensorboard_trace_handler,
        # with_trace=True,
    ) as prof:
        # with torch.profiler.record_function("stft and istft"):
        for i in range(n):
            X = stft.forward(audio)  # N, C, F, T, 2
            x = stft.backward(X, tau)
            prof.step()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))


if __name__ == "__main__":

    from common_args import wav_file, stft_cfgs

    # import torch.autograd.profiler as profiler

    # test_stft_module(wav_file)
    test_stft_batch_module(wav_file)

    print("Test passed.")

    print("Profiling speed")
    profile_module_speed(wav_file, stft_cfgs)


# %%
