"""
This code contains oracle beamformers and neural networks that was trained with oracle time-varying signal

[extended_summary]

Returns
-------
[type]
    [description]
"""

from asteroid import ConvTasNet
from models.smolnet import SMoLNet_freq_only_with_instance_norm_single_channel

from models.complex_net_tcn_dense_unet_16k import CSeqUNetDense_Singlechannel
from typing import *

from beamformers.adaptive_beamformer import AdaptiveBeamformer
from beamformers.TI_MVDR import TI_MVDR

import torch
from torch import nn
import math


from .beamformers_ops import (
    torch_spatial_covariance,
    multichannel_mpdr_rank_1_with_inverse,
    multichannel_wf_rank_1,
    apply_beamformer,
)
from abc import abstractmethod

from torch_complex_ops import complex_inverse, conj, complex_mm


class Oracle_Adaptive_Beamformer(AdaptiveBeamformer):
    def forward(self, input: Dict[str, torch.Tensor]):
        y, x, v = input["mixture"], input["target"], input["noise"]

        return super().forward(y, x_hat=x, v_hat=v)


class Oracle_TI_MVDR(Oracle_Adaptive_Beamformer, TI_MVDR):
    pass

class Oracle_TI_MVDR(Oracle_Adaptive_Beamformer):
    # def __init__(self,)
    def __init__(
        self, STFT, ref_mic_idx, diag_loadings: float = 0.001, **kwargs
    ):
        super().__init__(STFT, ref_mic_idx)

        self.diag_loadings = diag_loadings

    def forward_in_STFT(self, Y, X, V_hat):


        device = Y.device
        Y = Y.transpose(1, 2)  # N, 2, C, T, F
        X = X.transpose(1, 2)  # N, 2, C, T, F
        N, _, C, T, F = Y.shape

        eye = torch.eye(C).to(device)
        eye = eye.to(device)

        phi_yy = torch_spatial_covariance(Y)  # shape of (N,2,F,T,C,C)
        phi_yy = phi_yy + self.diag_loadings * eye  # diagonal loading

        phi_yy_inv = complex_inverse(phi_yy, RI_dim=1)  # shape of (N,2,F,T,C,C)

        phi_xx = torch_spatial_covariance(X)  # shape of (N,2,F,T,C,C)

        W_bf = multichannel_mpdr_rank_1_with_inverse(phi_xx, phi_yy_inv)
        X_bf = apply_beamformer(W_bf, Y)  # N, 2, C, T, F

        X_bf = X_bf.transpose(1, 2)  # N, C, 2, T, F
        return X_bf


class Oracle_TI_MWF(Oracle_Adaptive_Beamformer):
    # def __init__(self,)
    def __init__(
        self, STFT, ref_mic_idx, diag_loadings: float = 0.001, **kwargs
    ):
        super().__init__(STFT, ref_mic_idx)

        self.diag_loadings = diag_loadings

    def forward_in_STFT(self, Y, X, V_hat):


        device = Y.device
        Y = Y.transpose(1, 2)  # N, 2, C, T, F
        X = X.transpose(1, 2)  # N, 2, C, T, F
        N, _, C, T, F = Y.shape

        eye = torch.eye(C).to(device)
        eye = eye.to(device)

        phi_yy = torch_spatial_covariance(Y)  # shape of (N,2,F,T,C,C)
        phi_yy = phi_yy + self.diag_loadings * eye  # diagonal loading

        phi_yy_inv = complex_inverse(phi_yy, RI_dim=1)  # shape of (N,2,F,T,C,C)

        phi_xx = torch_spatial_covariance(X)  # shape of (N,2,F,T,C,C)

        W_bf = multichannel_wf_rank_1(phi_xx, phi_yy_inv)
        X_bf = apply_beamformer(W_bf, Y)  # N, 2, C, T, F

        X_bf = X_bf.transpose(1, 2)  # N, C, 2, T, F
        return X_bf


class Oracle_TV_MPDR_rank_1(Oracle_Adaptive_Beamformer):
    def __init__(
        self, STFT, ref_mic_idx, diag_loadings: float = 0.001 / math.sqrt(2), **kwargs
    ):
        super().__init__(STFT, ref_mic_idx)

        self.diag_loadings = diag_loadings

    def forward_in_STFT(self, Y, X, V):

        device = Y.device
        Y = Y.transpose(1, 2)  # N, 2, C, T, F
        X = X.transpose(1, 2)  # N, 2, C, T, F
        N, _, C, T, F = Y.shape

        eye = torch.eye(C).to(device)
        eye = eye.to(device)

        phi_yy = torch_spatial_covariance(Y)  # shape of (N,2,F,T,C,C)
        phi_yy = phi_yy + self.diag_loadings * eye  # diagonal loading

        phi_yy_inv = complex_inverse(phi_yy, RI_dim=1)

        phi_xx = torch_spatial_covariance(X)  # shape of (N,2,F,T,C,C)

        W_bf = multichannel_mpdr_rank_1_with_inverse(phi_xx, phi_yy_inv)

        X_bf = apply_beamformer(W_bf, Y)  # N, 2, C, T, F

        X_bf = X_bf.transpose(1, 2)  # N, C, 2, T, F
        # X_bf = X_bf.transpose(1, 2)  # N, C, 2, T, F

        return X_bf


class Oracle_TV_MPDR_rank_1_vary_x(Oracle_Adaptive_Beamformer):
    def __init__(
        self, STFT, ref_mic_idx, diag_loadings: float = 0.001 / math.sqrt(2), **kwargs
    ):
        super().__init__(STFT, ref_mic_idx)
        # self.STFT = STFT
        # self.ref_mic_idx = ref_mic_idx
        # self.dummy_weight = nn.Parameter(torch.ones(1, requires_grad=True))
        self.diag_loadings = diag_loadings

    def forward_in_STFT(self, Y, X, V):
        device = Y.device
        Y = Y.transpose(1, 2)  # N, 2, C, T, F
        X = X.transpose(1, 2)  # N, 2, C, T, F
        N, _, C, T, F = Y.shape

        eye = torch.eye(C).to(device)
        eye = eye.to(device)

        phi_yy = torch_spatial_covariance(Y)  # shape of (N,2,F,T,C,C)
        phi_yy = torch.mean(phi_yy, dim=3, keepdims=True)
        phi_yy = phi_yy + self.diag_loadings * eye  # diagonal loading

        phi_yy_inv = complex_inverse(phi_yy, RI_dim=1)

        phi_xx = torch_spatial_covariance(X)  # shape of (N,2,F,T,C,C)

        W_bf = multichannel_mpdr_rank_1_with_inverse(phi_xx, phi_yy_inv)

        X_bf = apply_beamformer(W_bf, Y)  # N, 2, C, T, F

        X_bf = X_bf.transpose(1, 2)  # N, C, 2, T, F

        return X_bf


class Oracle_TV_MPDR_rank_1_accumulate_x(Oracle_Adaptive_Beamformer):
    def __init__(
        self,
        STFT,
        ref_mic_idx,
        diag_loadings: float = 0.001 / math.sqrt(2),
        alpha=0.1,
        **kwargs
    ):
        super().__init__(STFT, ref_mic_idx)
        # self.STFT = STFT
        # self.ref_mic_idx = ref_mic_idx
        # self.dummy_weight = nn.Parameter(torch.ones(1, requires_grad=True))
        self.diag_loadings = diag_loadings
        self.alpha = alpha

    def forward_in_STFT(self, Y, X, V):
        device = Y.device
        Y = Y.transpose(1, 2)  # N, 2, C, T, F
        X = X.transpose(1, 2)  # N, 2, C, T, F
        N, _, C, T, F = Y.shape

        eye = torch.eye(C).to(device)
        eye = eye.to(device)

        phi_yy = torch_spatial_covariance(Y)  # shape of (N,2,F,T,C,C)
        phi_yy = torch.mean(phi_yy, dim=3, keepdims=True)
        phi_yy = phi_yy + self.diag_loadings * eye  # diagonal loading

        phi_yy_inv = complex_inverse(phi_yy, RI_dim=1)

        phi_xx = torch_spatial_covariance(X)  # shape of (N,2,F,T,C,C)

        accum_phi_xx = torch.zeros_like(phi_xx)

        for t in range(T):

            phi_xx_t = phi_xx[:, :, :, t : t + 1]
            if t == 0:
                accum_phi_xx_t = phi_xx[:, :, :, t : t + 1]  # N, 2, F, 1, C, C
            else:
                accum_phi_xx_t = (
                    self.alpha * accum_phi_xx_t + (1 - self.alpha) * phi_xx_t
                )
            accum_phi_xx[:, :, :, t : t + 1] = accum_phi_xx_t

        W_bf = multichannel_mpdr_rank_1_with_inverse(
            accum_phi_xx, phi_yy_inv
        )  # shape of (N,2,F,T,C,C)

        # W_bf[:, :, :, 0:1] = torch.eye(C, C, device=device)

        X_bf = apply_beamformer(W_bf, Y)  # N, 2, C, T, F

        X_bf = X_bf.transpose(1, 2)  # N, C, 2, T, F

        return X_bf


class Oracle_AR_MPDR_rank_1(Oracle_Adaptive_Beamformer):
    """

    Computes weight via:

    ..math::
        \mathbf{w}_{\mathsf{BF}}^{(t)}:=\alpha\mathbf{w}_{\mathsf{TV}}+(1-\alpha)\mathbf{w}^{(t-1)}_{\mathsf{BF}},
    where
    ..math::
        \mathbf{w}_{\mathsf{TV}}(t,f)=\frac{\mathbf\Phi_{y}^{-1}(t,f)\mathbf{\Phi}_x(t,f)}{\mathsf{tr}(\mathbf\Phi_{y}^{-1}(t,f)\mathbf{\Phi}_x(t,f))}\mathbf{u}_{m_0}

    Parameters
    ----------
    Oracle_Adaptive_Beamformer : [type]
        [description]
    """

    def __init__(
        self,
        STFT,
        ref_mic_idx,
        diag_loadings: float = 0.001 / math.sqrt(2),
        alpha=0.1,
        **kwargs
    ):
        super().__init__(STFT, ref_mic_idx)
        # self.STFT = STFT
        # self.ref_mic_idx = ref_mic_idx
        # self.dummy_weight = nn.Parameter(torch.ones(1, requires_grad=True))
        self.diag_loadings = diag_loadings
        self.alpha = alpha

    def forward_in_STFT(self, Y, X, V):
        device = Y.device
        Y = Y.transpose(1, 2)  # N, 2, C, T, F
        X = X.transpose(1, 2)  # N, 2, C, T, F
        N, _, C, T, F = Y.shape

        eye = torch.eye(C).to(device)
        eye = eye.to(device)

        phi_yy = torch_spatial_covariance(Y)  # shape of (N,2,F,T,C,C)
        # phi_yy = torch.mean(phi_yy, dim=3, keepdims=True)
        phi_yy = phi_yy + self.diag_loadings * eye  # diagonal loading

        phi_yy_inv = complex_inverse(phi_yy, RI_dim=1)

        phi_xx = torch_spatial_covariance(X)  # shape of (N,2,F,T,C,C)
        W_bf = multichannel_mpdr_rank_1_with_inverse(
            phi_xx, phi_yy_inv
        )  # shape of (N,2,F,T,C,C)

        # torch.cumsum()

        for t in range(1, T):
            W_bf[:, :, :, t] = (
                self.alpha * W_bf[:, :, :, t - 1] + (1 - self.alpha) * W_bf[:, :, :, t]
            )

        # W_bf[:, :, :, 0:1] = torch.eye(C, C, device=device)

        X_bf = apply_beamformer(W_bf, Y)  # N, 2, C, T, F

        X_bf = X_bf.transpose(1, 2)  # N, C, 2, T, F

        return X_bf


class Oracle_TV_MPDR_rank_1_accumulate_x_delayed(Oracle_Adaptive_Beamformer):
    def __init__(
        self,
        STFT,
        ref_mic_idx,
        diag_loadings: float = 0.001 / math.sqrt(2),
        alpha=0.1,
        **kwargs
    ):
        super().__init__(STFT, ref_mic_idx)
        # self.STFT = STFT
        # self.ref_mic_idx = ref_mic_idx
        # self.dummy_weight = nn.Parameter(torch.ones(1, requires_grad=True))
        self.diag_loadings = diag_loadings
        self.alpha = alpha

    def forward_in_STFT(self, Y, X, V):
        device = Y.device
        Y = Y.transpose(1, 2)  # N, 2, C, T, F
        X = X.transpose(1, 2)  # N, 2, C, T, F
        N, _, C, T, F = Y.shape

        eye = torch.eye(C).to(device)
        eye = eye.to(device)

        phi_yy = torch_spatial_covariance(Y)  # shape of (N,2,F,T,C,C)
        phi_yy = torch.mean(phi_yy, dim=3, keepdims=True)
        phi_yy = phi_yy + self.diag_loadings * eye  # diagonal loading

        phi_yy_inv = complex_inverse(phi_yy, RI_dim=1)

        phi_xx = torch_spatial_covariance(X)  # shape of (N,2,F,T,C,C)

        accum_phi_xx = torch.zeros_like(phi_xx)

        for t in range(1, T):

            phi_xx_t = phi_xx[:, :, :, t : t + 1]
            if t == 1:
                accum_phi_xx_t = phi_xx[:, :, :, t : t + 1]  # N, 2, F, 1, C, C
            else:
                accum_phi_xx_t = (
                    self.alpha * accum_phi_xx_t + (1 - self.alpha) * phi_xx_t
                )
            accum_phi_xx[:, :, :, t : t + 1] = accum_phi_xx_t

        W_bf = multichannel_mpdr_rank_1_with_inverse(
            accum_phi_xx, phi_yy_inv
        )  # shape of (N,2,F,T,C,C)

        W_bf[:, :, :, 0:1] = torch.eye(C, C, device=device)

        X_bf = apply_beamformer(W_bf, Y)  # N, 2, C, T, F

        X_bf = X_bf.transpose(1, 2)  # N, C, 2, T, F

        return X_bf


class Oracle_TV_MPDR_rank_1_vary_y(Oracle_Adaptive_Beamformer):
    def __init__(
        self, STFT, ref_mic_idx, diag_loadings: float = 0.001 / math.sqrt(2), **kwargs
    ):
        super().__init__(STFT, ref_mic_idx)
        self.diag_loadings = diag_loadings

    def forward_in_STFT(self, Y, X, V):

        device = Y.device
        Y = Y.transpose(1, 2)  # N, 2, C, T, F
        X = X.transpose(1, 2)  # N, 2, C, T, F
        N, _, C, T, F = Y.shape

        eye = torch.eye(C).to(device)
        eye = eye.to(device)

        phi_yy = torch_spatial_covariance(Y)  # shape of (N,2,F,T,C,C)

        phi_yy = phi_yy + self.diag_loadings * eye  # diagonal loading

        phi_yy_inv = complex_inverse(phi_yy, RI_dim=1)

        phi_xx = torch_spatial_covariance(X)  # shape of (N,2,F,T,C,C)

        phi_xx = torch.mean(phi_xx, dim=3, keepdims=True)
        W_bf = multichannel_mpdr_rank_1_with_inverse(phi_xx, phi_yy_inv)

        X_bf = apply_beamformer(W_bf, Y)  # N, 2, C, T, F

        X_bf = X_bf.transpose(1, 2)  # N, C, 2, T, F

        return X_bf


class Oracle_TI_MPDR_rank_1(Oracle_Adaptive_Beamformer):
    def __init__(
        self, STFT, ref_mic_idx, diag_loadings: float = 0.001 / math.sqrt(2), **kwargs
    ):
        super().__init__(STFT, ref_mic_idx)
        self.diag_loadings = diag_loadings

    def forward_in_STFT(self, Y, X, V):

        device = Y.device
        Y = Y.transpose(1, 2)  # N, 2, C, T, F
        X = X.transpose(1, 2)  # N, 2, C, T, F
        N, _, C, T, F = Y.shape

        eye = torch.eye(C).to(device)
        eye = eye.to(device)

        phi_yy = torch_spatial_covariance(Y)  # shape of (N,2,F,T,C,C)
        phi_yy = torch.mean(phi_yy, dim=3, keepdims=True)
        phi_yy = phi_yy + self.diag_loadings * eye  # diagonal loading

        phi_yy_inv = complex_inverse(phi_yy, RI_dim=1)

        phi_xx = torch_spatial_covariance(X)  # shape of (N,2,F,T,C,C)
        phi_xx = torch.mean(phi_xx, dim=3, keepdims=True)
        W_bf = multichannel_mpdr_rank_1_with_inverse(phi_xx, phi_yy_inv)

        X_bf = apply_beamformer(W_bf, Y)  # N, 2, C, T, F

        X_bf = X_bf.transpose(1, 2)  # N, C, 2, T, F

        return X_bf


class Oracle_TI_MVDR_rank_1(Oracle_Adaptive_Beamformer):
    def __init__(
        self, STFT, ref_mic_idx, diag_loadings: float = 0.001 / math.sqrt(2), **kwargs
    ):
        super().__init__(STFT, ref_mic_idx)
        self.diag_loadings = diag_loadings

    def forward_in_STFT(self, Y, X, V):

        device = Y.device
        Y = Y.transpose(1, 2)  # N, 2, C, T, F
        X = X.transpose(1, 2)  # N, 2, C, T, F
        N, _, C, T, F = Y.shape

        eye = torch.eye(C).to(device)
        eye = eye.to(device)

        V = Y - X
        phi_vv = torch_spatial_covariance(V)  # shape of (N,2,F,T,C,C)
        phi_vv = torch.mean(phi_vv, dim=3, keepdims=True)
        phi_vv = phi_vv + self.diag_loadings * eye  # diagonal loading

        phi_vv_inv = complex_inverse(phi_vv, RI_dim=1)

        phi_xx = torch_spatial_covariance(X)  # shape of (N,2,F,T,C,C)
        phi_xx = torch.mean(phi_xx, dim=3, keepdims=True)
        W_bf = multichannel_mpdr_rank_1_with_inverse(phi_xx, phi_vv_inv)

        X_bf = apply_beamformer(W_bf, Y)  # N, 2, C, T, F

        X_bf = X_bf.transpose(1, 2)  # N, C, 2, T, F

        return X_bf


class Oracle_TI_MPDR(Oracle_Adaptive_Beamformer):
    def __init__(
        self, STFT, ref_mic_idx, diag_loadings: float = 0.001 / math.sqrt(2), **kwargs
    ):
        super().__init__(STFT, ref_mic_idx)
        self.diag_loadings = diag_loadings

    def compute_steering_vector(self, phi_xx):

        C = phi_xx.shape[-1]
        w, v = torch.linalg.eig(phi_xx)  # eigenvalues, eigenvectors

        # v:  N, 1, F, 1, C, C
        r = v[:, :, :, :, :, 0:1]  # N, 1, F, 1, C, 1

        d = torch.zeros_like(phi_xx)  # N, 1, F, 1, C, C

        for c in range(C):
            d[:, :, :, :, :, c : c + 1] = r / r[:, :, :, :, c : c + 1]

        return d

    def compute_beamformer(self, phi_yy_inv, d):
        W_num = phi_yy_inv @ d
        W_den = torch.conj(d.transpose(-1, -2)) @ W_num

        W_bf = W_num / W_den

        return W_bf

    def forward_in_STFT(self, Y, X, V):

        device = Y.device
        Y = Y.transpose(1, 2)  # N, 2, C, T, F
        X = X.transpose(1, 2)  # N, 2, C, T, F
        N, _, C, T, F = Y.shape

        eye = torch.eye(C).to(device)
        eye = eye.to(device)

        phi_yy = torch_spatial_covariance(Y)  # shape of (N,2,F,T,C,C)
        phi_yy = torch.mean(phi_yy, dim=3, keepdims=True)
        phi_yy = phi_yy + self.diag_loadings * eye  # diagonal loading

        phi_yy_inv = complex_inverse(phi_yy, RI_dim=1)

        phi_yy_inv = phi_yy_inv.unsqueeze(-1)
        phi_yy_inv = phi_yy_inv.transpose(1, -1)  # N, 1, F, 1, C, C, 2
        phi_yy_inv = phi_yy_inv.contiguous()
        phi_yy_inv = torch.view_as_complex(phi_yy_inv)  # N, 1, F, 1, C, C,

        phi_xx = torch_spatial_covariance(X)  # shape of (N,2,F,T,C,C)
        phi_xx = torch.mean(phi_xx, dim=3, keepdims=True)

        # assert torch.allclose(conj(phi_xx.transpose(-1, -2)), phi_xx)  # check hermitian

        phi_xx = phi_xx.unsqueeze(-1)
        phi_xx = phi_xx.transpose(1, -1)  # N, 1, F, 1, C, C, 2
        phi_xx = phi_xx.contiguous()
        phi_xx = torch.view_as_complex(phi_xx)  # N, 1, F, 1, C, C,

        d = self.compute_steering_vector(phi_xx)
        W_bf = self.compute_beamformer(phi_yy_inv, d)

        W_bf = torch.view_as_real(W_bf)  # N, 1, F, 1, C, C, 2

        W_bf = W_bf.transpose(1, -1)  # N, 2, F, 1, C, C, 1
        W_bf = W_bf.squeeze(-1)  # N, 2, F, 1, C, C

        X_bf = apply_beamformer(W_bf, Y)  # N, 2, C, T, F

        X_bf = X_bf.transpose(1, 2)  # N, C, 2, T, F

        return X_bf


class Oracle_TI_MPDR_change_dim_for_d(Oracle_TI_MPDR):
    def __init__(
        self, STFT, ref_mic_idx, diag_loadings: float = 0.001 / math.sqrt(2), **kwargs
    ):
        super().__init__(STFT, ref_mic_idx)
        self.diag_loadings = diag_loadings

    def compute_steering_vector(self, phi_xx):

        C = phi_xx.shape[-1]
        w, v = torch.linalg.eig(phi_xx)  # eigenvalues, eigenvectors

        # v:  N, 1, F, 1, C, C
        r = v[:, :, :, :, :, 0:1]  # N, 1, F, 1, C, 1

        d = torch.zeros_like(phi_xx)  # N, 1, F, 1, C, C

        for c in range(C):
            d[:, :, :, :, c : c + 1] = (r / r[:, :, :, :, c : c + 1]).transpose(-1, -2)

        return d


class Oracle_TI_MPDR_change_selected_principal_vector(Oracle_TI_MPDR):
    def __init__(
        self, STFT, ref_mic_idx, diag_loadings: float = 0.001 / math.sqrt(2), **kwargs
    ):
        super().__init__(STFT, ref_mic_idx)
        self.diag_loadings = diag_loadings

    def compute_steering_vector(self, phi_xx):

        C = phi_xx.shape[-1]
        w, v = torch.linalg.eig(phi_xx)  # eigenvalues, eigenvectors

        # v:  N, 1, F, 1, C, C
        r = v[:, :, :, :, :, -1].unsqueeze(5)  # N, 1, F, 1, C, 1

        d = torch.zeros_like(phi_xx)  # N, 1, F, 1, C, C

        for c in range(C):
            d[:, :, :, :, :, c : c + 1] = r / r[:, :, :, :, c : c + 1]

        return d


class Oracle_TI_MPDR_change_vector_select_row(Oracle_TI_MPDR):
    def __init__(
        self, STFT, ref_mic_idx, diag_loadings: float = 0.001 / math.sqrt(2), **kwargs
    ):
        super().__init__(STFT, ref_mic_idx)
        self.diag_loadings = diag_loadings

    def compute_steering_vector(self, phi_xx):
        C = phi_xx.shape[-1]
        w, v = torch.linalg.eig(phi_xx)  # eigenvalues, eigenvectors

        # v:  N, 1, F, 1, C, C
        r = v[:, :, :, :, 0:1, :]  # N, 1, F, 1, 1, C
        d = torch.zeros_like(phi_xx)  # N, 1, F, 1, C, C

        for c in range(C):
            d[:, :, :, :, :, c : c + 1] = (r / r[:, :, :, :, :, c : c + 1]).transpose(
                -1, -2
            )

        return d


class Oracle_TI_MPDR_change_vector_select_row_selected_principal_vector(Oracle_TI_MPDR):
    def __init__(
        self, STFT, ref_mic_idx, diag_loadings: float = 0.001 / math.sqrt(2), **kwargs
    ):
        super().__init__(STFT, ref_mic_idx)
        self.diag_loadings = diag_loadings

    def compute_steering_vector(self, phi_xx):
        C = phi_xx.shape[-1]
        w, v = torch.linalg.eig(phi_xx)  # eigenvalues, eigenvectors

        # v:  N, 1, F, 1, C, C
        r = v[:, :, :, :, -2:-1, :]  # N, 1, F, 1, 1, C
        d = torch.zeros_like(phi_xx)  # N, 1, F, 1, C, C

        for c in range(C):
            d[:, :, :, :, :, c : c + 1] = (r / r[:, :, :, :, :, c : c + 1]).transpose(
                -1, -2
            )

        return d


class Oracle_TI_MPDR_change_to_no_norm_of_steering_vector(Oracle_TI_MPDR):
    def __init__(
        self, STFT, ref_mic_idx, diag_loadings: float = 0.001 / math.sqrt(2), **kwargs
    ):
        super().__init__(STFT, ref_mic_idx)
        self.diag_loadings = diag_loadings

    def compute_steering_vector(self, phi_xx):

        C = phi_xx.shape[-1]
        w, v = torch.linalg.eig(phi_xx)  # eigenvalues, eigenvectors

        # v:  N, 1, F, 1, C, C
        r = v[:, :, :, :, :, 0:1]  # N, 1, F, 1, C, 1

        d = torch.zeros_like(phi_xx)  # N, 1, F, 1, C, C

        for c in range(C):
            d[:, :, :, :, :, c : c + 1] = r

        return d


class Oracle_TV_steering_vector_MPDR(Oracle_Adaptive_Beamformer):
    def __init__(
        self, STFT, ref_mic_idx, diag_loadings: float = 0.001 / math.sqrt(2), **kwargs
    ):
        super().__init__(STFT, ref_mic_idx)
        self.diag_loadings = diag_loadings

    def forward_in_STFT(self, Y, X, V):

        device = Y.device
        Y = Y.transpose(1, 2)  # N, 2, C, T, F
        X = X.transpose(1, 2)  # N, 2, C, T, F
        N, _, C, T, F = Y.shape

        eye = torch.eye(C).to(device)
        eye = eye.to(device)

        phi_yy = torch_spatial_covariance(Y)  # shape of (N,2,F,T,C,C)
        phi_yy = torch.mean(phi_yy, dim=3, keepdims=True)
        phi_yy = phi_yy + self.diag_loadings * eye  # diagonal loading

        phi_yy_inv = complex_inverse(phi_yy, RI_dim=1)

        phi_yy_inv = phi_yy_inv.unsqueeze(-1)
        phi_yy_inv = phi_yy_inv.transpose(1, -1)  # N, 1, F, 1, C, C, 2
        phi_yy_inv = phi_yy_inv.contiguous()
        phi_yy_inv = torch.view_as_complex(phi_yy_inv)  # N, 1, F, 1, C, C,

        phi_xx = torch_spatial_covariance(X)  # shape of (N,2,F,T,C,C)
        # phi_xx = torch.mean(phi_xx, dim=3, keepdims=True)

        # assert torch.allclose(conj(phi_xx.transpose(-1, -2)), phi_xx)  # check hermitian

        phi_xx = phi_xx.unsqueeze(-1)
        phi_xx = phi_xx.transpose(1, -1)  # N, 1, F, 1, C, C, 2
        phi_xx = phi_xx.contiguous()
        phi_xx = torch.view_as_complex(phi_xx)  # N, 1, F, 1, C, C,

        w, v = torch.linalg.eig(phi_xx)  # eigenvalues, eigenvectors

        # v:  N, 1, F, 1, C, C
        r = v[:, :, :, :, :, 0:1]  # N, 1, F, 1, C, 1

        d = torch.zeros_like(phi_xx)  # N, 1, F, 1, C, C

        for c in range(C):
            d[:, :, :, :, :, c : c + 1] = r / r[:, :, :, :, c : c + 1]

        W_num = phi_yy_inv @ d
        W_den = torch.conj(d.transpose(-1, -2)) @ W_num

        # W_bf = multichannel_mpdr_rank_1_with_inverse(phi_xx, phi_vv_inv) # N, 2, F, 1, C, C
        W_bf = W_num / W_den
        W_bf = torch.view_as_real(W_bf)  # N, 1, F, 1, C, C, 2

        W_bf = W_bf.transpose(1, -1)  # N, 2, F, 1, C, C, 1
        W_bf = W_bf.squeeze(-1)  # N, 2, F, 1, C, C

        X_bf = apply_beamformer(W_bf, Y)  # N, 2, C, T, F

        X_bf = X_bf.transpose(1, 2)  # N, C, 2, T, F

        return X_bf

    # def compute_steering_vector(self, phi_xx):

    #     C = phi_xx.shape[-1]
    #     w, v = torch.linalg.eig(phi_xx)  # eigenvalues, eigenvectors

    #     # v:  N, 1, F, 1, C, C
    #     r = v[:, :, :, :, :, 0:1]  # N, 1, F, 1, C, 1

    #     d = torch.zeros_like(phi_xx)  # N, 1, F, 1, C, C

    #     for c in range(C):
    #         d[:, :, :, :, :, c : c + 1] = r / r[:, :, :, :, c : c + 1]

    #     return d

    # def compute_beamformer(self, phi_vv_inv, d):
    #     W_num = phi_vv_inv @ d
    #     W_den = torch.conj(d.transpose(-1, -2)) @ W_num

    #     W_bf = W_num / W_den

    #     return W_bf

    # def forward_in_STFT(self, Y, X):

    #     device = Y.device
    #     Y = Y.transpose(1, 2)  # N, 2, C, T, F
    #     X = X.transpose(1, 2)  # N, 2, C, T, F
    #     N, _, C, T, F = Y.shape

    #     eye = torch.eye(C).to(device)
    #     eye = eye.to(device)

    #     V = Y - X
    #     phi_vv = torch_spatial_covariance(V)  # shape of (N,2,F,T,C,C)
    #     phi_vv = torch.mean(phi_vv, dim=3, keepdims=True)
    #     phi_vv = phi_vv + self.diag_loadings * eye  # diagonal loading

    #     phi_vv_inv = complex_inverse(phi_vv, RI_dim=1)

    #     phi_vv_inv = phi_vv_inv.unsqueeze(-1)
    #     phi_vv_inv = phi_vv_inv.transpose(1, -1)  # N, 1, F, 1, C, C, 2
    #     phi_vv_inv = phi_vv_inv.contiguous()
    #     phi_vv_inv = torch.view_as_complex(phi_vv_inv)  # N, 1, F, 1, C, C,

    #     phi_xx = torch_spatial_covariance(X)  # shape of (N,2,F,T,C,C)
    #     phi_xx = torch.mean(phi_xx, dim=3, keepdims=True)

    #     # assert torch.allclose(conj(phi_xx.transpose(-1, -2)), phi_xx)  # check hermitian

    #     phi_xx = phi_xx.unsqueeze(-1)
    #     phi_xx = phi_xx.transpose(1, -1)  # N, 1, F, 1, C, C, 2
    #     phi_xx = phi_xx.contiguous()
    #     phi_xx = torch.view_as_complex(phi_xx)  # N, 1, F, 1, C, C,

    #     w, v = torch.linalg.eig(phi_xx)  # eigenvalues, eigenvectors

    #     # v:  N, 1, F, 1, C, C
    #     d = self.compute_steering_vector(phi_xx)

    #     W_bf = self.compute_beamformer(phi_vv_inv, d)
    #     W_bf = torch.view_as_real(W_bf)  # N, 1, F, 1, C, C, 2

    #     W_bf = W_bf.transpose(1, -1)  # N, 2, F, 1, C, C, 1
    #     W_bf = W_bf.squeeze(-1)  # N, 2, F, 1, C, C

    #     X_bf = apply_beamformer(W_bf, Y)  # N, 2, C, T, F

    #     X_bf = X_bf.transpose(1, 2)  # N, C, 2, T, F

    #     return X_bf


class Oracle_TI_MVDR_wo_norm_of_rtf(Oracle_TI_MVDR):
    def __init__(
        self, STFT, ref_mic_idx, diag_loadings: float = 0.001 / math.sqrt(2), **kwargs
    ):
        super().__init__(STFT, ref_mic_idx)
        self.diag_loadings = diag_loadings

    def compute_steering_vector(self, phi_xx):

        C = phi_xx.shape[-1]
        w, v = torch.linalg.eig(phi_xx)  # eigenvalues, eigenvectors

        # v:  N, 1, F, 1, C, C
        r = v[:, :, :, :, :, 0:1]  # N, 1, F, 1, C, 1

        d = torch.zeros_like(phi_xx)  # N, 1, F, 1, C, C

        for c in range(C):
            d[:, :, :, :, :, c : c + 1] = r

        return d


class Oracle_TV_steering_vector_MVDR(Oracle_Adaptive_Beamformer):
    def __init__(
        self, STFT, ref_mic_idx, diag_loadings: float = 0.001 / math.sqrt(2), **kwargs
    ):
        super().__init__(STFT, ref_mic_idx)
        self.diag_loadings = diag_loadings

    def forward_in_STFT(self, Y, X, V):

        device = Y.device
        Y = Y.transpose(1, 2)  # N, 2, C, T, F
        X = X.transpose(1, 2)  # N, 2, C, T, F
        N, _, C, T, F = Y.shape

        eye = torch.eye(C).to(device)
        eye = eye.to(device)

        V = Y - X
        phi_vv = torch_spatial_covariance(V)  # shape of (N,2,F,T,C,C)
        phi_vv = torch.mean(phi_vv, dim=3, keepdims=True)
        phi_vv = phi_vv + self.diag_loadings * eye  # diagonal loading

        phi_vv_inv = complex_inverse(phi_vv, RI_dim=1)

        phi_vv_inv = phi_vv_inv.unsqueeze(-1)
        phi_vv_inv = phi_vv_inv.transpose(1, -1)  # N, 1, F, 1, C, C, 2
        phi_vv_inv = phi_vv_inv.contiguous()
        phi_vv_inv = torch.view_as_complex(phi_vv_inv)  # N, 1, F, 1, C, C,

        phi_xx = torch_spatial_covariance(X)  # shape of (N,2,F,T,C,C)
        # phi_xx = torch.mean(phi_xx, dim=3, keepdims=True)

        # assert torch.allclose(conj(phi_xx.transpose(-1, -2)), phi_xx)  # check hermitian

        phi_xx = phi_xx.unsqueeze(-1)
        phi_xx = phi_xx.transpose(1, -1)  # N, 1, F, T, C, C, 2
        phi_xx = phi_xx.contiguous()
        phi_xx = torch.view_as_complex(phi_xx)  # N, T, F, 1, C, C,

        w, v = torch.linalg.eig(phi_xx)  # eigenvalues, eigenvectors

        # v:  N, 1, F, 1, C, C
        r = v[:, :, :, :, :, 0:1]  # N, 1, F, 1, C, 1

        d = torch.zeros_like(phi_xx)  # N, 1, F, 1, C, C

        for c in range(C):
            d[:, :, :, :, :, c : c + 1] = r / r[:, :, :, :, c : c + 1]

        W_num = phi_vv_inv @ d
        W_den = torch.conj(d.transpose(-1, -2)) @ W_num

        # W_bf = multichannel_mpdr_rank_1_with_inverse(phi_xx, phi_vv_inv) # N, 2, F, 1, C, C
        W_bf = W_num / W_den
        W_bf = torch.view_as_real(W_bf)  # N, 1, F, 1, C, C, 2

        W_bf = W_bf.transpose(1, -1)  # N, 2, F, 1, C, C, 1
        W_bf = W_bf.squeeze(-1)  # N, 2, F, 1, C, C

        X_bf = apply_beamformer(W_bf, Y)  # N, 2, C, T, F

        X_bf = X_bf.transpose(1, 2)  # N, C, 2, T, F

        return X_bf


class Oracle_BF_ConvTasNet(ConvTasNet):
    def __init__(self, oracle_beamformer, **kwargs):

        super().__init__(**kwargs)

        self.oracle_beamformer = oracle_beamformer

    def forward(self, input):

        enh = self.oracle_beamformer(input)["est_target"]
        enh = enh.unsqueeze(1)  # N, n_src, L
        est_target = super().forward(enh)  # (batch, nsrc, nfilters, nframes)
        est_target = est_target[:, 0]

        return {"est_target": est_target}


class Oracle_BF_CSeqUNetDense(CSeqUNetDense_Singlechannel):
    def __init__(self, oracle_beamformer_cls, ref_mic_idx, **kwargs):

        super().__init__(ref_mic_idx, **kwargs)
        self.oracle_beamformer = oracle_beamformer_cls(self.STFT, ref_mic_idx)

    def forward(self, input):

        y = input["mixture"]
        x = input["target"]

        N, C, L = y.shape
        Y = self.STFT(y)
        X = self.STFT(x)

        X_bf = self.oracle_beamformer.forward_in_STFT(Y, X, Y - X)  # N, C, 2, T, F

        est_target = super().forward_in_STFT(X_bf)  # N, C, 2, T, F

        est_target = self.STFT.backward(est_target)

        est_target = est_target[:, 0]

        return {"est_target": est_target}


class Oracle_TI_MPDR_rank_1_CSeqUNetDense(Oracle_BF_CSeqUNetDense):
    def __init__(self, ref_mic_idx, **kwargs):

        self.ref_mic_idx = ref_mic_idx
        oracle_beamformer_cls = Oracle_TI_MPDR_rank_1
        super().__init__(oracle_beamformer_cls, ref_mic_idx, **kwargs)

    # def forward(self, input):

    #     enh = self.oracle_beamformer(input)["est_target"]
    #     # enh = enh.unsqueeze(1)  #
    #     est_target = super().forward(enh)
    #     est_target = est_target[:, 0]

    #     return {"est_target": est_target}


class Oracle_TV_MPDR_rank_1_vary_x_CSeqUNetDense(Oracle_BF_CSeqUNetDense):
    def __init__(self, ref_mic_idx, **kwargs):

        self.ref_mic_idx = ref_mic_idx
        oracle_beamformer_cls = Oracle_TV_MPDR_rank_1_vary_x
        super().__init__(oracle_beamformer_cls, ref_mic_idx, **kwargs)


class Oracle_TV_MPDR_rank_1_accumulate_x_CSeqUNetDense(Oracle_BF_CSeqUNetDense):
    def __init__(self, ref_mic_idx, **kwargs):

        self.ref_mic_idx = ref_mic_idx
        oracle_beamformer_cls = Oracle_TV_MPDR_rank_1_accumulate_x
        super().__init__(oracle_beamformer_cls, ref_mic_idx, **kwargs)


class Oracle_BF_CSeqUNetDense(CSeqUNetDense_Singlechannel):
    def __init__(self, oracle_beamformer_cls, ref_mic_idx, **kwargs):

        super().__init__(ref_mic_idx, **kwargs)
        self.oracle_beamformer = oracle_beamformer_cls(self.STFT, ref_mic_idx)

    def forward(self, input):

        y = input["mixture"]
        x = input["target"]

        N, C, L = y.shape
        Y = self.STFT(y)
        X = self.STFT(x)

        X_bf = self.oracle_beamformer.forward_in_STFT(Y, X, Y - X)  # N, C, 2, T, F

        est_target = super().forward_in_STFT(X_bf)  # N, C, 2, T, F

        est_target = self.STFT.backward(est_target)

        est_target = est_target[:, 0]

        return {"est_target": est_target}


class Oracle_TI_MPDR_rank_1_CSeqUNetDense(Oracle_BF_CSeqUNetDense):
    def __init__(self, ref_mic_idx, **kwargs):

        self.ref_mic_idx = ref_mic_idx
        oracle_beamformer_cls = Oracle_TI_MPDR_rank_1
        super().__init__(oracle_beamformer_cls, ref_mic_idx, **kwargs)

    # def forward(self, input):

    #     enh = self.oracle_beamformer(input)["est_target"]
    #     # enh = enh.unsqueeze(1)  #
    #     est_target = super().forward(enh)
    #     est_target = est_target[:, 0]

    #     return {"est_target": est_target}


class Oracle_TV_MPDR_rank_1_vary_x_CSeqUNetDense(Oracle_BF_CSeqUNetDense):
    def __init__(self, ref_mic_idx, **kwargs):

        self.ref_mic_idx = ref_mic_idx
        oracle_beamformer_cls = Oracle_TV_MPDR_rank_1_vary_x
        super().__init__(oracle_beamformer_cls, ref_mic_idx, **kwargs)


class Oracle_TV_MPDR_rank_1_accumulate_x_CSeqUNetDense(Oracle_BF_CSeqUNetDense):
    def __init__(self, ref_mic_idx, **kwargs):

        self.ref_mic_idx = ref_mic_idx
        oracle_beamformer_cls = Oracle_TV_MPDR_rank_1_accumulate_x
        super().__init__(oracle_beamformer_cls, ref_mic_idx, **kwargs)


class Oracle_BF_CSeqUNetDense_Diff_Beamformer(CSeqUNetDense_Singlechannel):
    def __init__(self, oracle_beamformer, ref_mic_idx, **kwargs):

        super().__init__(ref_mic_idx, **kwargs)
        self.oracle_beamformer = oracle_beamformer

    def forward(self, input):

        # y = input["mixture"]
        # x = input["target"]

        # N, C, L = y.shape
        # Y = self.STFT(y)
        # X = self.STFT(x)

        X_bf = self.oracle_beamformer.forward(
            input
        )  # .forward_in_STFT(Y, X, Y - X)  # N, C, 2, T, F

        X_bf["mixture"] = X_bf["est_target_multi_channel"]
        est_target = super().forward(X_bf)  # N, C, 2, T, F
        # est_target = self.STFT.backward(est_target)

        est_target = est_target["est_target"]

        return {"est_target": est_target}


class Oracle_TI_MPDR_rank_1_CSeqUNetDense_Diff_BF(
    Oracle_BF_CSeqUNetDense_Diff_Beamformer
):
    def __init__(self, STFT, ref_mic_idx, **kwargs):

        self.ref_mic_idx = ref_mic_idx
        oracle_beamformer = Oracle_TI_MPDR_rank_1(STFT, ref_mic_idx)
        super().__init__(oracle_beamformer, ref_mic_idx, **kwargs)


class Oracle_TV_MPDR_rank_1_vary_x_CSeqUNetDense_Diff_BF(
    Oracle_BF_CSeqUNetDense_Diff_Beamformer
):
    def __init__(self, STFT, ref_mic_idx, **kwargs):

        self.ref_mic_idx = ref_mic_idx
        oracle_beamformer_cls = Oracle_TV_MPDR_rank_1_vary_x(STFT, ref_mic_idx)
        super().__init__(oracle_beamformer_cls, ref_mic_idx, **kwargs)


class Oracle_TV_MPDR_rank_1_accumulate_x_CSeqUNetDense_Diff_BF(
    Oracle_BF_CSeqUNetDense_Diff_Beamformer
):
    def __init__(self, STFT, ref_mic_idx, **kwargs):

        self.ref_mic_idx = ref_mic_idx
        oracle_beamformer_cls = Oracle_TV_MPDR_rank_1_accumulate_x(STFT, ref_mic_idx)
        super().__init__(oracle_beamformer_cls, ref_mic_idx, **kwargs)


# class Oracle_TI_MPDR_rank_1_CSeqUNetDense(Oracle_BF_CSeqUNetDense):
#     def __init__(self, ref_mic_idx, **kwargs):

#         self.ref_mic_idx = ref_mic_idx
#         oracle_beamformer_cls = Oracle_TI_MPDR_rank_1
#         super().__init__(oracle_beamformer_cls, ref_mic_idx, **kwargs)

#     # def forward(self, input):

#     #     enh = self.oracle_beamformer(input)["est_target"]
#     #     # enh = enh.unsqueeze(1)  #
#     #     est_target = super().forward(enh)
#     #     est_target = est_target[:, 0]

#     #     return {"est_target": est_target}


# class Oracle_TV_MPDR_rank_1_vary_x_CSeqUNetDense(Oracle_BF_CSeqUNetDense):
#     def __init__(self, ref_mic_idx, **kwargs):

#         self.ref_mic_idx = ref_mic_idx
#         oracle_beamformer_cls = Oracle_TV_MPDR_rank_1_vary_x
#         super().__init__(oracle_beamformer_cls, ref_mic_idx, **kwargs)


# class Oracle_TV_MPDR_rank_1_accumulate_x_CSeqUNetDense(Oracle_BF_CSeqUNetDense):
#     def __init__(self, ref_mic_idx, **kwargs):

#         self.ref_mic_idx = ref_mic_idx
#         oracle_beamformer_cls = Oracle_TV_MPDR_rank_1_accumulate_x
#         super().__init__(oracle_beamformer_cls, ref_mic_idx, **kwargs)

# def forward(self, input):

#     enh = self.oracle_beamformer(input)["est_target"]
#     # enh = enh.unsqueeze(1)  #
#     est_target = super().forward(enh)
#     est_target = est_target[:, 0]

#     return {"est_target": est_target}


class Oracle_TI_MPDR_rank_1_ConvTasNet(Oracle_BF_ConvTasNet):
    def __init__(self, STFT, ref_mic_idx, **kwargs):

        oracle_beamformer = Oracle_TI_MPDR_rank_1(STFT, ref_mic_idx)
        self.ref_mic_idx = ref_mic_idx
        super().__init__(oracle_beamformer, **kwargs)


class Oracle_TV_MPDR_rank_1_vary_x_ConvTasNet(Oracle_BF_ConvTasNet):
    def __init__(self, STFT, ref_mic_idx, **kwargs):

        oracle_beamformer = Oracle_TV_MPDR_rank_1_vary_x(STFT, ref_mic_idx)
        self.ref_mic_idx = ref_mic_idx
        super().__init__(oracle_beamformer, **kwargs)


class Oracle_TV_MPDR_rank_1_accumulate_x_ConvTasnet(Oracle_BF_ConvTasNet):
    def __init__(self, STFT, ref_mic_idx, **kwargs):

        oracle_beamformer = Oracle_TV_MPDR_rank_1_accumulate_x(STFT, ref_mic_idx)
        self.ref_mic_idx = ref_mic_idx
        super().__init__(oracle_beamformer, **kwargs)


class Oracle_BF_SMoLnet(SMoLNet_freq_only_with_instance_norm_single_channel):
    def __init__(self, STFT, oracle_beamformer, **kwargs):

        super().__init__(STFT, **kwargs)

        self.oracle_beamformer = oracle_beamformer
        self.STFT = STFT

    def forward(self, input):

        y = input["mixture"]
        x = input["target"]

        N, C, L = y.shape
        Y = self.STFT(y)
        X = self.STFT(x)

        X_bf = self.oracle_beamformer.forward_in_STFT(Y, X, Y - X)  # N, C, 2, T, F

        est_target = super().forward_in_STFT(X_bf)  # N, C, 2, T, F

        est_target = self.STFT.backward(est_target)

        est_target = est_target[:, 0]

        return {"est_target": est_target}


class Oracle_TI_MPDR_rank_1_SMoLnet(Oracle_BF_SMoLnet):
    def __init__(self, STFT, ref_mic_idx, **kwargs):

        oracle_beamformer = Oracle_TI_MPDR_rank_1(STFT, ref_mic_idx)
        self.ref_mic_idx = ref_mic_idx
        super().__init__(STFT, oracle_beamformer, **kwargs)


class Oracle_TV_MPDR_rank_1_vary_x_SMoLnet(Oracle_BF_SMoLnet):
    def __init__(self, STFT, ref_mic_idx, **kwargs):

        oracle_beamformer = Oracle_TV_MPDR_rank_1_vary_x(STFT, ref_mic_idx)
        self.ref_mic_idx = ref_mic_idx
        super().__init__(STFT, oracle_beamformer, **kwargs)


class Oracle_TV_MPDR_rank_1_accumulate_x_SMoLnet(Oracle_BF_SMoLnet):
    def __init__(self, STFT, ref_mic_idx, **kwargs):

        oracle_beamformer = Oracle_TV_MPDR_rank_1_accumulate_x(STFT, ref_mic_idx)
        self.ref_mic_idx = ref_mic_idx
        super().__init__(STFT, oracle_beamformer, **kwargs)


class Oracle_TV_MPDR_rank_1_accumulate_x_delayed_SMoLnet(Oracle_BF_SMoLnet):
    def __init__(self, STFT, ref_mic_idx, **kwargs):

        oracle_beamformer = Oracle_TV_MPDR_rank_1_accumulate_x_delayed(
            STFT, ref_mic_idx
        )
        self.ref_mic_idx = ref_mic_idx
        super().__init__(STFT, oracle_beamformer, **kwargs)
