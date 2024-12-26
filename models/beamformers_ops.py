import numpy as np
import torch
import torch.nn as nn
from torch_complex_ops import *
from utils_beamforming import steering_vector, UMA8
from typing import *
import torch.jit


def apply_beamformer(W, Y):
    WH = conj(W.transpose(-1, -2), RI_dim=1)  # N, 1, F, 1, C, C

    # for t in range(T)

    Y_bf_in = Y.transpose(2, 4)  # N, 2, F, T, C
    Y_bf_in = Y_bf_in.unsqueeze(-1)  # N, 2, F, T, C, 1
    X_bf = complex_mm(WH, Y_bf_in, RI_dim=1)  # N, 2, F, 1, C, 1
    X_bf = X_bf.squeeze(-1)  # N, 2, F, T, C
    X_bf = X_bf.transpose(2, 4)  # N, 2, C, T, F

    return X_bf


def spatial_covariance(STFT):
    """Computes the spatial covariance matrix.

    .. math ::
       \Phi_{\mathbf{V}}(k)=\frac{1}{T}\sum_{m=0}^{T-1}\mathbf{V}(k,m)\mathbf{V}^H(k,m), \isin\R^{\mathsf{C\times{}C}}

    .. math ::
       \mathbf{V(k,m)} = \mathbf{\hat{V}}=[\hat{V}_1(k,m) \cdots \hat{V}_\mathsf{C}(k,m)]^T

    where :math:`\mathbf{V(k,m)}` is the STFT of the signal, :math:`k` is the frequency frame, :math:`t` is the time frame.

    Parameters
    ----------
    STFT : complex :obj:`numpy.ndarray` of shape of C, T, F
       The short-time fourier transformed signal, :math:`\mathbf{V(k,m)}`

    Returns
    -------
    complex :obj:`numpy.ndarray` of shape F, C, C
       :math:`\Phi(k)` of the STFT signal
    """
    C, T, F = STFT.shape
    # STFT = STFT.transpose((2, 1,0)) # [F, T, C]

    spatial_cov = np.zeros((F, C, C), dtype="complex128")
    # print(STFT[0].shape)
    for f in range(F):
        for t in range(T):
            # spatial_cov[f] += np.cov(STFT[f, t])
            # print(STFT[f, t] @ np.conj(STFT[f, t]).T)
            # print(STFT[:, f:f+1, t].T.shape)
            cov = STFT[:, f : f + 1, t] @ np.conj(STFT[:, f : f + 1, t]).T
            # print(cov.shape)
            spatial_cov[f] += cov

            # spatial_cov[f] += np.corrcoef(STFT[f, t])
        spatial_cov[f] = spatial_cov[f] / T

    # Matrix operation is done on the last
    # spatial_cov =
    # STFT = np.expand_dims(STFT, -1) # [F, T, C, 1]
    # spatial_cov = STFT @ np.conj(STFT).transpose((0,1,3,2)) # [F, T, C, C]
    # spatial_cov = np.average(spatial_cov, axis=1) # [F, C, C]

    # assert spatial_cov.shape == (F, C, C)
    return spatial_cov  # [F, C, C]


@torch.jit.export
def torch_spatial_covariance(STFT: torch.Tensor):
    """Computes the spatial covariance matrix.

    .. math ::
       \Phi_{\mathbf{V}}(k,m)=\mathbf{V}(k,m)\mathbf{V}^H(k,m), \in\R^{\mathsf{C\times{}C}}

    .. math ::
       \mathbf{V(k,m)} = \mathbf{\hat{V}}=[\hat{V}_1(k,m) \cdots \hat{V}_\mathsf{C}(k,m)]^T

    where :math:`\mathbf{V(k,m)}` is the STFT of the signal, :math:`k` is the frequency frame, :math:`t` is the time frame.

    Parameters
    ----------
    STFT :obj:`torch.Tensor` of shape of N, 2, C, T, F
       The short-time fourier transformed signal, :math:`\mathbf{V(k,m)}` in real-imaginary stacked form

    Returns
    -------
    complex :obj:`numpy.ndarray` of shape N, 2, F, T, C, C
       :math:`\Phi(k)` of the STFT signal
    """
    N, _, C, T, F = STFT.shape
    # STFT = STFT.transpose((2, 1,0)) # [F, T, C]
    RI_dim = 1
    _stft = STFT.permute((0, 1, 4, 3, 2))  # shape of N, 2, F, T, C
    # spatial_cov = torch.zeros((N,2,F,T,C,C)).to(STFT.get_device())

    # spatial_cov = spatial_cov / T
    # _stft = STFT
    _stft = _stft.unsqueeze(-1)  # shape of N, 2, F, T, C, 1
    conj_stft = conj(_stft.transpose(-1, -2), RI_dim)  # shape of N, 2, F, T, 1, C

    spatial_cov = complex_mm(_stft, conj_stft, RI_dim)  # shape of N, 2, F, T, C, C

    return spatial_cov  # shape of N, 2, F, T, C, C


# torch_spatial_covariance = torch.jit.trace(torch_spatial_covariance, torch.rand(8, 2, 8,  8, 8), optimize=True)


def torch_spatial_cross_covarance(A, B):
    # cov(A, B) = E[(A-mu_a)(B-mu_b)^H]
    # assert A.shape == B.shape
    RI_dim = 1
    assert A.shape[RI_dim] == 2 and B.shape[RI_dim] == 2

    N, _, C, T, F = A.shape
    N, _, C, T, F = B.shape

    A = A.transpose(-1, -3)  # N, 2, F, T, C
    B = B.transpose(-1, -3)  # N, 2, F, T, C

    A = A.unsqueeze(-1)  # N, 2, F, T, C, 1
    B = B.unsqueeze(-1)  # N, 2, F, T, C, 1

    mu_a = torch.mean(A, dim=[0, 3], keepdim=True)  # 1, 2, F, 1, C, 1
    mu_b = torch.mean(B, dim=[0, 3], keepdim=True)  # 1, 2, F, 1, C, 1

    cross_cov = complex_mm(A - mu_a, conj(B - mu_b, RI_dim).transpose(-1, -2), RI_dim)
    # assert cross_cov.shape == (N, 2, F, T, C, C)
    return cross_cov


def mvdr_rank_1(
    phi_xx: torch.Tensor,
    phi_vv: torch.Tensor,
    ref_mic_idx: int = 0,
    diag_loadings: float = 0.001 / np.sqrt(2),
):
    """Computes the weight of the MVDR beamforming by assuming that phi_xx is of rank-1

    Parameters
    ----------
    phi_xx : torch.Tensor (N, 2, F, T, C, C)
       The spatial covariance matrix of the target signal
    phi_vv : torch.Tensor (N, 2, T, T, C, C)
       The spatial covariance matrix of the noise signal
    diag_loadings : float, optional
       The constant to add to the diagonals of phi_vv before inverse, by default 0.001/np.sqrt(2)
    """

    N, _, F, _, C, C = phi_xx.shape
    device = phi_xx.device
    # if not hasattr(self, "eye"): #.eye is None:

    eye = torch.eye(C)
    # if torch.cuda.is_available():
    eye = eye.to(device)
    phi_vv = phi_vv + diag_loadings * eye  # diagonal loading
    inv_phi_vv = complex_inverse(phi_vv, RI_dim=1)

    # if self.e_ref is None:
    # if not hasattr(self, "e_ref"):

    e_ref = torch.zeros(C)
    e_ref[ref_mic_idx] = 1.0
    e_ref = e_ref.to(device)

    G = complex_mm(inv_phi_vv, phi_xx, RI_dim=1)  # N, 2, F, 1, C, C
    l = torch.sum(torch.diagonal(G, 0, -2, -1), dim=-1, keepdim=True)
    # l[:,0] 		-= l[:,0] - C 																	# N, 2, F, 1, 1
    inv_lamda = complex_reciprocal(l, RI_dim=1)  # N, 1, F, 1, 1
    w = complex_mm(inv_lamda, G @ e_ref, RI_dim=1)  # N, 2, F, 1, C
    w = w.unsqueeze(-1)  # N, 2, F, 1, C, 1
    # w 				= conj(w.transpose(-1,-2), RI_dim=1)				# N, 2, F, 1, C
    # w           = w.unsqueeze(-1)
    # w 				= w.transpose(-1,-2)																# N, 2, F, 1, C
    # w				= w.unsqueeze(-3)																	# N, 2, F, 1, 1, C

    # w = complex_mm(w_nume, complex_inverse(w_deno,RI_dim=1), RI_dim=1)
    # wH = conj(w.transpose(-1,-2), RI_dim=1)	# shape of N, 2, F, 1, 1, C
    # temp = output.clone()
    return w


def mpdr_rank_1(
    phi_xx: torch.Tensor,
    phi_yy: torch.Tensor,
    ref_mic_idx: int = 0,
    diag_loadings: float = 0.001 / np.sqrt(2),
):
    """Computes the weight of the MVDR beamforming by assuming that phi_xx is of rank-1

    Parameters
    ----------
    phi_xx : torch.Tensor (N, 2, F, T, C, C)
       The spatial covariance matrix of the target signal
    phi_yy : torch.Tensor (N, 2, T, T, C, C)
       The spatial covariance matrix of the noisy signal
    diag_loadings : float, optional
       The constant to add to the diagonals of phi_yy before inverse, by default 0.001/np.sqrt(2)
    """

    N, _, F, _, C, C = phi_xx.shape
    device = phi_xx.device
    # if not hasattr(self, "eye"): #.eye is None:

    eye = torch.eye(C)
    # if torch.cuda.is_available():
    eye = eye.to(device)
    phi_yy = phi_yy + diag_loadings * eye  # diagonal loading
    inv_phi_yy = complex_inverse(phi_yy, RI_dim=1)

    e_ref = torch.zeros(C)
    e_ref[ref_mic_idx] = 1.0
    e_ref = e_ref.to(device)

    G = complex_mm(inv_phi_yy, phi_xx, RI_dim=1)  # N, 2, F, 1, C, C
    l = torch.sum(torch.diagonal(G, 0, -2, -1), dim=-1, keepdim=True)  # N, 2, F, 1, 1
    # l           = torch.unsqueeze(l, -1)                                             # N, 2, F, 1, 1, 1

    inv_lamda = complex_reciprocal(l, RI_dim=1)  # N, 2, F, 1, 1, 1
    W = complex_mul(inv_lamda, G @ e_ref, RI_dim=1)  # N, 2, F, 1, C, 1
    W = W.unsqueeze(-1)

    # w 				= conj(w.transpose(-1,-2), RI_dim=1)																   # N, 2, F, 1, C
    # w				= w.unsqueeze(-3)																	# N, 2, F, 1, 1, C

    # w = complex_mm(w_nume, complex_inverse(w_deno,RI_dim=1), RI_dim=1)
    # wH = conj(w.transpose(-1,-2), RI_dim=1)	# shape of N, 2, F, 1, 1, C
    # temp = output.clone()
    return W


def mpdr_rank_1_with_inverse(
    phi_xx: torch.Tensor, inv_phi_yy: torch.Tensor, ref_mic_idx: int = 0
):
    """Computes the weight of the MVDR beamforming by assuming that phi_xx is of rank-1

    Parameters
    ----------
    phi_xx : torch.Tensor (N, 2, F, T, C, C)
       The spatial covariance matrix of the target signal
    inv_phi_yy : torch.Tensor (N, 2, T, T, C, C)
       The inverse spatial covariance matrix of the noisy signal
    diag_loadings : float, optional
       The constant to add to the diagonals of phi_yy before inverse, by default 0.001/np.sqrt(2)
    """

    N, _, F, _, C, C = phi_xx.shape
    device = phi_xx.device
    # if not hasattr(self, "eye"): #.eye is None:

    eye = torch.eye(C)
    # if torch.cuda.is_available():
    eye = eye.to(device)

    e_ref = torch.zeros(C)
    e_ref[ref_mic_idx] = 1.0
    e_ref = e_ref.to(device)

    G = complex_mm(inv_phi_yy, phi_xx, RI_dim=1)  # N, 2, F, 1, C, C
    l = torch.sum(torch.diagonal(G, 0, -2, -1), dim=-1, keepdim=True)  # N, 2, F, 1, 1
    # l           = torch.unsqueeze(l, -1)                                             # N, 2, F, 1, 1, 1

    inv_lamda = complex_reciprocal(l, RI_dim=1)  # N, 2, F, 1, 1, 1
    W = complex_mul(inv_lamda, G @ e_ref, RI_dim=1)  # N, 2, F, 1, C, 1
    W = W.unsqueeze(-1)

    # w 				= conj(w.transpose(-1,-2), RI_dim=1)																   # N, 2, F, 1, C
    # w				= w.unsqueeze(-3)																	# N, 2, F, 1, 1, C

    # w = complex_mm(w_nume, complex_inverse(w_deno,RI_dim=1), RI_dim=1)
    # wH = conj(w.transpose(-1,-2), RI_dim=1)	# shape of N, 2, F, 1, 1, C
    # temp = output.clone()
    return W


def multichannel_mvdr_rank_1(
    phi_xx: torch.Tensor,
    phi_vv: torch.Tensor,
    diag_loadings: float = 0.001 / np.sqrt(2),
):
    """Computes the weight of the MVDR beamforming by assuming that phi_xx is of rank-1

    Parameters
    ----------
    phi_xx : torch.Tensor (N, 2, F, T, C, C)
       The spatial covariance matrix of the target signal
    phi_vv : torch.Tensor (N, 2, T, T, C, C)
       The spatial covariance matrix of the noise signal
    diag_loadings : float, optional
       The constant to add to the diagonals of phi_vv before inverse, by default 0.001/np.sqrt(2)
    """

    N, _, F, _, C, C = phi_xx.shape
    device = phi_xx.device
    # if not hasattr(self, "eye"): #.eye is None:

    eye = torch.eye(C)
    # if torch.cuda.is_available():
    eye = eye.to(device)
    phi_vv = phi_vv + diag_loadings * eye  # diagonal loading
    inv_phi_vv = complex_inverse(phi_vv, RI_dim=1)

    # if self.e_ref is None:
    # if not hasattr(self, "e_ref"):

    # e_ref = torch.zeros(C)
    # e_ref[ref_mic_idx] = 1.0
    # e_ref = e_ref.to(device)

    G = complex_mm(inv_phi_vv, phi_xx, RI_dim=1)  # N, 2, F, 1, C, C
    l = torch.sum(torch.diagonal(G, 0, -2, -1), dim=-1, keepdim=True)  # N, 2, F, 1, 1
    l = torch.unsqueeze(l, -1)  # N, 2, F, 1, 1, 1
    # print(l.shape)
    # print(G.shape)
    inv_lamda = complex_reciprocal(l, RI_dim=1)  # N, 2, F, 1, 1, 1
    W = complex_mul(inv_lamda, G, RI_dim=1)  # N, 2, F, 1, C, C

    # w 				= conj(w.transpose(-1,-2), RI_dim=1)																   # N, 2, F, 1, C
    # w				= w.unsqueeze(-3)																	# N, 2, F, 1, 1, C

    # w = complex_mm(w_nume, complex_inverse(w_deno,RI_dim=1), RI_dim=1)
    # wH = conj(w.transpose(-1,-2), RI_dim=1)	# shape of N, 2, F, 1, 1, C
    # temp = output.clone()
    return W


def multichannel_mpdr_rank_1(
    phi_xx: torch.Tensor,
    phi_yy: torch.Tensor,
    diag_loadings: float = 0.001 / np.sqrt(2),
):
    """Computes the weight of the MVDR beamforming by assuming that phi_xx is of rank-1

    Parameters
    ----------
    phi_xx : torch.Tensor (N, 2, F, T, C, C)
       The spatial covariance matrix of the target signal
    phi_yy : torch.Tensor (N, 2, T, T, C, C)
       The spatial covariance matrix of the noisy signal
    diag_loadings : float, optional
       The constant to add to the diagonals of phi_yy before inverse, by default 0.001/np.sqrt(2)
    """

    N, _, F, _, C, C = phi_xx.shape
    device = phi_xx.device
    # if not hasattr(self, "eye"): #.eye is None:

    eye = torch.eye(C)
    # if torch.cuda.is_available():
    eye = eye.to(device)
    phi_yy = phi_yy + diag_loadings * eye  # diagonal loading
    inv_phi_yy = complex_inverse(phi_yy, RI_dim=1)

    # if self.e_ref is None:
    # if not hasattr(self, "e_ref"):

    # e_ref = torch.zeros(C)
    # e_ref[ref_mic_idx] = 1.0
    # e_ref = e_ref.to(device)

    G = complex_mm(inv_phi_yy, phi_xx, RI_dim=1)  # N, 2, F, 1, C, C
    l = torch.sum(torch.diagonal(G, 0, -2, -1), dim=-1, keepdim=True)  # N, 2, F, 1, 1
    l = torch.unsqueeze(l, -1)  # N, 2, F, 1, 1, 1
    # print(l.shape)
    # print(G.shape)
    inv_lamda = complex_reciprocal(l, RI_dim=1)  # N, 2, F, 1, 1, 1
    W = complex_mul(inv_lamda, G, RI_dim=1)  # N, 2, F, 1, C, C

    # w 				= conj(w.transpose(-1,-2), RI_dim=1)																   # N, 2, F, 1, C
    # w				= w.unsqueeze(-3)																	# N, 2, F, 1, 1, C

    # w = complex_mm(w_nume, complex_inverse(w_deno,RI_dim=1), RI_dim=1)
    # wH = conj(w.transpose(-1,-2), RI_dim=1)	# shape of N, 2, F, 1, 1, C
    # temp = output.clone()
    return W


def multichannel_mpdr_rank_1_with_inverse(
    phi_xx: torch.Tensor,
    inv_phi_yy: torch.Tensor,
):
    """Computes the weight of the MVDR beamforming by assuming that phi_xx is of rank-1

    Parameters
    ----------
    phi_xx : torch.Tensor (N, 2, F, T, C, C)
       The spatial covariance matrix of the target signal
    inv_phi_yy : torch.Tensor (N, 2, T, T, C, C)
       The inverse of the spatial covariance matrix of the noisy signal
    """

    N, _, F, _, C, C = phi_xx.shape

    # if self.e_ref is None:
    # if not hasattr(self, "e_ref"):

    # e_ref = torch.zeros(C)
    # e_ref[ref_mic_idx] = 1.0
    # e_ref = e_ref.to(device)

    G = complex_mm(inv_phi_yy, phi_xx, RI_dim=1)  # N, 2, F, 1, C, C
    l = torch.sum(torch.diagonal(G, 0, -2, -1), dim=-1, keepdim=True)  # N, 2, F, 1, 1
    l = torch.unsqueeze(l, -1)  # N, 2, F, 1, 1, 1
    # print(l.shape)
    # print(G.shape)
    inv_lamda = complex_reciprocal(l, RI_dim=1)  # N, 2, F, 1, 1, 1
    W = complex_mul(inv_lamda, G, RI_dim=1)  # N, 2, F, 1, C, C

    # w 				= conj(w.transpose(-1,-2), RI_dim=1)																   # N, 2, F, 1, C
    # w				= w.unsqueeze(-3)																	# N, 2, F, 1, 1, C

    # w = complex_mm(w_nume, complex_inverse(w_deno,RI_dim=1), RI_dim=1)
    # wH = conj(w.transpose(-1,-2), RI_dim=1)	# shape of N, 2, F, 1, 1, C
    # temp = output.clone()
    return W


def multichannel_mpdr_with_steering_vector(
    d: torch.Tensor,
    phi_yy: torch.Tensor,
    diag_loadings: float = 0.001 / np.sqrt(2),
):
    """Computes the weight of the MVDR beamforming by assuming that phi_xx is of rank-1

    Parameters
    ----------
    d : torch.Tensor (1, 1, F, 1, C, 1)
       The steering vector to the source
    phi_yy : torch.Tensor (N, 2, T, T, C, C)
       The spatial covariance matrix of the noisy signal
    diag_loadings : float, optional
       The constant to add to the diagonals of phi_yy before inverse, by default 0.001/np.sqrt(2)
    """

    N, _, F, _, C, C = phi_yy.shape
    device = phi_yy.device
    # if not hasattr(self, "eye"): #.eye is None:

    eye = torch.eye(C)
    # if torch.cuda.is_available():
    eye = eye.to(device)
    phi_yy = phi_yy + diag_loadings * eye  # diagonal loading
    inv_phi_yy = complex_inverse(phi_yy, RI_dim=1)

    W_numerator = complex_mm(inv_phi_yy, d, RI_dim=1)  # N, 2, F, 1, C, 1
    dH = conj(d.transpose(-1, -2), RI_dim=1)
    W_denominator = complex_mm(complex_mm(dH, inv_phi_yy, RI_dim=1), d, RI_dim=1)
    W = complex_mm(W_numerator, complex_inverse(W_denominator, RI_dim=1), RI_dim=1)

    return W


def multichannel_wf_rank_1(
    phi_xx: torch.Tensor,
    phi_yy: torch.Tensor,
    diag_loadings: float = 0.001 / np.sqrt(2),
):
    """Computes the weight of the MWF beamforming by assuming that phi_xx is of rank-1

    Parameters
    ----------
    phi_xx : torch.Tensor (N, 2, F, T, C, C)
       The spatial covariance matrix of the target signal
    phi_yy : torch.Tensor (N, 2, T, T, C, C)
       The spatial covariance matrix of the noisy signal
    diag_loadings : float, optional
       The constant to add to the diagonals of phi_yy before inverse, by default 0.001/np.sqrt(2)
    """

    N, _, F, _, C, C = phi_xx.shape
    device = phi_xx.device
    # if not hasattr(self, "eye"): #.eye is None:

    eye = torch.eye(C)
    # if torch.cuda.is_available():
    eye = eye.to(device)
    phi_yy = phi_yy + diag_loadings * eye  # diagonal loading
    inv_phi_yy = complex_inverse(phi_yy, RI_dim=1)

    # if self.e_ref is None:
    # if not hasattr(self, "e_ref"):

    # e_ref = torch.zeros(C)
    # e_ref[ref_mic_idx] = 1.0
    # e_ref = e_ref.to(device)

    W = complex_mm(inv_phi_yy, phi_xx, RI_dim=1)  # N, 2, F, 1, C, C
    # l 				= torch.sum(torch.diagonal(G, 0, -2, -1), dim=-1, keepdim=True)      # N, 2, F, 1, 1
    # l           = torch.unsqueeze(l, -1)                                             # N, 2, F, 1, 1, 1
    # print(l.shape)
    # print(G.shape)
    # inv_lamda 	= complex_reciprocal(l, RI_dim=1) 												# N, 2, F, 1, 1, 1
    # W 				= complex_mul(inv_lamda, G, RI_dim=1)                              	# N, 2, F, 1, C, C

    # w 				= conj(w.transpose(-1,-2), RI_dim=1)																   # N, 2, F, 1, C
    # w				= w.unsqueeze(-3)																	# N, 2, F, 1, 1, C

    # w = complex_mm(w_nume, complex_inverse(w_deno,RI_dim=1), RI_dim=1)
    # wH = conj(w.transpose(-1,-2), RI_dim=1)	# shape of N, 2, F, 1, 1, C
    # temp = output.clone()
    return W


# def plot_3d_beampattern


class MVDR_oracle(nn.Module):
    def __init__(self, hparams, d=None):
        """Perform oracle MVDR based on spatial covariance matrix. The mixture and noise is assumed to be uncorrelated.
        .. math::

           \Phi_{\mathbf{V}}(k,m)=\mathbf{V}(k,m)\mathbf{V}^H(k,m), R\in{\mathsf{C\times{}C}
           G(f) = \Phi_{v}^{-1}\Phi_{y}


        .. math::
           H(f) = \frac{1}{l(f)} (G(f) - I)e_ref
        """
        super(MVDR_oracle, self).__init__()
        self.ref_mic_idx = hparams.ref_mic_idx
        self.dummy_weight = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, input):

        y = input["mixture"]
        x = input["target"]
        v = input["noise"]

        N, _, C, T, F = y.shape

        device = y.get_device()
        eye = torch.eye(C).to(device)
        G = torch.zeros((N, 2, F, 1, C, C)).to(device)  # gain matrix
        e_ref = torch.zeros(C).to(y.get_device())
        e_ref[self.ref_mic_idx] = 1.0
        I = torch.zeros_like(G)
        I[:, 0] += eye  # only the real part

        # H = np.zeros((F, C), dtype="complex128") # final transfer function

        phi_noise = torch_spatial_covariance(v)  # shape of (N,2,F,T,C,C))
        phi_noise = (
            torch.sum(phi_noise, dim=3, keepdim=True) / T
        )  # shape of (N,2,F,1,C,C))

        phi_mixture = torch_spatial_covariance(y)  # shape of (N,2,F,T,C,C))
        phi_mixture = (
            torch.sum(phi_mixture, dim=3, keepdim=True) / T
        )  # shape of (N,2,F,1,C,C))

        loading_abs = 0.001
        load_rl_img = 0.001 / np.sqrt(2)
        phi_noise += load_rl_img * eye  # diagonal loading

        phi_noise_inv = complex_inverse(phi_noise, RI_dim=1)  # shape of (N,2,F,1,C,C)

        G = complex_mm(phi_noise_inv, phi_mixture, RI_dim=1)  # shape of (N,2,F,1,C,C)

        trace = lambda A: torch.sum(torch.diagonal(A, 0, -2, -1), dim=-1, keepdim=True)
        # l = torch.trace(G) - C
        l = trace(G) - C  # shape of (N, 2, F, 1, 1)

        inv_l = complex_inverse(l, RI_dim=1)  # shape of (N, 2, F, 1, 1)

        H = complex_mm(inv_l, (G - I) @ e_ref, RI_dim=1)  # shape of [N, 2, F, 1, C]

        y = torch.transpose(y, -1, 2)  # [N, 2, F, T, C]
        y.unsqueeze_(-1)  # [N, 2, F, T, C, 1]
        H.unsqueeze_(-1)  # [N, 2, F, 1, C, 1]
        H_H = conj(H, RI_dim=1).transpose(-1, -2)  # [N, 2, F, 1, 1, C]
        X_bf = complex_mm(H_H, y, RI_dim=1)  # shape of [N, 2, F, T, 1, 1]
        X_bf.squeeze_(-1)  # [N, 2, F, T, 1]
        X_bf = X_bf.transpose(-1, -3)  # shape of [N, 2, 1, T, F, 1]
        output = {}
        output["est_target"] = X_bf

        return output


class MVDR_Oracle_no_steering_vector(nn.Module):
    def __init__(self, hparams, d=None):
        """Perform oracle MVDR based on spatial covariance matrix. The mixture and noise is assumed to be uncorrelated.
        .. math::

           \Phi_{\mathbf{V}}(k,m)=\mathbf{V}(k,m)\mathbf{V}^H(k,m), R\in{\mathsf{C\times{}C}
           G(f) = \Phi_{v}^{-1}\Phi_{y}


        .. math::
           H(f) = \frac{1}{l(f)} (G(f) - I)e_ref
        """
        super(MVDR_Oracle_no_steering_vector, self).__init__()
        self.ref_mic_idx = hparams.ref_mic_idx
        self.dummy_weight = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, input):

        y = input["mixture"]
        x = input["target"]
        v = input["noise"]

        N, _, C, T, F = y.shape

        device = y.get_device()
        eye = torch.eye(C).to(device)
        G = torch.zeros((N, 2, F, 1, C, C)).to(device)  # gain matrix
        e_ref = torch.zeros(C).to(y.get_device())
        e_ref[self.ref_mic_idx] = 1.0
        I = torch.zeros_like(G)
        I += eye / np.sqrt(2)

        phi_noise = torch_spatial_covariance(v)  # shape of (N,2,F,T,C,C))
        phi_noise = torch.mean(
            phi_noise, dim=3, keepdim=True
        )  # shape of (N,2,F,1,C,C))

        phi_mixture = torch_spatial_covariance(y)  # shape of (N,2,F,T,C,C))
        phi_mixture = torch.mean(
            phi_mixture, dim=3, keepdim=True
        )  # shape of (N,2,F,1,C,C))

        loading_abs = 0.001
        load_rl_img = 0.001 / np.sqrt(2)
        phi_noise += load_rl_img * eye  # diagonal loading

        phi_noise_inv = complex_inverse(phi_noise, RI_dim=1)  # shape of (N,2,F,1,C,C)

        G = complex_mm(phi_noise_inv, phi_mixture, RI_dim=1)  # shape of (N,2,F,1,C,C)

        trace = lambda A: torch.sum(torch.diagonal(A, 0, -2, -1), dim=-1, keepdim=True)
        l = trace(G)
        l -= C / np.sqrt(2)  # shape of (N, 2, F, 1, 1)
        # only the real part
        inv_l = complex_reciprocal(l, RI_dim=1)  # shape of (N, 2, F, 1, 1)

        H = complex_mul(inv_l, (G - I) @ e_ref, RI_dim=1)  # [N, 2, F, 1, C]
        # H_H = conj(H,RI_dim=1) # [N, 2, F, 1, C]
        y = torch.transpose(y, -1, -3)  # [N, 2, F, T, C]

        # X_bf = torch.sum(torch.cat([complex_mul(H[:,:,:,:,c:c+1], y[:,:,:,:,c:c+1], RI_dim=1) for c in range(C)],dim=-1), dim=-1, keepdim=True)
        # X_bf = torch.transpose(X_bf, -1, -3) # [N, 2, F, T, C]
        #
        y.unsqueeze_(-1)  # [N, 2, F, T, C, 1]
        H.unsqueeze_(-1)  # [N, 2, F, 1, C, 1]
        H_H = conj(H, RI_dim=1).transpose(-1, -2)  # [N, 2, F, 1, 1, C]
        X_bf = complex_mm(H_H, y, RI_dim=1)  # shape of [N, 2, F, T, 1, 1]
        X_bf.squeeze_(-1)  # [N, 2, F, T, 1]
        X_bf = X_bf.transpose(-1, -3)  # shape of [N, 2, 1, T, F, 1]
        output = {}
        output["est_target"] = X_bf

        return output

    def loss(self, input, output, ref_mic_idx=None):
        return self.dummy_weight * 0.1


class MVDR_Oracle_no_steering_vector_using_target(nn.Module):
    def __init__(self, hparams, d=None):
        """Perform oracle MVDR based on spatial covariance matrix. The mixture and noise is assumed to be uncorrelated.
        .. math::

           \Phi_{\mathbf{V}}(k,m)=\mathbf{V}(k,m)\mathbf{V}^H(k,m), R\in{\mathsf{C\times{}C}
           G(f) = \Phi_{v}^{-1}\Phi_{y}


        .. math::
           H(f) = \frac{1}{l(f)} (G(f) - I)e_ref
        """
        super(MVDR_Oracle_no_steering_vector_using_target, self).__init__()
        self.ref_mic_idx = hparams.ref_mic_idx
        self.dummy_weight = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, input):

        y = input["mixture"]

        # if self.train:
        # x = input["mixture"]
        # else:
        if "target" in input.keys():
            x = input["target"]
        else:
            x = input["mixture"]
            # print("I am target :)")
        v = input["noise"]

        N, _, C, T, F = y.shape

        device = y.get_device()
        eye = torch.eye(C).to(device)
        G = torch.zeros((N, 2, F, 1, C, C)).to(device)  # gain matrix
        e_ref = torch.zeros(C).to(y.get_device())
        e_ref[self.ref_mic_idx] = 1.0

        phi_noise = torch_spatial_covariance(v)  # shape of (N,2,F,T,C,C))
        phi_noise = torch.mean(
            phi_noise, dim=3, keepdim=True
        )  # shape of (N,2,F,1,C,C))

        phi_target = torch_spatial_covariance(x)  # shape of (N,2,F,T,C,C))
        phi_target = torch.mean(
            phi_target, dim=3, keepdim=True
        )  # shape of (N,2,F,1,C,C))

        loading_abs = 0.001
        load_rl_img = 0.001 / np.sqrt(2)
        phi_noise += load_rl_img * eye  # diagonal loading

        phi_noise_inv = complex_inverse(phi_noise, RI_dim=1)  # shape of (N,2,F,1,C,C)

        G = complex_mm(phi_noise_inv, phi_target, RI_dim=1)  # shape of (N,2,F,1,C,C)

        trace = lambda A: torch.sum(torch.diagonal(A, 0, -2, -1), dim=-1, keepdim=True)
        l = trace(G)
        # l[:, 0] -= C# shape of (N, 2, F, 1, 1)
        # only the real part
        inv_l = complex_reciprocal(l, RI_dim=1)  # shape of (N, 2, F, 1, 1)

        W = complex_mul(inv_l, G @ e_ref, RI_dim=1)  # [N, 2, F, 1, C]
        # H_H = conj(H,RI_dim=1) # [N, 2, F, 1, C]
        y = torch.transpose(y, -1, -3)  # [N, 2, F, T, C]

        # X_bf = torch.sum(torch.cat([complex_mul(H[:,:,:,:,c:c+1], y[:,:,:,:,c:c+1], RI_dim=1) for c in range(C)],dim=-1), dim=-1, keepdim=True)
        # X_bf = torch.transpose(X_bf, -1, -3) # [N, 2, F, T, C]
        #
        y.unsqueeze_(-1)  # [N, 2, F, T, C, 1]
        W.unsqueeze_(-1)  # [N, 2, F, 1, C, 1]
        W_H = conj(W, RI_dim=1).transpose(-1, -2)  # [N, 2, F, 1, 1, C]
        X_bf = complex_mm(W_H, y, RI_dim=1)  # shape of [N, 2, F, T, 1, 1]
        X_bf.squeeze_(-1)  # [N, 2, F, T, 1]
        X_bf = X_bf.transpose(-1, -3)  # shape of [N, 2, 1, T, F, 1]
        output = {}
        output["est_target"] = X_bf

        return output

    def loss(self, input, output, ref_mic_idx=None):
        return self.dummy_weight * 0.1


class MVDR_oracle_using_steering_vector(nn.Module):
    def __init__(self, hparams, d=None, **kwargs):
        super(MVDR_oracle_using_steering_vector, self).__init__()
        self.ref_mic_idx = hparams.ref_mic_idx
        self.dummy_weight = nn.Parameter(torch.zeros(1, requires_grad=True))

        self.fs = hparams.fs
        self.win_len = hparams.win_len
        bin_size = self.fs / self.win_len
        self.freq = np.arange(0, self.fs // 2 + 1, bin_size)

    def forward(self, input):
        y = input["mixture"]
        # x = input["target"]
        v = input["noise"]
        d = input["steering_vector"]
        RI_dim = 1
        # print(y.shape)
        N, _, C, T, F = y.shape
        # d = d.unsqueeze(0) # shape of N, _, F, C, T
        device = y.get_device()
        eye = torch.eye(C).to(device)
        G = torch.zeros((N, 2, F, 1, C, C)).to(device)  # gain matrix
        e_ref = torch.zeros(C).to(y.get_device())
        e_ref[self.ref_mic_idx] = 1.0

        # H = np.zeros((F, C), dtype="complex128") # final transfer function

        phi_noise = torch_spatial_covariance(v)  # shape of (N,2,F,T,C,C))
        # phi_noise = torch.sum(phi_noise, dim=3, keepdim=True) / T # shape of (N,2,F,1,C,C))
        phi_noise = torch.mean(
            phi_noise, dim=3, keepdim=True
        )  # shape of (N,2,F,1,C,C))

        # phi_mixture = torch_spatial_covariance(y) # shape of (N,2,F,T,C,C))
        # phi_mixture = torch.sum(phi_mixture, dim=3, keepdim=True) / T # shape of (N,2,F,1,C,C))

        loading_abs = 0.001
        load_rl_img = 0.001 / np.sqrt(2)
        phi_noise += load_rl_img * eye  # diagonal loading

        phi_noise_inv = complex_inverse(phi_noise, RI_dim)  # shape of (N,2,F,1,C,C)
        # print(d.shape)
        phi_noise_inv.squeeze_(3)  # shape of (N, 2, F, C, C)
        w_nume = complex_mm(phi_noise_inv, d, RI_dim)  # shape of N, 2, F, C, 1
        dH = conj(d.transpose(-1, -2), RI_dim=1)
        w_deno = complex_mm(
            complex_mm(dH, phi_noise_inv, RI_dim), d, RI_dim
        )  # shape of N, 2, F, 1, 1
        w = complex_mm(
            w_nume, complex_inverse(w_deno, RI_dim=1), RI_dim=1
        )  # shape of N, 2, F, 1, C, 1
        # w = complex_mul(w_nume, complex_reciprocal(w_deno,RI_dim), RI_dim)
        y = torch.transpose(y, -1, 2)  # [N, 2, F, T, C]
        y.unsqueeze_(-1)  # [N, 2, F, T, C, 1]
        w.unsqueeze_(3)  # [N, 2, F, 1, C, 1]
        wH = conj(w, RI_dim=1).transpose(-1, -2)  # [N, 2, F, 1, 1, C]
        X_bf = complex_mm(wH, y, RI_dim=1)  # shape of [N, 2, F, T, 1, 1]
        X_bf.squeeze_(-1)  # [N, 2, F, T, 1]
        X_bf = X_bf.transpose(-1, -3)  # shape of [N, 2, 1, T, F, 1]

        # plot beampattern here first

        # beampattern_by_elevation(w.cpu())

        output = {}
        output["est_target"] = X_bf

        return output

    def loss(self, input, output, ref_mic_idx=None):
        return self.dummy_weight * 0.1


class Oracle_Multichannel_Wiener_Filtering(nn.Module):
    def __init__(self, hparams, d=None):
        super(Oracle_Multichannel_Wiener_Filtering, self).__init__()
        self.ref_mic_idx = hparams.ref_mic_idx
        self.dummy_weight = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, input):
        y = input["mixture"]
        x = input["target"]
        # s = input["speech_unconvolved"]
        # v = input["noise"]
        d = input["steering_vector"]
        RI_dim = 1
        # print(y.shape)
        N, _, C, T, F = y.shape
        # d = d.unsqueeze(0) # shape of N, _, F, C, T
        device = y.get_device()
        eye = torch.eye(C).to(device)
        G = torch.zeros((N, 2, F, 1, C, C)).to(device)  # gain matrix
        e_ref = torch.zeros(C).to(y.get_device())
        e_ref[self.ref_mic_idx] = 1.0
        I = torch.zeros_like(G)
        I[:, 0] += eye  # only the real part

        # H = np.zeros((F, C), dtype="complex128") # final transfer function

        phi_mixture = torch_spatial_covariance(y)  # shape of (N,2,F,T,C,C))
        phi_mixture = (
            torch.sum(phi_mixture, dim=3, keepdim=True) / T
        )  # shape of (N,2,F,1,C,C))
        loading_abs = 0.001
        load_rl_img = 0.001 / np.sqrt(2)
        phi_mixture += load_rl_img * eye  # diagonal loading
        phi_mixture_inv = complex_inverse(phi_mixture, RI_dim)  # shape of (N,2,F,1,C,C)
        phi_mixture_inv = phi_mixture_inv.squeeze(3)  # shape of (N, 2, F, C, C)

        # phi_ys = torch_spatial_cross_covarance(y, s) # shape of N, 2, F, T, C, 1
        # phi_ys = torch.mean(phi_ys, dim=3, keepdim=True) # shape of (N,2,F,1,C,1))
        # phi_ys = phi_ys.squeeze(3) # shape of (N,2,F,C,C))
        # w = complex_mm(phi_mixture_inv,phi_ys , RI_dim=1)  # shape of N, 2, F, C, 1

        phi_yx0 = torch_spatial_cross_covarance(
            y, x[:, :, self.ref_mic_idx : self.ref_mic_idx + 1]
        )  # shape of N, 2, F, T, C, 1
        phi_yx0 = torch.mean(phi_yx0, dim=3, keepdim=True)  # shape of (N,2,F,1,C,1))
        phi_yx0 = phi_yx0.squeeze(3)  # shape of (N,2,F,C,C))
        w = complex_mm(phi_mixture_inv, phi_yx0, RI_dim=1)  # shape of N, 2, F, C, 1
        #    _y = y.transpose(-1, -3) # N, 2, F, T, C
        #    _y = y.unsqueeze(-1)
        #    mu_y = torch.mean(_y, dim=[1, 3], keepdim=True)
        # # A = A.unsqueeze(-1) # N, 2, F, T, C, 1
        # B = B.unsqueeze(-1) # N, 2, F, T, C, 1

        # 1, 2, F, 1, C, 1

        y = torch.transpose(y, -1, 2)  # [N, 2, F, T, C]
        y = y.unsqueeze(-1)  # [N, 2, F, T, C, 1]
        w = w.unsqueeze(3)  # [N, 2, F, 1, C, 1]
        wH = conj(w, RI_dim=1).transpose(-1, -2)  # [N, 2, F, 1, 1, C]
        X_bf = complex_mm(wH, y, RI_dim=1)  # shape of [N, 2, F, T, 1, 1]
        X_bf.squeeze_(-1)  # wH.[N, 2, F, T, 1]
        X_bf = X_bf.transpose(-1, -3)  # shape of [N, 2, 1, T, F, 1]

        output = {}
        output["est_target"] = X_bf

        return output

    def loss(self, input, output, ref_mic_idx=None):
        return self.dummy_weight * 0.1


class MVDR_oracle_using_steering_vector2(nn.Module):
    def __init__(self, hparams, d=None):
        super(MVDR_oracle_using_steering_vector2, self).__init__()
        self.ref_mic_idx = hparams.ref_mic_idx
        self.dummy_weight = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, input):
        y = input["mixture"]
        v = input["noise"]
        x = input["target"]
        d = input["steering_vector"]
        N, _, C, T, F = y.shape
        d = d.unsqueeze(0)  # shape of N, _, F, C, T
        device = y.get_device()
        eye = torch.eye(C).to(device)
        G = torch.zeros((N, 2, F, 1, C, C)).to(device)  # gain matrix
        e_ref = torch.zeros(C).to(y.get_device())
        e_ref[self.ref_mic_idx] = 1.0

        # H = np.zeros((F, C), dtype="complex128") # final transfer function

        phi_signal = torch_spatial_covariance(x)  # shape of (N,2,F,T,C,C))
        phi_signal = (
            torch.sum(phi_signal, dim=3, keepdim=True) / T
        )  # shape of (N,2,F,1,C,C))

        phi_mixture = torch_spatial_covariance(y)  # shape of (N,2,F,T,C,C))
        phi_mixture = (
            torch.sum(phi_mixture, dim=3, keepdim=True) / T
        )  # shape of (N,2,F,1,C,C))

        # loading_abs = 0.001
        load_rl_img = 0.001 / np.sqrt(2)
        phi_signal += load_rl_img * eye  # diagonal loading
        # phi_noise_inv = complex_inverse(phi_noise, RI_dim=1) # shape of (N,2,F,1,C,C)

        phi_signal.squeeze_(3)  # shape of (N, 2, F, C, C)
        w_nume = complex_mm(phi_signal, d, RI_dim=1)  # shape of N, 2, F, C, 1
        dH = conj(d.transpose(-1, -2), RI_dim=1)
        w_deno = complex_mm(
            complex_mm(dH, phi_signal, RI_dim=1), d, RI_dim=1
        )  # shape of N, 2, F, 1, 1
        w = complex_mm(
            w_nume, complex_inverse(w_deno, RI_dim=1), RI_dim=1
        )  # shape of N, 2, F, C, 1
        y = torch.transpose(y, -1, 2)  # [N, 2, F, T, C]
        y.unsqueeze_(-1)  # [N, 2, F, T, C, 1]
        w.unsqueeze_(3)  # [N, 2, F, 1, C, 1]
        wH = conj(w, RI_dim=1).transpose(-1, -2)  # [N, 2, F, 1, 1, C]
        X_bf = complex_mm(wH, y, RI_dim=1)  # shape of [N, 2, F, T, 1, 1]
        X_bf.squeeze_(-1)  # [N, 2, F, T, 1]
        X_bf = X_bf.transpose(-1, -3)  # shape of [N, 2, 1, T, F, 1]

        output = {}
        output["est_target"] = X_bf

        return output
