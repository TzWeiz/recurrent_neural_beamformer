"""

Contains functions to perform complex operations in torch by employing real-imaginary composites.

"""

import torch
from typing import *


def conj(X, RI_dim: int):
    """Perform complex conj on real and imaginary stacked along specified dimension.

    Parameters
    ----------
    X : :obj:`torch.Tensor` of at least 3 dims
       Tensor with the real and imaginary stacked along the RI_dim.
    RI_dim : int
       The dimension of where the real and imag components are stacked.
    """
    X_rl, X_img = torch.split(X, 1, dim=RI_dim)
    X_rl.requires_grad_()
    X_img.requires_grad_()
    X_img = -1 * X_img
    return torch.cat([X_rl, X_img], dim=RI_dim)


def complex_abs(X, RI_dim: int, eps: float = torch.finfo(torch.float32).eps):
    """Perform complex absolute on real and imaginary stacked along specified dimension.


    Parameters
    ----------
    X : :obj:`torch.Tensor` of at least 3 dims
       Tensor with the real and imaginary stacked along the RI_dim.
    RI_dim : int
       The dimension of where the real and imag components are stacked.
    """
    X_rl, X_img = torch.split(X, 1, dim=RI_dim)
    X_abs = torch.sqrt(X_rl ** 2 + X_img ** 2 + eps)
    # X_abs = X_rl **2 + X_img **2
    return X_abs


def complex_mag(X, RI_dim: int):
    """Perform complex magnitude on real and imaginary stacked along specified dimension.


    Parameters
    ----------
    X : :obj:`torch.Tensor` of at least 3 dims
       Tensor with the real and imaginary stacked along the RI_dim.
    RI_dim : int
       The dimension of where the real and imag components are stacked.
    """
    X_rl, X_img = torch.split(X, 1, dim=RI_dim)
    X_mag = torch.abs(X_rl ** 2 + X_img ** 2)
    # X_abs = X_rl **2 + X_img **2
    return X_mag


def complex_mul(A: torch.Tensor, B: torch.Tensor, RI_dim: int):
    """Perform element-wise complex multiplication on real and imaginary stacked along specificed dimension."""

    A_rl, A_img = torch.split(A, 1, dim=RI_dim)
    B_rl, B_img = torch.split(B, 1, dim=RI_dim)

    rl_result = A_rl * B_rl - A_img * B_img
    img_result = A_rl * B_img + A_img * B_rl

    torch_result = torch.cat([rl_result, img_result], dim=RI_dim)
    return torch_result


def complex_mm(X: torch.Tensor, Y: torch.Tensor, RI_dim: int):
    """Perform complex matrix multiplication on real and imaginary stacked along specified dimension.

    The stacked dimension cannot be at last 2 dimensions of the tensor.

    Parameters
    ----------
    X : :obj:`torch.Tensor` of at least 3 dims
       Tensor with the real and imaginary stacked along the RI_dim.
       The matrix operation is performed on the last 2 dimensions.
    Y : :obj:`torch.Tensor` of at least 3 dims
       Tensor with the real and imaginary stacked along the RI_dim
    RI_dim : int
       The dimension of where the real and imag components are stacked.
       Cannot be at the last 2 dimensions of the tensor.

    """

    assert (
        X.shape[RI_dim] == Y.shape[RI_dim] == 2
    ), f"The real and imaginary stacked dimension, {RI_dim} must be at the same for X of size {X.shape} and Y of size {Y.shape}"
    # assert X.dim() == Y.dim(), "The 2 matrix must have the same number "
    n_dim = min(X.dim(), Y.dim())
    if RI_dim >= 0:
        assert (
            RI_dim < n_dim - 2
        ), "The real and imaginary component stack cannot be at the last 2 dimensions"
    if RI_dim < 0:
        assert (
            RI_dim < -2
        ), "The real and imaginary component stack cannot be at the last 2 dimensions"

    # two splits doesn't work when X and Y come from the same tensor.
    # X_rl, X_img = torch.split(X, 1,dim=RI_dim)
    # Y_rl, Y_img = torch.split(Y, 1,dim=RI_dim)
    _X = X.transpose(RI_dim, 0)
    X_rl, X_img = _X[0:1], _X[1:2]
    X_rl = X_rl.transpose(RI_dim, 0)
    X_img = X_img.transpose(RI_dim, 0)

    _Y = Y.transpose(RI_dim, 0)
    Y_rl, Y_img = _Y[0:1], _Y[1:2]
    Y_rl = Y_rl.transpose(RI_dim, 0)
    Y_img = Y_img.transpose(RI_dim, 0)

    # print(X_rl.shape)
    # print(X_img.shape)

    rl = X_rl @ Y_rl - X_img @ Y_img
    img = X_rl @ Y_img + X_img @ Y_rl

    return torch.cat([rl, img], dim=RI_dim)


def complex_inverse(X, RI_dim: int):
    r"""Perform complex inverse on real and imaginary stacked along specified dimension. 

   The complex inverse is performed using the real inverse following the equation:

   .. math::

      \mathbf{A} &= \mathbf{X}_{\Re} \\
      \mathbf{B} &= \mathbf{X}_{\Im} \\
      \mathbf{X}^{-1} &= [\mathbf{A} + \mathbf{BA}^{-1}\mathbf{B}]^{-1} -i[\mathbf{B} + \mathbf{AB}^{-1}\mathbf{A}]^{-1}

   The stacked dimension cannot be at last 2 dimensions of the tensor.
   
   Parameters
   ----------
   X : :obj:`torch.Tensor` of at least 3 dims
      Tensor with the real and imaginary stacked along the RI_dim. 
      The matrix operation is performed on the last 2 dimensions.
   RI_dim : int
      The dimension of where the real and imag components are stacked. 
      Cannot be at the last 2 dimensions of the tensor.
   """
    n_dim = len(X.shape)
    if RI_dim >= 0:
        assert (
            RI_dim < n_dim - 2
        ), "The real and imaginary component stack cannot be in the first 2 dimensions"
    if RI_dim < 0:
        assert (
            RI_dim < -2
        ), "The real and imaginary component stack cannot be in the first 2 dimensions"

    rl, img = torch.split(X, 1, dim=RI_dim)
    rl_inv = torch.inverse(rl)
    img_inv = torch.inverse(img)

    rl_result = rl + img @ rl_inv @ img
    rl_result = 1.0 * torch.inverse(rl_result)
    # rl_result = rl_result.type(torch.FloatTensor)
    img_result = img + rl @ img_inv @ rl
    img_result = -1.0 * torch.inverse(img_result)
    # img_result = img_result.type(torch.FloatTensor)

    inv = torch.cat([rl_result, img_result], dim=RI_dim)
    return inv


def complex_reciprocal(X, RI_dim: int):
    nume = conj(X, RI_dim)
    deno = complex_mul(X, conj(X, RI_dim), RI_dim)  # shape of 2, N
    X = nume / complex_abs(deno, RI_dim)
    return X


def complex_l2_norm(X, dim, RI_dim: int, keepdim: bool = False):
    return torch.sqrt(torch.sum(complex_abs(X, RI_dim) ** 2, dim=dim, keepdim=keepdim))
