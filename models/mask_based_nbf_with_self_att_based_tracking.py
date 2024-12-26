import math
import torch
import torch.nn as nn
from typing import Dict, Optional
from asteroid import ConvTasNet



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




class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, input_embedding_dim, embed_dim, num_heads):

        super().__init__()

        self.wq = torch.nn.Linear(input_embedding_dim, embed_dim, bias=False)

        self.wk = torch.nn.Linear(input_embedding_dim, embed_dim, bias=False)

        self.wv = torch.nn.Linear(input_embedding_dim, embed_dim, bias=False)

        self.wo = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        # self.rq
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0, "The input embedding size must be divisable by the number of heads."
        self.head_embedding_size=  int(embed_dim // num_heads)
        self.embed_dim = embed_dim
        

    def forward(self, x, v=None, need_weights=False):

        # *, seq, input_embedding
        q = self.wq.forward(x) # *, seq, embed_dim
        k = self.wk.forward(x) # *, seq, embed_dim

        seq, embed_dim = q.shape[-2:]

        v_given = not v is None
        v_head_embedding_size = None
        if not v_given:
            v = self.wv.forward(x) # *, seq, embed_dim
            # v_heads = torch.stack(torch.split(v, self.head_embedding_size, dim=-1), dim=-3) # *, num_heads, seq, head_embedding_size
            v_head_embedding_size = self.head_embedding_size
        else:
            v_embedding_size = v.shape[-1]

            assert v_embedding_size % self.num_heads == 0
            v_head_embedding_size = int(v_embedding_size // self.num_heads)

            # v_head_embedding_size = self.head_embedding_size if not v_given else 

        assert embed_dim == self.embed_dim

        q_heads = torch.stack(torch.split(q, self.head_embedding_size, dim=-1), dim=-3) # *, num_heads, seq, head_embedding_size
        k_heads = torch.stack(torch.split(k, self.head_embedding_size, dim=-1), dim=-3) # *, num_heads, seq, head_embedding_size
        v_heads = torch.stack(torch.split(v, v_head_embedding_size, dim=-1), dim=-3) # *, num_heads, seq, v_head_embedding_size

        assert q_heads.shape[-3:] == (self.num_heads, seq, self.head_embedding_size)

        if need_weights:

            weights = torch.softmax(q_heads @ k_heads.transpose(-1,-2)/ self.head_embedding_size ** 0.5, dim=-1)    # *, num_heads, seq, seq

            heads = weights @ v_heads # *, num_heads, seq, head_embedding_size
            # heads = heads.transpose(-2, -3) # *, seq, num_heads, head_embedding_size
            # heads = heads.reshape_as(q)
            # heads = hea
            cat_heads = torch.cat([heads[..., i,:,:] for i in range(self.num_heads)], -1)
            out = self.wo.forward(cat_heads)

            return out, weights
        else:
            heads = torch.softmax(q_heads @ k_heads.transpose(-1,-2)/ self.head_embedding_size ** 0.5, dim=-1) @ v_heads  # *, num_heads, seq, seq
            cat_heads = torch.cat([heads[..., i,:,:] for i in range(self.num_heads)], -1)


            if not v_given:
                out = self.wo.forward(cat_heads)
            else:
                out = cat_heads
        
            return out


        # x = torch.stack(torch.split(x, self.head_embedding_size, dim=-1), dim=-2), # *, seq, num_heads, head_embedding_size


        



class TransformerBlock(torch.nn.Module):

    def __init__(self, input_embedding_dim, embed_dim, num_heads, ffn_dim, vdim=None, kdim=None):

        super().__init__()
        # self.att = torch.nn.MultiheadAttention(embed_dim, num_heads, vdim=input_embedding_dim, kdim=input_embedding_dim, batch_first=True)
        self.att = MultiheadSelfAttention(input_embedding_dim, embed_dim, num_heads)
        # if input_embedding_dim != embed_dim:
            # self.residual_conn = torch.nn.Conv1d(in)
        self.ffn1 = torch.nn.Linear(embed_dim, ffn_dim)
        self.ffn2 = torch.nn.Linear(ffn_dim, embed_dim)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x):
        
        # print(x.shape)
        # mha, mha_weight =self.att.forward(x, x, x, need_weights=False)
        mha = self.att(x)
        # x = self.layer_norm(mha + x) # not sure they got use
        x = self.ffn1.forward(mha)
        x = torch.nn.functional.relu(x)
        x = self.ffn2.forward(x)

        return x


class Transformer(torch.nn.Module):
    def __init__(self, initial_dim, embed_dim, num_heads, ffn_dim, n_blocks):
        super().__init__()
        blocks = []

        # self.initial_proj = torch.nn.Linear(initial_dim, embed_dim)
        for i in range(n_blocks-1):
            if i ==0:
                block = TransformerBlock(initial_dim, embed_dim, num_heads, ffn_dim, kdim=embed_dim, vdim=embed_dim)
                # print(block.print("Query weight shape:", multihead_attn.in_proj_weight[:embed_dim].shape))
            else:
                block = TransformerBlock(embed_dim, embed_dim, num_heads, ffn_dim)
            blocks.append(block)

        self.blocks = torch.nn.Sequential(*blocks)

        self.last_att= TransformerBlock(embed_dim, embed_dim, num_heads, ffn_dim).att
        

    def pos_enc(self, seq_len, d_model):
        """
        Generate sinusoidal position encoding in PyTorch.
        
        :param seq_len: Length of the sequence (number of positions)
        :param d_model: Dimensionality of the model (i.e., the number of features per position)
        :return: A tensor of shape (seq_len, d_model) representing the position encodings
        """
        # Initialize a tensor of shape (seq_len, d_model)
        position_encoding = torch.zeros(seq_len, d_model)

        # Compute the position encodings
        positions = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)  # shape (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # shape (d_model / 2,)

        # Apply sin to the even indices and cos to the odd indices
        position_encoding[:, 0::2] = torch.sin(positions * div_term)  # Even indices
        position_encoding[:, 1::2] = torch.cos(positions * div_term)  # Odd indices

        return position_encoding


    def forward(self, x):
        
        
        x = x + self.pos_enc(x.shape[-2], x.shape[-1]).to(x.device)
        out  = self.blocks(x)
        # out = self.last_att(out, out, v=x, need_weights=False)[0] # take only the output 

        out = self.last_att(out, v=x)
        
        return out
    



class MaskBasedNBFwSelfAttBasedTrackingOracleBeamformer(
    torch.nn.Module
):  # jit.ScriptModule):
    def __init__(
        self,
        ref_mic_idx: int,
        STFT: STFT_module,
        # conv_block: torch.nn.Module,
        # compute_beamformer_weights: torch.nn.Module,
        n_mics: int = 1,
        num_heads:int =4,
        embed_dim: int=256,
        ffn_dim:int=2048,
        n_blocks:int=6,
        **kwargs,
    ):
        """

        [extended_summary]

        Parameters
        ----------
        ref_mic_idx : int
            The reference channel for evaluation
        STFT : STFT_module
            The STFT module from :ref:`STFT`
        alpha : float
            A scalar between 0 and 1, which controls the update between the SCM of previous enhanced speech frame and its even earlier frames
        n_mics : int, optional
            The number of microphone channels, by default 1
        n_kernels : int, optional
            The number of kernels for the convolutional layers in SMoLnet, by default 64
        kernel_size : int, optional
            The kernel size for the convolutional layers in SMoLnet, by default 3
        n_layers : int, optional
            The number of convolutional layers in SMoLnet, by default 10
        """

        super().__init__()

        self.STFT = STFT

        self.n_mics = n_mics
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.embed_dim = embed_dim
        self.ref_mic_idx = ref_mic_idx
        # self.alpha_noisy = alpha_noisy

        # init_weights = (
        #     torch.eye(self.n_mics).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # ) / self.n_mics
        # self.register_buffer("init_weights", init_weights)
        # torch.cuda.empty_cache()

        # self.build_tdu_net(
        #     conv_block=conv_block,
        #     in_channels=n_mics * 2, out_channels=n_mics * 2, 
        #     selector_rate=selector_rate,
        #     n_mics=n_mics,
        # )
        # self.build_atten()
        # self.att = torch.nn.MultiheadAttention(1000, 1) # for temporary usage

        initial_dim = self.n_mics ** 2 * (STFT.win_length // 2) * 2

        print(f"The initial embedding size is of {initial_dim}")
        # initial_dim = 5 * 5 * (STFT.win_length // 2) * 2
        # self.transformer_s = Transformer(initial_dim, embed_dim, num_heads, ffn_dim, n_blocks)
        # self.transformer_n = Transformer(initial_dim, embed_dim, num_heads, ffn_dim, n_blocks)
        self.dummy_weight = torch.nn.Parameter(torch.tensor([1.0]))
        
    def compute_oracle_mask(self, N_STFT, X_STFT, cdim):

        
        torch_zero = torch.Tensor([0]).to(N_STFT.device).to(torch.int)
        torch_one = torch.Tensor([1]).to(N_STFT.device).to(torch.int)

        torch_abs = lambda STFT, cdim=cdim: torch.sqrt(torch.index_select(STFT, cdim, torch_zero) ** 2 + torch.index_select(STFT, cdim, torch_one) ** 2)
        m = torch_abs(X_STFT) ** 2 / (torch_abs(X_STFT) **2 + torch_abs(N_STFT)**2)

        return m

    
    
    def compute_masked_SCM(self, mask, ISCM):
        """Computes the masked instantenous spatial covariance matrix.

            .. math ::
            \{si(k,t)= M(k,t)\cdot\Phi(k,t)

            where `\cdot`M(k,t)' is the estimated mask,  :math:`\cdot` is the element-wise product  :math:`\phi(k,t)` is the instantenous covariance matrix of the signal, :math:`k` is the frequency frame, :math:`t=1\ldots,T` is the time frame.

            Parameters
            ----------
            mask :obj:`torch.Tensor` of shape of *, C, T, F
            The magnitude mask, :math:`m(k,t)`

            ISCM :obj:`torch.Tensor` of shape of *, 2, F, T, C, C
            The spatial covariance matrix :math:`\phi(k,t)`, in real-imaginary stacked form

            Returns
            -------
            complex :obj:`numpy.ndarray` of shape *, 2, F, T, C, C
            masked ISCM :math:`\psi(k,t)`
        """
        C, T, F = mask.shape[-3:]
        
        mask = mask.transpose(-1, -3) # *, F, T, C
        mask = mask.unsqueeze(-4).unsqueeze(-1) # *, F,T,C,1
        # print(mask.shape)
        assert mask.shape[-4:] == (F, T, C, 1)



        return mask * ISCM

    def vectorize(self, SCM):
        """Vectorize an SCM to an 1D feature

            Parameters
            ----------

            SCM :obj:`torch.Tensor` of shape of *, 2, F, T, C, C
            The spatial covariance matrix :math:`\phi(k,t)`, in real-imaginary stacked form

            Returns
            -------
            complex :obj:`torch.Tensor`  of shape *, T,2FTCC
            Vectorized SCM 
        """
        T = SCM.shape[-3]
        SCM = SCM.transpose(-3, -5) # *, T, F, 2, C, C
        
        SCM = SCM.reshape(*SCM.shape[0:-5], T, -1)

        return SCM
    

    def complex_trace(self, x, RI_dim=1):
        trace = lambda mat:  mat.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1, keepdim=True)
        x_rl, x_img = x.split(1, RI_dim)
        x_abs = x_rl ** 2 + x_img **2
        return trace(x_abs)


    def unvectorize(self, vec_SCM, F, C):
        """Inverse function of Vectorize

            Parameters
            ----------

            vec_SCM :obj:`torch.Tensor` of shape of *, T, 2FCC
            The spatial covariance matrix :math:`\phi(k,t)`, in vectorzied from

            Returns
            -------
            complex :obj:`torch.Tensor`  of shape *, 2, F, T, C, C
            The spatial covariance matrix :math:`\phi(k,t)`, in real-imaginary stacked form
 
        """
        T = vec_SCM.shape[-2]
        vec_SCM = vec_SCM.reshape(*vec_SCM.shape[0:-2], T, F, 2, C, C)
        vec_SCM = vec_SCM.transpose(-3, -5) # *, 2, F, T, C, C

        return vec_SCM

    def diagonal_loading(self, x, eps=1e-10):

        C = x.shape[-1]
        trace = self.complex_trace(x, RI_dim=1) # B, 2, F, T, 1
        trace = trace.unsqueeze(-1) #  B, 2, F, T, 1, 1
        x = x + eps * trace * torch.eye(C, device=x.device)

        return x

    def forward_train(
        self, input: Dict[str, torch.Tensor],
    ):



        with torch.no_grad():
            
            y = input["mixture"]
            x = input["target"]
            v = input["noise"]

            # print(y.shape)
            length = y.shape[-1]
            y_stft = self.STFT.forward(y)  # shape of N, C, 2, T, F
            x_stft = self.STFT.forward(x)  # shape of N, C, 2, T, F
            v_stft = self.STFT.forward(v)  # shape of N, C, 2, T, F


            y_dc, y_stft =  y_stft[..., 0:1],  y_stft[..., 1:]
            x_dc, x_stft  =  x_stft[..., 0:1],  x_stft[..., 1:]
            v_dc, v_stft =  v_stft[..., 0:1],  v_stft[..., 1:]

            # set number of channels
            y_stft = y_stft[:, 0:self.n_mics]
            x_stft = x_stft[:, 0:self.n_mics]
            v_stft = v_stft[:, 0:self.n_mics]

            N, C, _, T, F = x_stft.shape
            mask = self.compute_oracle_mask(v_stft, x_stft, cdim=-3) # shape of N, C, 1, T, F
            # mask = torch.mean(mask, dim=1) # # shape of N, T, F
            
            mask = mask.transpose(1, 2) 
            mask = mask.squeeze(1)
            assert mask.shape == (N, C, T, F)
            noise_mask = 1 - mask


            y_stft = y_stft.transpose(1, 2) # shape of N, 2, C, T, F
            y_scm = torch_spatial_covariance(y_stft) # *, 2, F, T, C, C

            est_s_iscm = self.compute_masked_SCM(mask, y_scm) # *, 2, F, T, C, C
            est_n_iscm = self.compute_masked_SCM(noise_mask, y_scm) # *, 2, F, T, C, C
            
            est_n_iscm = torch.mean(est_n_iscm, dim=-3, keepdim=True)   # *, 2, F, T, C, C
            est_s_iscm = torch.mean(est_s_iscm, dim=-3, keepdim=True)   # *, 2, F, T, C, C
            # est_target = 

            # assert n_iscm.shape == s_iscm.shape
            # assert n_iscm.shape[-5:] ==  (2, F, T, C, C)

            # F = s_iscm.shape[-4]
            # C = s_iscm.shape[-1]

            # # vectorize 
            # s_iscm_vec = self.vectorize(s_iscm) # *, T, 2FCC
            # n_iscm_vec = self.vectorize(n_iscm) # *, T, 2FCC
        

        # est_s_iscm_vec = self.transformer_s(s_iscm_vec) # *, T, 2FCC
        # est_n_iscm_vec = self.transformer_n(n_iscm_vec) # *, T, 2FCC

        # est_s_iscm = self.unvectorize(est_s_iscm_vec, F, C) # *, 2, F, T, C, C
        # est_n_iscm = self.unvectorize(est_n_iscm_vec, F, C) # *, 2, F, T, C, C

        # computing the MVDR weights 

        est_n_iscm = self.diagonal_loading(est_n_iscm)
        
        est_n_iscm = est_n_iscm[:, 0] + 1j * est_n_iscm[:, 1] # *, F, T, C, C
        est_s_iscm = est_s_iscm[:, 0] + 1j * est_s_iscm[:, 1] # *, F, T, C, C

        est_n_inv_iscm = torch.inverse(est_n_iscm) # *, F, T, C, C
        weight = est_n_inv_iscm @ est_s_iscm    # *, F, T, C, C
        trace = lambda mat:  mat.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1, keepdim=True)
        trace_value = trace(weight)
        # print(weight.shape)
        trace_value = trace_value.unsqueeze(-1)
        weight = weight / trace_value # *, F, T, C, C
 
        u = torch.zeros_like(weight[..., :, 0:1])   # *, F, T, C, 1
        u[..., self.ref_mic_idx, :] += 1.0                  # *, F, T, C, 1

        weight = weight @ u # *, F, T, C, 1

        weight_H = torch.conj(weight).transpose(-1, -2) # *, F, T, 1, C
        y_stft_ = y_stft.transpose(-1, -3) # *, 2, F, T, C
        y_stft_ = y_stft_[:, 0] + 1j * y_stft_[:, 1]
        y_stft_ = y_stft_.unsqueeze(-1) # *, F, T, C, 1

        est_target_STFT = weight_H @ y_stft_  # *, F, T, 1, 1
        est_target_STFT = torch.stack([torch.real(est_target_STFT), torch.imag(est_target_STFT)], dim=1)

        est_target_STFT = est_target_STFT * self.dummy_weight / self.dummy_weight
        # using our own implementation of beamformer
        # est_n_inv_iscm = complex_inverse(est_n_iscm, RI_dim=1)

        # u = torch.zeros_like(est_n_inv_iscm[..., :, 0:1])   # *, 2, F, T, C, 1
        # u[..., self.ref_mic_idx, :] += 1.0                  # *, 2, F, T, C, 1

        # weight = complex_mm(est_n_inv_iscm, est_s_iscm, RI_dim=1) @ u # *, 2, F, T, C, 1
        # trace = self.complex_trace(complex_mm(est_n_inv_iscm, est_s_iscm, RI_dim=1)) # *, 2, F, T, 1
        # trace = trace.unsqueeze(-1) # *, 2, F, T, 1, 1
        
        # weight = weight / trace
        # weight_H = conj(weight.transpose(-1, -2), RI_dim=1) # *, 2, F, T, 1, C

        # N, 2, 1, T, F
        # y_stft = y_stft.transpose(-1, -3) # *, 2, F, T, C
        # y_stft = y_stft.unsqueeze(-1) # *, 2, F, T, C, 1

        # print("jhellfslahshdkh")
        
        # est_target_STFT = complex_mm(weight_H, y_stft, RI_dim=1) # *, 2, F, T, 1, 1
        est_target_STFT = est_target_STFT.squeeze(-1) # *, 2, F, T, 1
        est_target_STFT = est_target_STFT.transpose(-1, -3) # *, 2, 1, T, F
        est_target_STFT = est_target_STFT.transpose(-4, -3) # *, 1, 2, T, F

        # est_target_STFT = est_target_STFT
        c = self.ref_mic_idx
        est_target_STFT = torch.cat([y_dc[:, c:c+1], est_target_STFT], dim=-1)
        est_target = self.STFT.backward(est_target_STFT, length=length)
        # est_target_STFT_multi_channel =
        

        output = {
            "STFT": self.STFT,
            "est_target" : est_target[:, c],
            "est_target_STFT": est_target_STFT[:, c],
            "ref_mic_idx": self.ref_mic_idx,
            # "logging": {
            #     "Y_bf": {
            #         "ref_mic_idx": self.ref_mic_idx, 
            #         "STFT": self.STFT, 
            #         "est_target_multi_channel": Y_bf, 
            #         "est_target_STFT_multi_channel":Y_bf_STFT,
            #         "est_target": Y_bf[:, self.ref_mic_idx],
            #         "est_target_STFT": Y_bf_STFT[:, self.ref_mic_idx],
            #         },
                # "Y": {
                #     "ref_mic_idx": self.ref_mic_idx, 
                #     "STFT": self.STFT, 
                #     "est_target_multi_channel": input["mixture"], 
                #     "est_target_STFT_multi_channel":y_stft,
                #     "est_target": input["mixture"][:, self.ref_mic_idx],
                #     "est_target_STFT": y_tft[:, self.ref_mic_idx],
                #     },
                # "X_tilde_bf": {"ref_mic_idx": self.ref_mic_idx, "STFT": self.STFT, "est_target_multi_channel": X_tilde_bf, "est_target_STFT_multi_channel":X_tilde_bf_STFT},
                # "X_tilde": {"ref_mic_idx": self.ref_mic_idx, "STFT": self.STFT, "est_target_multi_channel": X_tilde, "est_target_STFT_multi_channel":X_tilde_STFT},
            # },
        }

        return output

    def precompute_inv_phi_Y(self, Y) -> torch.Tensor:
        phi_Y = torch_spatial_covariance(Y)  # N, 2, F, T, C, C
        phi_Y = torch.mean(phi_Y, dim=3, keepdim=True)  # N, 2, F, 1, C, C
        inv_phi_Y = diag_load_and_inverse(phi_Y)  # N, 2, F, 1, C, C
        return inv_phi_Y  # N, 2, F, 1, C, C

    def update_beamformer_weight(self, t: int, X_t, init_phi_X, inv_phi_Y_t):
        # phi_Y_t = torch_spatial_covariance(Y_t)

        if t == 0:

            self.phi_X = init_phi_X

        if t > 0:
            phi_X_t = torch_spatial_covariance(X_t)
            self.phi_X = phi_X_t

        if isinstance(self.phi_X, float):
            C = self.n_mics
            N = X_t.shape[0]
            F = X_t.shape[-1]
            device = X_t.device
            W = self.init_weights.repeat(N, 2, F, 1, 1, 1)

        else:
            W = self.compute_beamformer_weights(self.phi_X, inv_phi_Y_t)

        return W

    @torch.jit.export
    def forward(
        self, input: Dict[str, torch.Tensor], test: bool = False
    ) -> Dict[str, torch.Tensor]:

        if self.training:
            return self.forward_train(input)
        else:
            return self.forward_train(input)

    # def training_epoch_end(self, pl_module):
    #     # print("hello")
    #     # pl_module.logger.experiment.add_histogram(
    #     #     "filter_selector_y",
    #     #     self.filter_selector_y.weight.squeeze(),
    #     #     global_step=pl_module.current_epoch,
    #     # )
    #     # pl_module.logger.experiment.add_histogram(
    #     #     "filter_selector_xbf",
    #     #     self.filter_selector_bf.weight.squeeze(),
    #     #     global_step=pl_module.current_epoch,
    #     # )



class MaskBasedNBFwSelfAttBasedTrackingOracleMaskedOutput(
    torch.nn.Module
):  # jit.ScriptModule):
    def __init__(
        self,
        ref_mic_idx: int,
        STFT: STFT_module,
        # conv_block: torch.nn.Module,
        # compute_beamformer_weights: torch.nn.Module,
        n_mics: int = 1,
        num_heads:int =4,
        embed_dim: int=256,
        ffn_dim:int=2048,
        n_blocks:int=6,
        **kwargs,
    ):
        """

        [extended_summary]

        Parameters
        ----------
        ref_mic_idx : int
            The reference channel for evaluation
        STFT : STFT_module
            The STFT module from :ref:`STFT`
        alpha : float
            A scalar between 0 and 1, which controls the update between the SCM of previous enhanced speech frame and its even earlier frames
        n_mics : int, optional
            The number of microphone channels, by default 1
        n_kernels : int, optional
            The number of kernels for the convolutional layers in SMoLnet, by default 64
        kernel_size : int, optional
            The kernel size for the convolutional layers in SMoLnet, by default 3
        n_layers : int, optional
            The number of convolutional layers in SMoLnet, by default 10
        """

        super().__init__()

        self.STFT = STFT

        self.n_mics = n_mics
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.embed_dim = embed_dim
        self.ref_mic_idx = ref_mic_idx
        # self.alpha_noisy = alpha_noisy

        # init_weights = (
        #     torch.eye(self.n_mics).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # ) / self.n_mics
        # self.register_buffer("init_weights", init_weights)
        # torch.cuda.empty_cache()

        # self.build_tdu_net(
        #     conv_block=conv_block,
        #     in_channels=n_mics * 2, out_channels=n_mics * 2, 
        #     selector_rate=selector_rate,
        #     n_mics=n_mics,
        # )
        # self.build_atten()
        # self.att = torch.nn.MultiheadAttention(1000, 1) # for temporary usage

        initial_dim = self.n_mics ** 2 * (STFT.win_length // 2) * 2

        print(f"The initial embedding size is of {initial_dim}")
        # initial_dim = 5 * 5 * (STFT.win_length // 2) * 2
        # self.transformer_s = Transformer(initial_dim, embed_dim, num_heads, ffn_dim, n_blocks)
        # self.transformer_n = Transformer(initial_dim, embed_dim, num_heads, ffn_dim, n_blocks)
        self.dummy_weight = torch.nn.Parameter(torch.tensor([1.0]))
        
    def compute_oracle_mask(self, N_STFT, X_STFT, cdim):

        
        torch_zero = torch.Tensor([0]).to(N_STFT.device).to(torch.int)
        torch_one = torch.Tensor([1]).to(N_STFT.device).to(torch.int)

        torch_abs = lambda STFT, cdim=cdim: torch.sqrt(torch.index_select(STFT, cdim, torch_zero) ** 2 + torch.index_select(STFT, cdim, torch_one) ** 2)
        m = torch_abs(X_STFT) ** 2 / (torch_abs(X_STFT) **2 + torch_abs(N_STFT)**2)

        return m

    
    
    def compute_masked_SCM(self, mask, ISCM):
        """Computes the masked instantenous spatial covariance matrix.

            .. math ::
            \{si(k,t)= M(k,t)\cdot\Phi(k,t)

            where `\cdot`M(k,t)' is the estimated mask,  :math:`\cdot` is the element-wise product  :math:`\phi(k,t)` is the instantenous covariance matrix of the signal, :math:`k` is the frequency frame, :math:`t=1\ldots,T` is the time frame.

            Parameters
            ----------
            mask :obj:`torch.Tensor` of shape of *, C, T, F
            The magnitude mask, :math:`m(k,t)`

            ISCM :obj:`torch.Tensor` of shape of *, 2, F, T, C, C
            The spatial covariance matrix :math:`\phi(k,t)`, in real-imaginary stacked form

            Returns
            -------
            complex :obj:`numpy.ndarray` of shape *, 2, F, T, C, C
            masked ISCM :math:`\psi(k,t)`
        """
        C, T, F = mask.shape[-3:]
        
        mask = mask.transpose(-1, -3) # *, F, T, C
        mask = mask.unsqueeze(-4).unsqueeze(-1) # *, F,T,C,1
        # print(mask.shape)x
        assert mask.shape[-4:] == (F, T, C, 1)



        return mask * ISCM

    def vectorize(self, SCM):
        """Vectorize an SCM to an 1D feature

            Parameters
            ----------

            SCM :obj:`torch.Tensor` of shape of *, 2, F, T, C, C
            The spatial covariance matrix :math:`\phi(k,t)`, in real-imaginary stacked form

            Returns
            -------
            complex :obj:`torch.Tensor`  of shape *, T,2FTCC
            Vectorized SCM 
        """
        T = SCM.shape[-3]
        SCM = SCM.transpose(-3, -5) # *, T, F, 2, C, C
        
        SCM = SCM.reshape(*SCM.shape[0:-5], T, -1)

        return SCM
    

    def complex_trace(self, x, RI_dim=1):
        trace = lambda mat:  mat.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1, keepdim=True)
        x_rl, x_img = x.split(1, RI_dim)
        x_abs = x_rl ** 2 + x_img **2
        return trace(x_abs)


    def unvectorize(self, vec_SCM, F, C):
        """Inverse function of Vectorize

            Parameters
            ----------

            vec_SCM :obj:`torch.Tensor` of shape of *, T, 2FCC
            The spatial covariance matrix :math:`\phi(k,t)`, in vectorzied from

            Returns
            -------
            complex :obj:`torch.Tensor`  of shape *, 2, F, T, C, C
            The spatial covariance matrix :math:`\phi(k,t)`, in real-imaginary stacked form
 
        """
        T = vec_SCM.shape[-2]
        vec_SCM = vec_SCM.reshape(*vec_SCM.shape[0:-2], T, F, 2, C, C)
        vec_SCM = vec_SCM.transpose(-3, -5) # *, 2, F, T, C, C

        return vec_SCM

    def diagonal_loading(self, x, eps=1e-10):

        C = x.shape[-1]
        trace = self.complex_trace(x, RI_dim=1) # B, 2, F, T, 1
        trace = trace.unsqueeze(-1) #  B, 2, F, T, 1, 1
        x = x + eps * trace * torch.eye(C, device=x.device)

        return x

    def forward_train(
        self, input: Dict[str, torch.Tensor],
    ):



        # with torch.no_grad():
            
        y = input["mixture"]
        x = input["target"]
        v = input["noise"]

            # print(y.shape)
        length = y.shape[-1]
        y_stft = self.STFT.forward(y)  # shape of N, C, 2, T, F
        x_stft = self.STFT.forward(x)  # shape of N, C, 2, T, F
        v_stft = self.STFT.forward(v)  # shape of N, C, 2, T, F


        y_dc, y_stft =  y_stft[..., 0:1],  y_stft[..., 1:]
        x_dc, x_stft  =  x_stft[..., 0:1],  x_stft[..., 1:]
        v_dc, v_stft =  v_stft[..., 0:1],  v_stft[..., 1:]

        # set number of channels
        y_stft = y_stft[:, 0:self.n_mics]
        x_stft = x_stft[:, 0:self.n_mics]
        v_stft = v_stft[:, 0:self.n_mics]

        N, C, _, T, F = x_stft.shape
        mask = self.compute_oracle_mask(v_stft, x_stft, cdim=-3) # shape of N, C, 1, T, F
        # mask = torch.mean(mask, dim=1) # # shape of N, T, F
        
        mask = mask.transpose(1, 2) # shape of N, 1, C, T, F
        # mask = mask.squeeze(1)
        assert mask.shape == (N, 1, C, T, F)
        noise_mask = 1 - mask


        y_stft = y_stft.transpose(1, 2) # shape of N, 2, C, T, F
        est_target_STFT = y_stft * mask # shape of N, 2, C, T, F
        est_target_STFT = est_target_STFT.transpose(1,2)
            # y_scm = torch_spatial_covariance(y_stft) # *, 2, F, T, C, C
            # apply mask 
            
            # est_s_iscm = self.compute_masked_SCM(mask, y_scm) # *, 2, F, T, C, C
            # est_n_iscm = self.compute_masked_SCM(noise_mask, y_scm) # *, 2, F, T, C, C

            # est_target = 

            # assert n_iscm.shape == s_iscm.shape
            # assert n_iscm.shape[-5:] ==  (2, F, T, C, C)

            # F = s_iscm.shape[-4]
            # C = s_iscm.shape[-1]

            # # vectorize 
            # s_iscm_vec = self.vectorize(s_iscm) # *, T, 2FCC
            # n_iscm_vec = self.vectorize(n_iscm) # *, T, 2FCC
        

        # est_s_iscm_vec = self.transformer_s(s_iscm_vec) # *, T, 2FCC
        # est_n_iscm_vec = self.transformer_n(n_iscm_vec) # *, T, 2FCC

        # est_s_iscm = self.unvectorize(est_s_iscm_vec, F, C) # *, 2, F, T, C, C
        # est_n_iscm = self.unvectorize(est_n_iscm_vec, F, C) # *, 2, F, T, C, C

        # computing the MVDR weights 

        # est_n_iscm = self.diagonal_loading(est_n_iscm)

        # est_n_inv_iscm = complex_inverse(est_n_iscm, RI_dim=1)
        # # todo: check the formulation again

        # u = torch.zeros_like(est_n_inv_iscm[..., :, 0:1])   # *, 2, F, T, C, 1
        # u[..., self.ref_mic_idx, :] += 1.0                  # *, 2, F, T, C, 1

        # weight = complex_mm(est_n_inv_iscm, est_s_iscm, RI_dim=1) @ u # *, 2, F, T, C, 1
        # trace = self.complex_trace(complex_mm(est_n_inv_iscm, est_s_iscm, RI_dim=1)) # *, 2, F, T, 1
        # trace = trace.unsqueeze(-1) # *, 2, F, T, 1, 1
        
        # weight = weight / trace
        # weight_H = conj(weight.transpose(-1, -2), RI_dim=1) # *, 2, F, T, 1, C

        # N, 2, 1, T, F
        # y_stft = y_stft.transpose(-1, -3) # *, 2, F, T, C
        # y_stft = y_stft.unsqueeze(-1) # *, 2, F, T, C, 1

        
        # est_target_STFT = complex_mm(weight_H, y_stft, RI_dim=1) # *, 2, F, T, 1, 1
        # est_target_STFT = est_target_STFT.squeeze(-1) # *, 2, F, T, 1
        # est_target_STFT = est_target_STFT.transpose(-1, -3) # *, 2, 1, T, F
        # est_target_STFT = est_target_STFT.transpose(-4, -3) # *, 1, 2, T, F

        c = self.ref_mic_idx
        # y_dc: *, 2,  
        est_target_STFT = torch.cat([y_dc[:, c:c+1], est_target_STFT[:, c:c+1]], dim=-1)
        # self.dummy_weight = self.dummy_weight/self.dummy_weight
        est_target_STFT = est_target_STFT * self.dummy_weight / self.dummy_weight
        est_target = self.STFT.backward(est_target_STFT, length=length)
        # est_target_STFT_multi_channel =
        

        output = {
            "STFT": self.STFT,
            "est_target" : est_target[:, c],
            "est_target_STFT": est_target_STFT[:, c] ,
            "ref_mic_idx": self.ref_mic_idx,
            # "logging": {
            #     "Y_bf": {
            #         "ref_mic_idx": self.ref_mic_idx, 
            #         "STFT": self.STFT, 
            #         "est_target_multi_channel": Y_bf, 
            #         "est_target_STFT_multi_channel":Y_bf_STFT,
            #         "est_target": Y_bf[:, self.ref_mic_idx],
            #         "est_target_STFT": Y_bf_STFT[:, self.ref_mic_idx],
            #         },
                # "Y": {
                #     "ref_mic_idx": self.ref_mic_idx, 
                #     "STFT": self.STFT, 
                #     "est_target_multi_channel": input["mixture"], 
                #     "est_target_STFT_multi_channel":y_stft,
                #     "est_target": input["mixture"][:, self.ref_mic_idx],
                #     "est_target_STFT": y_tft[:, self.ref_mic_idx],
                #     },
                # "X_tilde_bf": {"ref_mic_idx": self.ref_mic_idx, "STFT": self.STFT, "est_target_multi_channel": X_tilde_bf, "est_target_STFT_multi_channel":X_tilde_bf_STFT},
                # "X_tilde": {"ref_mic_idx": self.ref_mic_idx, "STFT": self.STFT, "est_target_multi_channel": X_tilde, "est_target_STFT_multi_channel":X_tilde_STFT},
            # },
        }

        return output

    def precompute_inv_phi_Y(self, Y) -> torch.Tensor:
        phi_Y = torch_spatial_covariance(Y)  # N, 2, F, T, C, C
        phi_Y = torch.mean(phi_Y, dim=3, keepdim=True)  # N, 2, F, 1, C, C
        inv_phi_Y = diag_load_and_inverse(phi_Y)  # N, 2, F, 1, C, C
        return inv_phi_Y  # N, 2, F, 1, C, C

    def update_beamformer_weight(self, t: int, X_t, init_phi_X, inv_phi_Y_t):
        # phi_Y_t = torch_spatial_covariance(Y_t)

        if t == 0:

            self.phi_X = init_phi_X

        if t > 0:
            phi_X_t = torch_spatial_covariance(X_t)
            self.phi_X = phi_X_t

        if isinstance(self.phi_X, float):
            C = self.n_mics
            N = X_t.shape[0]
            F = X_t.shape[-1]
            device = X_t.device
            W = self.init_weights.repeat(N, 2, F, 1, 1, 1)

        else:
            W = self.compute_beamformer_weights(self.phi_X, inv_phi_Y_t)

        return W

    @torch.jit.export
    def forward(
        self, input: Dict[str, torch.Tensor], test: bool = False
    ) -> Dict[str, torch.Tensor]:

        if self.training:
            return self.forward_train(input)
        else:
            return self.forward_train(input)

    # def training_epoch_end(self, pl_module):
    #     # print("hello")
    #     # pl_module.logger.experiment.add_histogram(
    #     #     "filter_selector_y",
    #     #     self.filter_selector_y.weight.squeeze(),
    #     #     global_step=pl_module.current_epoch,
    #     # )
    #     # pl_module.logger.experiment.add_histogram(
    #     #     "filter_selector_xbf",
    #     #     self.filter_selector_bf.weight.squeeze(),
    #     #     global_step=pl_module.current_epoch,
    #     # )


class MaskBasedNBFwSelfAttBasedTracking(
    torch.nn.Module
):  # jit.ScriptModule):
    def __init__(
        self,
        ref_mic_idx: int,
        STFT: STFT_module,
        # conv_block: torch.nn.Module,
        # compute_beamformer_weights: torch.nn.Module,
        n_mics: int = 1,
        num_heads:int =4,
        embed_dim: int=256,
        ffn_dim:int=2048,
        n_blocks:int=6,
        **kwargs,
    ):
        """

        [extended_summary]

        Parameters
        ----------
        ref_mic_idx : int
            The reference channel for evaluation
        STFT : STFT_module
            The STFT module from :ref:`STFT`
        alpha : float
            A scalar between 0 and 1, which controls the update between the SCM of previous enhanced speech frame and its even earlier frames
        n_mics : int, optional
            The number of microphone channels, by default 1
        n_kernels : int, optional
            The number of kernels for the convolutional layers in SMoLnet, by default 64
        kernel_size : int, optional
            The kernel size for the convolutional layers in SMoLnet, by default 3
        n_layers : int, optional
            The number of convolutional layers in SMoLnet, by default 10
        """

        super().__init__()

        self.STFT = STFT

        self.n_mics = n_mics
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.embed_dim = embed_dim
        self.ref_mic_idx = ref_mic_idx
        # self.alpha_noisy = alpha_noisy

        # init_weights = (
        #     torch.eye(self.n_mics).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # ) / self.n_mics
        # self.register_buffer("init_weights", init_weights)
        # torch.cuda.empty_cache()

        # self.build_tdu_net(
        #     conv_block=conv_block,
        #     in_channels=n_mics * 2, out_channels=n_mics * 2, 
        #     selector_rate=selector_rate,
        #     n_mics=n_mics,
        # )
        # self.build_atten()
        # self.att = torch.nn.MultiheadAttention(1000, 1) # for temporary usage

        initial_dim = self.n_mics ** 2 * (STFT.win_length // 2) * 2

        print(f"The initial embedding size is of {initial_dim}")
        # initial_dim = 5 * 5 * (STFT.win_length // 2) * 2
        self.transformer_s = Transformer(initial_dim, embed_dim, num_heads, ffn_dim, n_blocks)
        self.transformer_n = Transformer(initial_dim, embed_dim, num_heads, ffn_dim, n_blocks)

        
    def compute_oracle_mask(self, N_STFT, X_STFT, cdim):

        
        torch_zero = torch.Tensor([0]).to(N_STFT.device).to(torch.int)
        torch_one = torch.Tensor([1]).to(N_STFT.device).to(torch.int)

        torch_abs = lambda STFT, cdim=cdim: torch.sqrt(torch.index_select(STFT, cdim, torch_zero) ** 2 + torch.index_select(STFT, cdim, torch_one) ** 2)
        m = torch_abs(X_STFT) ** 2 / (torch_abs(X_STFT) **2 + torch_abs(N_STFT)**2)

        return m
    
    def compute_masked_SCM(self, mask, ISCM):
        """Computes the masked instantenous spatial covariance matrix.

            .. math ::
            \{si(k,t)= M(k,t)\cdot\Phi(k,t)

            where `\cdot`M(k,t)' is the estimated mask,  :math:`\cdot` is the element-wise product  :math:`\phi(k,t)` is the instantenous covariance matrix of the signal, :math:`k` is the frequency frame, :math:`t=1\ldots,T` is the time frame.

            Parameters
            ----------
            mask :obj:`torch.Tensor` of shape of *, C, T, F
            The magnitude mask, :math:`m(k,t)`

            ISCM :obj:`torch.Tensor` of shape of *, 2, F, T, C, C
            The spatial covariance matrix :math:`\phi(k,t)`, in real-imaginary stacked form

            Returns
            -------
            complex :obj:`numpy.ndarray` of shape *, 2, F, T, C, C
            masked ISCM :math:`\psi(k,t)`
        """
        C, T, F = mask.shape[-3:]
        
        mask = mask.transpose(-1, -3) # *, F, T, C
        mask = mask.unsqueeze(-4).unsqueeze(-1) # *, F,T,C,1
        # print(mask.shape)
        assert mask.shape[-4:] == (F, T, C, 1)



        return mask * ISCM

    def vectorize(self, SCM):
        """Vectorize an SCM to an 1D feature

            Parameters
            ----------

            SCM :obj:`torch.Tensor` of shape of *, 2, F, T, C, C
            The spatial covariance matrix :math:`\phi(k,t)`, in real-imaginary stacked form

            Returns
            -------
            complex :obj:`torch.Tensor`  of shape *, T,2FTCC
            Vectorized SCM 
        """
        T = SCM.shape[-3]
        SCM = SCM.transpose(-3, -5) # *, T, F, 2, C, C
        
        SCM = SCM.reshape(*SCM.shape[0:-5], T, -1)

        return SCM
    

    def complex_trace(self, x, RI_dim=1):
        trace = lambda mat:  mat.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1, keepdim=True)
        x_rl, x_img = x.split(1, RI_dim)
        x_abs = x_rl ** 2 + x_img **2
        return trace(x_abs)


    def unvectorize(self, vec_SCM, F, C):
        """Inverse function of Vectorize

            Parameters
            ----------

            vec_SCM :obj:`torch.Tensor` of shape of *, T, 2FCC
            The spatial covariance matrix :math:`\phi(k,t)`, in vectorzied from

            Returns
            -------
            complex :obj:`torch.Tensor`  of shape *, 2, F, T, C, C
            The spatial covariance matrix :math:`\phi(k,t)`, in real-imaginary stacked form
 
        """
        T = vec_SCM.shape[-2]
        vec_SCM = vec_SCM.reshape(*vec_SCM.shape[0:-2], T, F, 2, C, C)
        vec_SCM = vec_SCM.transpose(-3, -5) # *, 2, F, T, C, C

        return vec_SCM

    def diagonal_loading(self, x, eps=1e-10):

        C = x.shape[-1]
        trace = self.complex_trace(x, RI_dim=1) # B, 2, F, T, 1
        trace = trace.unsqueeze(-1) #  B, 2, F, T, 1, 1
        x = x + eps * trace * torch.eye(C, device=x.device)

        return x

    def forward_train(
        self, input: Dict[str, torch.Tensor],
    ):



        with torch.no_grad():
            
            y = input["mixture"]
            x = input["target"]
            v = input["noise"]

            # print(y.shape)
            length = y.shape[-1]
            y_stft = self.STFT.forward(y)  # shape of N, C, 2, T, F
            x_stft = self.STFT.forward(x)  # shape of N, C, 2, T, F
            v_stft = self.STFT.forward(v)  # shape of N, C, 2, T, F


            y_dc, y_stft =  y_stft[..., 0:1],  y_stft[..., 1:]
            x_dc, x_stft  =  x_stft[..., 0:1],  x_stft[..., 1:]
            v_dc, v_stft =  v_stft[..., 0:1],  v_stft[..., 1:]

            # set number of channels
            y_stft = y_stft[:, 0:self.n_mics]
            x_stft = x_stft[:, 0:self.n_mics]
            v_stft = v_stft[:, 0:self.n_mics]

            N, C, _, T, F = x_stft.shape
            mask = self.compute_oracle_mask(v_stft, x_stft, cdim=-3) # shape of N, C, 1, T, F
            # mask = torch.mean(mask, dim=1) # # shape of N, T, F
            
            mask = mask.transpose(1, 2) 
            mask = mask.squeeze(1)
            assert mask.shape == (N, C, T, F)
            noise_mask = 1 - mask


            y_stft = y_stft.transpose(1, 2) # shape of N, 2, C, T, F
            y_scm = torch_spatial_covariance(y_stft) # *, 2, F, T, C, C

            s_iscm = self.compute_masked_SCM(mask, y_scm) # *, 2, F, T, C, C
            n_iscm = self.compute_masked_SCM(noise_mask, y_scm) # *, 2, F, T, C, C

            assert n_iscm.shape == s_iscm.shape
            assert n_iscm.shape[-5:] ==  (2, F, T, C, C)

            F = s_iscm.shape[-4]
            C = s_iscm.shape[-1]

            # vectorize 
            s_iscm_vec = self.vectorize(s_iscm) # *, T, 2FCC
            n_iscm_vec = self.vectorize(n_iscm) # *, T, 2FCC
        

        est_s_iscm_vec = self.transformer_s(s_iscm_vec) # *, T, 2FCC
        est_n_iscm_vec = self.transformer_n(n_iscm_vec) # *, T, 2FCC

        est_s_iscm = self.unvectorize(est_s_iscm_vec, F, C) # *, 2, F, T, C, C
        est_n_iscm = self.unvectorize(est_n_iscm_vec, F, C) # *, 2, F, T, C, C

        # computing the MVDR weights 
        est_n_iscm = self.diagonal_loading(est_n_iscm)
        
        est_n_iscm = est_n_iscm[:, 0] + 1j * est_n_iscm[:, 1] # *, F, T, C, C
        est_s_iscm = est_s_iscm[:, 0] + 1j * est_s_iscm[:, 1] # *, F, T, C, C

        est_n_inv_iscm = torch.inverse(est_n_iscm)  # *, F, T, C, C
        weight = est_n_inv_iscm @ est_s_iscm        # *, F, T, C, C
        trace = lambda mat:  mat.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1, keepdim=True)
        trace_value = trace(weight)
        # print(weight.shape)
        trace_value = trace_value.unsqueeze(-1)
        weight = weight / trace_value # *, F, T, C, C

        u = torch.zeros_like(weight[..., :, 0:1])   # *, F, T, C, 1
        u[..., self.ref_mic_idx, :] += 1.0                  # *, F, T, C, 1

        weight = weight @ u # *, F, T, C, 1

        weight_H = torch.conj(weight).transpose(-1, -2) # *, F, T, 1, C
        y_stft_ = y_stft.transpose(-1, -3) # *, 2, F, T, C
        y_stft_ = y_stft_[:, 0] + 1j * y_stft_[:, 1]
        y_stft_ = y_stft_.unsqueeze(-1) # *, F, T, C, 1

        est_target_STFT = weight_H @ y_stft_  # *, F, T, 1, 1
        est_target_STFT = torch.stack([torch.real(est_target_STFT), torch.imag(est_target_STFT)], dim=1)

        est_target_STFT = est_target_STFT.squeeze(-1) # *, 2, F, T, 1
        est_target_STFT = est_target_STFT.transpose(-1, -3) # *, 2, 1, T, F
        est_target_STFT = est_target_STFT.transpose(-4, -3) # *, 1, 2, T, F

        # est_target_STFT = est_target_STFT
        c = self.ref_mic_idx
        est_target_STFT = torch.cat([y_dc[:, c:c+1], est_target_STFT], dim=-1)
        est_target = self.STFT.backward(est_target_STFT, length=length)

        output = {
            "STFT": self.STFT,
            "est_target" : est_target[:, c],
            "est_target_STFT": est_target_STFT[:, c],
            "ref_mic_idx": self.ref_mic_idx,
            # "logging": {
            #     "Y_bf": {
            #         "ref_mic_idx": self.ref_mic_idx, 
            #         "STFT": self.STFT, 
            #         "est_target_multi_channel": Y_bf, 
            #         "est_target_STFT_multi_channel":Y_bf_STFT,
            #         "est_target": Y_bf[:, self.ref_mic_idx],
            #         "est_target_STFT": Y_bf_STFT[:, self.ref_mic_idx],
            #         },
                # "Y": {
                #     "ref_mic_idx": self.ref_mic_idx, 
                #     "STFT": self.STFT, 
                #     "est_target_multi_channel": input["mixture"], 
                #     "est_target_STFT_multi_channel":y_stft,
                #     "est_target": input["mixture"][:, self.ref_mic_idx],
                #     "est_target_STFT": y_tft[:, self.ref_mic_idx],
                #     },
                # "X_tilde_bf": {"ref_mic_idx": self.ref_mic_idx, "STFT": self.STFT, "est_target_multi_channel": X_tilde_bf, "est_target_STFT_multi_channel":X_tilde_bf_STFT},
                # "X_tilde": {"ref_mic_idx": self.ref_mic_idx, "STFT": self.STFT, "est_target_multi_channel": X_tilde, "est_target_STFT_multi_channel":X_tilde_STFT},
            # },
        }

        return output

    # @torch.jit.export
    def forward(
        self, input: Dict[str, torch.Tensor], test: bool = False
    ) -> Dict[str, torch.Tensor]:

        if self.training:
            return self.forward_train(input)
        else:
            return self.forward_train(input)

    # def training_epoch_end(self, pl_module):
    #     # print("hello")
    #     # pl_module.logger.experiment.add_histogram(
    #     #     "filter_selector_y",
    #     #     self.filter_selector_y.weight.squeeze(),
    #     #     global_step=pl_module.current_epoch,
    #     # )
    #     # pl_module.logger.experiment.add_histogram(
    #     #     "filter_selector_xbf",
    #     #     self.filter_selector_bf.weight.squeeze(),
    #     #     global_step=pl_module.current_epoch,
    #     # )




class MaskBasedNBFwSelfAttBasedTrackingFirstStage(
    torch.nn.Module
):  # jit.ScriptModule):
    def __init__(
        self,
        ref_mic_idx: int,
        STFT_m: STFT_module,
        # conv_block: torch.nn.Module,
        # compute_beamformer_weights: torch.nn.Module,
        n_mics: int = 1,
        **kwargs,
    ):
        """

        [extended_summary]

        Parameters
        ----------
        ref_mic_idx : int
            The reference channel for evaluation
        STFT : STFT_module
            The STFT module from :ref:`STFT`
        n_mics : int, optional
            The number of microphone channels, by default 1
        
        
        """

        super().__init__()

        self.STFT = STFT_m

        self.n_mics = n_mics

        self.ref_mic_idx = ref_mic_idx

        # initial_dim = 5 * 5 * (STFT.win_length // 2) * 2
        # self.transformer_s = Transformer(initial_dim, embed_dim, num_heads, ffn_dim, n_blocks)
        # self.transformer_n = Transformer(initial_dim, embed_dim, num_heads, ffn_dim, n_blocks)
        self.ConvTasNet = ConvTasNet(n_src=1, out_chan=1, n_blocks=8, n_repeats=4, bn_chan=256, hid_chan=512)
        # ConvTasNet_MIMO_freq(n_src=1, ref_mic_idx=self.ref_mic_idx, STFT_module=self.STFT, )

    def forward_train(
        self, input: Dict[str, torch.Tensor],
    ):

        y = input["mixture"]
        x = input["target"]
        v = input["noise"]

        # print(y.shape)
        length = y.shape[-1]

        y = self.STFT(y) # *, C, 2, T, F

        y_abs = torch.sqrt( y[:, :, 0] ** 2 + y[:, :, 1] ** 2 ) # *, C, T, F

        C = y.shape[1]
        x_hat =[]
        for c in range(C):
            y_in = y_abs[:, c]
            print(y_in.shape)
            _x_hat = self.ConvTasNet.forward(y_in)
            x_hat.append(_x_hat)

        x_hat = torch.stack(x_hat, dim=1)
        return out

        # # computing the MVDR weights 

        # # N, 2, 1, T, F
        # y_stft = y_stft.transpose(-1, -3) # *, 2, F, T, C
        # y_stft = y_stft.unsqueeze(-1) # *, 2, F, T, C, 1

        
        
        # est_target_STFT = complex_mm(weight_H, y_stft, RI_dim=1) # *, 2, F, T, C, 1
        # est_target_STFT = est_target_STFT.squeeze(-1) # *, 2, F, T, C
        # est_target_STFT = est_target_STFT.transpose(-1, -3) # *, 2, C, T, F
        # est_target_STFT = est_target_STFT.transpose(-4, -3) # *, C, 2, T, F

        # c = self.ref_mic_idx
        # est_target_STFT = torch.cat([y_dc[:, c:c+1], est_target_STFT], dim=-1)
        # est_target = self.STFT.backward(est_target_STFT, length=length)
        # # est_target_STFT_multi_channel =
        

        # output = {
        #     "STFT": self.STFT,
        #     "est_target" : est_target,
        #     "est_target_STFT": est_target_STFT,
        #     "ref_mic_idx": self.ref_mic_idx,
            # "logging": {
            #     "Y_bf": {
            #         "ref_mic_idx": self.ref_mic_idx, 
            #         "STFT": self.STFT, 
            #         "est_target_multi_channel": Y_bf, 
            #         "est_target_STFT_multi_channel":Y_bf_STFT,
            #         "est_target": Y_bf[:, self.ref_mic_idx],
            #         "est_target_STFT": Y_bf_STFT[:, self.ref_mic_idx],
            #         },
                # "Y": {
                #     "ref_mic_idx": self.ref_mic_idx, 
                #     "STFT": self.STFT, 
                #     "est_target_multi_channel": input["mixture"], 
                #     "est_target_STFT_multi_channel":y_stft,
                #     "est_target": input["mixture"][:, self.ref_mic_idx],
                #     "est_target_STFT": y_tft[:, self.ref_mic_idx],
                #     },
                # "X_tilde_bf": {"ref_mic_idx": self.ref_mic_idx, "STFT": self.STFT, "est_target_multi_channel": X_tilde_bf, "est_target_STFT_multi_channel":X_tilde_bf_STFT},
                # "X_tilde": {"ref_mic_idx": self.ref_mic_idx, "STFT": self.STFT, "est_target_multi_channel": X_tilde, "est_target_STFT_multi_channel":X_tilde_STFT},
            # },
        # }

        # return output

    def precompute_inv_phi_Y(self, Y) -> torch.Tensor:
        phi_Y = torch_spatial_covariance(Y)  # N, 2, F, T, C, C
        phi_Y = torch.mean(phi_Y, dim=3, keepdim=True)  # N, 2, F, 1, C, C
        inv_phi_Y = diag_load_and_inverse(phi_Y)  # N, 2, F, 1, C, C
        return inv_phi_Y  # N, 2, F, 1, C, C

    def update_beamformer_weight(self, t: int, X_t, init_phi_X, inv_phi_Y_t):
        # phi_Y_t = torch_spatial_covariance(Y_t)

        if t == 0:

            self.phi_X = init_phi_X

        if t > 0:
            phi_X_t = torch_spatial_covariance(X_t)
            self.phi_X = phi_X_t

        if isinstance(self.phi_X, float):
            C = self.n_mics
            N = X_t.shape[0]
            F = X_t.shape[-1]
            device = X_t.device
            W = self.init_weights.repeat(N, 2, F, 1, 1, 1)

        else:
            W = self.compute_beamformer_weights(self.phi_X, inv_phi_Y_t)

        return W

    @torch.jit.export
    def forward(
        self, input: Dict[str, torch.Tensor], test: bool = False
    ) -> Dict[str, torch.Tensor]:

        if self.training:
            return self.forward_train(input)
        else:
            return self.forward_test(input)

    # def training_epoch_end(self, pl_module):
    #     # print("hello")
    #     # pl_module.logger.experiment.add_histogram(
    #     #     "filter_selector_y",
    #     #     self.filter_selector_y.weight.squeeze(),
    #     #     global_step=pl_module.current_epoch,
    #     # )
    #     # pl_module.logger.experiment.add_histogram(
    #     #     "filter_selector_xbf",
    #     #     self.filter_selector_bf.weight.squeeze(),
    #     #     global_step=pl_module.current_epoch,
    #     # )


if __name__ == "__main__":

    ref_mic_idx = 0
    n_mics = 5
    ref_mic_idx: 0  # the index to the reference microphone
    batch_size = 16
    stft_module = STFT_module(n_fft=1024, window_func=torch.hann_window, hop_length=1024 // 4)

    model = MaskBasedNBFwSelfAttBasedTrackingFirstStage(ref_mic_idx=1, n_mics=n_mics, STFT_m=stft_module)
    batch_input = {}
    batch_input["mixture"] =  torch.rand(batch_size, n_mics, 16000) # 1s of 16kHz
    batch_input["target"] = torch.rand(batch_size, n_mics, 16000)
    batch_input["noise"] = torch.rand(batch_size, n_mics, 16000)
    
    # example input
    # print(f"Model size is : {sum(p.numel() for p in model.parameters())/1000000:.2f} M parameters")

    # output = model.forward(batch_input)
    # print(output["est_target"].shape)
    

    ref_mic_idx = 0
    n_mics = 5 # following their papers
    ref_mic_idx: 0  # the index to the reference microphone
    batch_size = 16
    stft_module = STFT_module(n_fft=1024, window_func=torch.hann_window, hop_length=1024 // 4)

    model = MaskBasedNBFwSelfAttBasedTracking(ref_mic_idx=ref_mic_idx, STFT=stft_module, n_mics=n_mics)

    # example input
    batch_input = {}

    batch_input["mixture"] =  torch.rand(batch_size, n_mics, 16000)
    batch_input["target"] = torch.rand(batch_size, n_mics, 16000)
    batch_input["noise"] = torch.rand(batch_size, n_mics, 16000)
    output = model.forward(batch_input)
    print(output["est_target"].shape)
    print(f"Model size is : {sum(p.numel() for p in model.parameters())/1000000:.2f} M parameters")
