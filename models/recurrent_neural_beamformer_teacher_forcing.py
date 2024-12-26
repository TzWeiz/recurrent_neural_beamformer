import math
import torch
import torch.nn as nn
from torch_complex_ops import (
    complex_inverse,
    complex_mm,
    conj,
)

from common_typing import *
from stft_module import STFT_module
from beamformers.ops import (
    torch_spatial_covariance,
)

import logging
logging.basicConfig()
# from models.du_net import (
#     Conv_BN_ReLU_2dBlock,
#     Conv_BN_ReLU_1dBlock,
#     Concat_Conv_BN_ReLU_1dBlock,
#     Conv_freq_norm_ReLU_1dBlock,
# )
from models.conv_blocks.conv_bn_relu_1d_block import ConvBnRelu1dBlock

def diag_load_and_inverse(SCM, diag_loadings=0.0001):

    # N, _, F, T, C, C = SCM.shape
    C, C = SCM.shape[-2:]
    eye = torch.eye(C, device=SCM.device)
    # if torch.cuda.is_available():
    # eye = eye.to(SCM.device)
    SCM = SCM + diag_loadings * eye  # diagonal loading
    inv_SCM = complex_inverse(SCM, RI_dim=1)

    return inv_SCM


def apply_beamformer(W, Y):
    # # W: 
    WH = conj(W.transpose(-1, -2), RI_dim=1)  # N, 1, F, 1, C, C

    # for t in range(T)
    # Y: N, 2, C, T, F
    Y_bf_in = Y.transpose(2, -1)  # N, 2, F, T, C
    Y_bf_in = Y_bf_in.unsqueeze(-1)  # N, 2, F, T, C, 1
    X_bf = complex_mm(WH, Y_bf_in, RI_dim=1)  # N, 2, F, 1, C, 1
    X_bf = X_bf.squeeze(-1)  # N, 2, F, T, C
    X_bf = X_bf.transpose(2, 4)  # N, 2, C, T, F

    return X_bf

def apply_beamformer_1d(W, Y):
    # # W: 
    WH = conj(W.transpose(-1, -2), RI_dim=1)  # N, 2, F, C, C

    # for t in range(T)
    # Y: N, 2, C, F
    Y_bf_in = Y.transpose(-2, -1)  # N, 2, F, C
    Y_bf_in = Y_bf_in.unsqueeze(-1)  # N, 2, F, C, 1
    X_bf = complex_mm(WH, Y_bf_in, RI_dim=1)  # N, 2, F, C, 1
    X_bf = X_bf.squeeze(-1)  # N, 2, F, C
    X_bf = X_bf.transpose(-2, -1)  # N, 2, C, F

    return X_bf


class RecurrentNeuralBeamformerTeacherForcing(
    torch.nn.Module
):  # jit.ScriptModule):
    def __init__(
        self,
        ref_mic_idx: int,
        STFT: STFT_module,
        conv_block: torch.nn.Module,
        compute_beamformer_weights: torch.nn.Module,
        n_mics: int = 1,
        n_kernels: int = 64,
        kernel_size: int = 3,
        n_layers: int = 10,
        selector_rate: int=10,
        norm="batch",
        **kwargs,
    ):
        """
        The proposed AR-MCWF SMoLnet

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

        self.ref_mic_idx = ref_mic_idx
        # self.alpha_noisy = alpha_noisy

        init_weights = (
            torch.eye(self.n_mics).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        ) / self.n_mics
        self.register_buffer("init_weights", init_weights)
        torch.cuda.empty_cache()

        self.build_tdu_net(
            conv_block=conv_block,
            in_channels=n_mics * 2, out_channels=n_mics * 2, 
            selector_rate=selector_rate,
            n_mics=n_mics,
        )

        self.compute_beamformer_weights=compute_beamformer_weights

    def build_tdu_net(
        self,
        conv_block: torch.nn.Module,
        in_channels:int,
        out_channels:int,
        selector_rate, #K
        kernel_size=3,
        n_dilated_layers=10,
        memory_efficient=False,
        n_mics=8,
    ):

        
        self.n_dilated_layers = n_dilated_layers

        self.received_signal_encoder = torch.nn.Sequential()
        self.beamformed_signal_encoder  = torch.nn.Sequential()

        self.received_signal_decoder = torch.nn.ModuleList()
        self.beamformed_signal_decoder = torch.nn.ModuleList()


        _in_channels = in_channels
        n_kernels = n_mics * selector_rate

        logging.info("Building TDU-net...")
        logging.info("For the encoders:")
        for i in range(n_dilated_layers):
            d = 2 ** i
            # to make it work for non-kernel 3
            self.received_signal_encoder.add_module(
                f"Conv:{i}",
                conv_block(
                    _in_channels,
                    # n_kernels,
                    n_kernels,
                    kernel_size,
                    padding=d,
                    dilation=d,
                ),
            )
            self.beamformed_signal_encoder.add_module(
                f"Conv:{i}",
                conv_block(
                    _in_channels,
                    # n_kernels,
                    n_kernels,
                    kernel_size,
                    padding=d,
                    dilation=d,
                ),
            )
            logging.info(f"Conv:{i} has dilation rate of {d} and has {_in_channels} input and {n_kernels} output kernels, respectively.")
            _in_channels = n_kernels
        
        logging.info("For the decoders:")
        for i in range(n_dilated_layers - 1, -1, -1):
            d = 2 ** i

            if i !=9: 
                _in_channels = 2 * n_kernels

            self.received_signal_decoder.add_module(
                f"ConcatConv:{i}",
                conv_block(
                    in_channels=_in_channels,
                    out_channels=n_kernels,
                    kernel_size=kernel_size,
                    padding=d,
                    dilation=d,
                ),
            )
            
            self.beamformed_signal_decoder.add_module(
                f"ConcatConv:{i}",
                conv_block(
                    in_channels=_in_channels,
                    out_channels=n_kernels,
                    kernel_size=kernel_size,
                    padding=d,
                    dilation=d,
                ),
            )
            logging.info(f"Conv:{i} has dilation rate of {d} and has {_in_channels} input and {n_kernels} output kernels, respectively.")

        self.build_feature_combinator(conv_block, selector_rate, n_mics=n_mics)
    
        


    def build_feature_combinator(self, conv_block, selector_rate:int, n_mics:int):
        ##
        n_kernels = selector_rate *2 *n_mics
        self.feature_combinator = torch.nn.Sequential()
        n_layers=selector_rate 
        in_channels = n_kernels
        for k in range(n_layers, 0):
            
            out_channels = k * (n_mics*2)
            self.feature_combinator.add_module(
                f"feat_combi_{n_layers-k}",
                    conv_block
            )
            in_channels = out_channels
        
        out_channels = n_mics*2
        self.feature_combinator.add_module(
            f"feat_combi_{n_layers}",
            torch.nn.Conv1d(in_channels, out_channels, 1)
        )


    def feature_combine(self, x_bf:torch.Tensor, x_y:torch.Tensor):
        ## x_bf: N, C, F
        ## x_y: N, C, F
        
        x_comb = torch.cat([x_bf, x_y], dim=1)   # N, 2C, F
        output = self.feature_combinator.forward(x_comb)
        
        return output
        
            
    def forward_train(
        self, input: Dict[str, torch.Tensor], init_phi_X=0.0, init_phi_Y=0.0
    ):

        Y = input["mixture"]
        Y_STFT = self.STFT.forward(Y)  # shape of N, C, 2, T, F
        Y = Y_STFT.transpose(1, 2)  # N, 2, C, T, F

        N, _, C, T, F = Y.shape
        device = Y.device
        # X_nn = []

        inv_phi_Y = self.precompute_inv_phi_Y(Y)  # N, 2,

        # noisy_branch = []
        # bf_branch = []
        Y_bf = torch.zeros_like(Y)
        X_nn = torch.zeros_like(Y)
        X_tilde_bf = torch.zeros_like(Y)
        X_tilde = torch.zeros_like(Y)

        if self.training:
            X = input["target"]
            X = self.STFT.forward(X)
            X = X.transpose(1, 2) # N, 2, C, T, F

            
        X_t = torch.zeros_like(Y[:, :, :, 0])
            

        # X_nn = []
        X_nn_t = torch.zeros_like(Y[:, :, :, 0])
        for t in range(T):
            # print(t, T)
            Y_t = Y[:, :, :, t]  # N, 2, C, 1, F
            X_t = X_t.unsqueeze(3)
            W = self.update_beamformer_weight(t, X_t, init_phi_X, inv_phi_Y)
            W = W.squeeze(3) # N, 2, F, C, C
            Y_bf_t = apply_beamformer_1d(W, Y_t)  # N, 2, C, 1, F
            # del W
            Y_bf[:, :, :, t] = Y_bf_t

            # X_tilde_t = self.apply_freq_spatial_network(Y_t)
            # X_tilde_bf_t = self.apply_freq_spatial_network(Y_bf_t)
            # X_tilde_bf[:, :, :, t] = X_tilde_bf_t
            # X_tilde[:, :, :, t] = X_tilde_t
            # del Y_bf
            # del Y_t
            # assert torch.allclose(noisy_branch + bf_branch, X_nn_t
            # )
            feature_maps ={}
            _x = Y_t
            _x = torch.cat([_x[:,0], _x[:,1]], dim=1)
            for l, layer in enumerate(self.received_signal_encoder):
                _x = layer.forward(_x)
                feature_maps.update({l:_x})
            
            L = self.n_dilated_layers
            for l, layer in enumerate(self.received_signal_decoder):
                if l == 0:
                    _x = layer.forward(_x)
                else:
                    _x = torch.cat([_x, feature_maps[L-l-1]], 1)
                    _x = layer.forward(_x)

            x_bf = _x

            feature_maps ={}
            _x = Y_bf_t
            _x = torch.cat([_x[:,0], _x[:,1]], dim=1)
            for l, layer in enumerate(self.beamformed_signal_encoder):
                _x = layer.forward(_x)
                feature_maps.update({l:_x})
            
            L = self.n_dilated_layers
            for l, layer in enumerate(self.beamformed_signal_decoder):
                if l == 0:
                    _x = layer.forward(_x)
                else:
                    _x = torch.cat([_x, feature_maps[L-l-1]], 1)
                    _x = layer.forward(_x)


            x_y = _x


            X_nn_t = self.feature_combine(x_bf, x_y)
            
            X_nn_t = X_nn_t.reshape(N, 2, -1, F)
            # X_nn_t = X_tilde_bf_t + X_tilde_t
            X_nn[:, :, :, t] = X_nn_t

            if self.training:
                X_t = X[:, :, :, t] # N, 2, C, 1, F
            else:
                X_t = X_nn_t
            # X_nn.append(X_nn_t.to("cpu"))
            # noisy_branch.append(noisy_branch_t)
            # bf_branch.append(bf_branch_t)

        # X_nn = torch.cat(X_nn, dim=3)  # N, 2, C, T, F
        # X_nn = X_nn.to(device)
        # noisy_branch = torch.cat(noisy_branch, dim=3)  # N, 64, T, F
        # bf_branch = torch.cat(bf_branch, dim=3)  # N, 64, T, F
        X_nn = X_nn.transpose(1, 2)  # N, C, 2, T, F
        est_target_STFT_multi_channel = X_nn
        est_target = self.STFT.backward(X_nn)

        with torch.no_grad():
            Y_bf_STFT = Y_bf.transpose(1, 2)  # N, C, 2, T, F
            Y_bf = self.STFT.backward(Y_bf_STFT)

            X_tilde_bf_STFT = X_tilde_bf.transpose(1, 2)
            X_tilde_bf = self.STFT.backward(X_tilde_bf_STFT)  # N, C, 2, T, F

            X_tilde_STFT = X_tilde.transpose(1, 2)
            X_tilde = self.STFT.backward(X_tilde_STFT)  # N, C, 2, T, F
        # legacy code
        if hasattr(self, "ref_channel"):
            self.ref_mic_idx = self.ref_channel

        output = {
            "STFT": self.STFT,
            "est_target_multi_channel": est_target,
            "est_target": est_target[:, self.ref_mic_idx],
            "est_target_STFT_multi_channel": est_target_STFT_multi_channel,
            "est_target_STFT": est_target_STFT_multi_channel[:, self.ref_mic_idx],
            "ref_mic_idx": self.ref_mic_idx,
            "logging": {
                "Y_bf": {
                    "ref_mic_idx": self.ref_mic_idx, 
                    "STFT": self.STFT, 
                    "est_target_multi_channel": Y_bf, 
                    "est_target_STFT_multi_channel":Y_bf_STFT,
                    "est_target": Y_bf[:, self.ref_mic_idx],
                    "est_target_STFT": Y_bf_STFT[:, self.ref_mic_idx],
                    },
                "Y": {
                    "ref_mic_idx": self.ref_mic_idx, 
                    "STFT": self.STFT, 
                    "est_target_multi_channel": input["mixture"], 
                    "est_target_STFT_multi_channel":Y_STFT,
                    "est_target": input["mixture"][:, self.ref_mic_idx],
                    "est_target_STFT": Y_STFT[:, self.ref_mic_idx],
                    },
                # "X_tilde_bf": {"ref_mic_idx": self.ref_mic_idx, "STFT": self.STFT, "est_target_multi_channel": X_tilde_bf, "est_target_STFT_multi_channel":X_tilde_bf_STFT},
                # "X_tilde": {"ref_mic_idx": self.ref_mic_idx, "STFT": self.STFT, "est_target_multi_channel": X_tilde, "est_target_STFT_multi_channel":X_tilde_STFT},
            },

        }
        return output

    def apply_freq_spatial_network(self, Y_t:torch.Tensor):

        N, _2, M, F = Y_t.shape
        # Y_bf = Y_bf.reshape(N, 2 * M, F)
        Y_t = Y_t.reshape(N, 2 * M, F)
        # Y_bf = torch.cat([Y_bf, Y_t], dim=2)
        # x = torch.stack([Y_bf, Y_t], dim=1)  # N, 2, 2*M, 1, F
        # x = x.reshape(N * 2, 2 * M, 1, F)

        x = Y_t
        device = x.device
        out = []
        for i, layer in enumerate(self.frequency_feature_extractor_y):
            x = layer(x)
            if i < self.n_dilated_layers - 1:
                out.append(x)  # .to("cpu"))

        for i, layer in enumerate(self.frequency_feature_deconv_extractor_y):

            if i == 0:
                x = layer([x])
            else:
                x = layer([out.pop(), x])


        return x

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
