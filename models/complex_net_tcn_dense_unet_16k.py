#!/usr/bin/env python

#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from beamformers.TI_MVDR import TI_MVDR
from beamformers.TV_MVDR import TV_MVDR

import os


import sys
from os.path import dirname

sys.path.append("../")

from .conv_block_for_tcn_dense_unet import (
    Conv2dBNReLUBlock,
    DenseBlockNoOutCat,
    TCNDepthWise,
    DenseBlockNoOutCatFM,
)

from typing import *

from stft_module import STFT_module


class CSeqUNetDense(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        input_dim,
        nlayer=2,
        n_units=512,
        target_dim=257,
        use_seqmodel=1,
        masking_or_mapping=1,
        output_activation="linear",
        rmax=5,
        approx_method="MSA-RIx2",
        loss_function="l1loss",
        use_batchnorm=3,
        use_convbnrelu=2,
        use_act=2,
        memory_efficient=1,
        n_outputs=2,
    ):
        """

        In the CHiME-4 paper, the arguments are:
        input_dim=x*257
        nlayer=2
        n_units=512
        target_dim=257
        use_seqmodel=1
        masking_or_mapping=1
        output_activation='linear'
        rmax=5
        approx_method='MSA-RIx2'
        loss_function='l1loss'
        use_batchnorm=3
        use_convbnrelu=2
        use_act=2
        memory_efficient=1


        approx_method : str, optional
            The approximation method for setting the target domain. Supports "MSA" (magnitude spectrum approximation) and "RIx2" (real-imaginary composites). Supports multiple approximations, seperated by '-'. By default "MSA-RIx2"
        loss_function : str, optional
            The loss function for evaluating the output layer and the target via the approx_method. Supports "l1loss" and "l2loss", by default "l1loss"
        """

        super(CSeqUNetDense, self).__init__()
        approx_method = approx_method.split("-")
        self.approx_method = approx_method

        if loss_function not in ["l1loss", "l2loss"]:
            raise
        self.loss_function = loss_function

        if target_dim not in [257]:
            raise

        if input_dim % target_dim != 0:
            raise
        in_channels = input_dim // target_dim
        assert in_channels >= 1

        t_ksize = 3
        #
        #              257,   1                                   2/4
        # (257-5)/2+1 = 127,  32 ---5*32--> 32  +  32  ---5*32---> 32
        # (127-3)/2+1 =  63,  32 ---5*32--> 32  +  32  ---5*32---> 32
        # (63-3)/2+1  =  31,  32 ---5*32--> 32  +  32  ---5*32---> 32
        # (31-3)/2+1  =  15,  64 ---5*64--> 64  +  64  ---5*64---> 64
        # (15-3)/2+1  =   7, 128                +  128
        # (7-3)/2+1   =   3, 256                +  256
        # (3-3)/1+1   =   1, 512                +  512
        #
        self.conv0 = nn.Conv2d(
            in_channels, 32, (t_ksize, 5), stride=(1, 2), padding=(t_ksize // 2, 0)
        )
        self.eden0 = DenseBlockNoOutCatFM(
            32,
            32,
            (t_ksize, 3),
            127,
            n_layers=5,
            use_batchnorm=use_batchnorm,
            use_act=use_act,
            use_convbnrelu=use_convbnrelu,
            memory_efficient=memory_efficient,
        )
        self.conv1 = Conv2dBNReLUBlock(
            32,
            32,
            (t_ksize, 3),
            stride=(1, 2),
            padding=(t_ksize // 2, 0),
            use_batchnorm=use_batchnorm,
            use_act=use_act,
            use_convbnrelu=use_convbnrelu,
            memory_efficient=memory_efficient,
        )
        self.eden1 = DenseBlockNoOutCatFM(
            32,
            32,
            (t_ksize, 3),
            63,
            n_layers=5,
            use_batchnorm=use_batchnorm,
            use_act=use_act,
            use_convbnrelu=use_convbnrelu,
            memory_efficient=memory_efficient,
        )
        self.conv2 = Conv2dBNReLUBlock(
            32,
            32,
            (t_ksize, 3),
            stride=(1, 2),
            padding=(t_ksize // 2, 0),
            use_batchnorm=use_batchnorm,
            use_act=use_act,
            use_convbnrelu=use_convbnrelu,
            memory_efficient=memory_efficient,
        )
        self.eden2 = DenseBlockNoOutCatFM(
            32,
            32,
            (t_ksize, 3),
            31,
            n_layers=5,
            use_batchnorm=use_batchnorm,
            use_act=use_act,
            use_convbnrelu=use_convbnrelu,
            memory_efficient=memory_efficient,
        )
        self.conv3 = Conv2dBNReLUBlock(
            32,
            64,
            (t_ksize, 3),
            stride=(1, 2),
            padding=(t_ksize // 2, 0),
            use_batchnorm=use_batchnorm,
            use_act=use_act,
            use_convbnrelu=use_convbnrelu,
            memory_efficient=memory_efficient,
        )
        self.eden3 = DenseBlockNoOutCatFM(
            64,
            64,
            (t_ksize, 3),
            15,
            n_layers=5,
            use_batchnorm=use_batchnorm,
            use_act=use_act,
            use_convbnrelu=use_convbnrelu,
            memory_efficient=memory_efficient,
        )
        self.conv4 = Conv2dBNReLUBlock(
            64,
            128,
            (t_ksize, 3),
            stride=(1, 2),
            padding=(t_ksize // 2, 0),
            use_batchnorm=use_batchnorm,
            use_act=use_act,
            use_convbnrelu=use_convbnrelu,
            memory_efficient=memory_efficient,
        )
        self.conv5 = Conv2dBNReLUBlock(
            128,
            256,
            (t_ksize, 3),
            stride=(1, 2),
            padding=(t_ksize // 2, 0),
            use_batchnorm=use_batchnorm,
            use_act=use_act,
            use_convbnrelu=use_convbnrelu,
            memory_efficient=memory_efficient,
        )
        self.conv6 = Conv2dBNReLUBlock(
            256,
            512,
            (t_ksize, 3),
            stride=(1, 1),
            padding=(t_ksize // 2, 0),
            use_batchnorm=use_batchnorm,
            use_act=use_act,
            use_convbnrelu=use_convbnrelu,
            memory_efficient=memory_efficient,
        )
        encoder_dim = 512
        input_dim = encoder_dim

        if use_seqmodel == 0:
            for i in range(nlayer):
                input_dim = input_dim if i == 0 else n_units * 2
                self.add_module(
                    "bilstm-%d" % i,
                    nn.LSTM(
                        input_dim,
                        n_units,
                        1,
                        batch_first=True,
                        dropout=0.0,
                        bidirectional=True,
                    ),
                )
                # Setting forget gate bias to a 2.0
                self["bilstm-%d" % i].bias_hh_l0.data[n_units : 2 * n_units] = 1.0
                self["bilstm-%d" % i].bias_ih_l0.data[n_units : 2 * n_units] = 1.0
                self["bilstm-%d" % i].bias_hh_l0_reverse.data[
                    n_units : 2 * n_units
                ] = 1.0
                self["bilstm-%d" % i].bias_ih_l0_reverse.data[
                    n_units : 2 * n_units
                ] = 1.0
        else:
            tcn_classname = TCNDepthWise
            for ii in range(1, nlayer + 1):
                self.add_module(
                    "tcn-conv%d-0" % ii,
                    tcn_classname(
                        input_dim,
                        input_dim,
                        t_ksize,
                        use_batchnorm=use_batchnorm,
                        use_act=use_act,
                        dilation=1,
                    ),
                )
                self.add_module(
                    "tcn-conv%d-1" % ii,
                    tcn_classname(
                        input_dim,
                        input_dim,
                        t_ksize,
                        use_batchnorm=use_batchnorm,
                        use_act=use_act,
                        dilation=2,
                    ),
                )
                self.add_module(
                    "tcn-conv%d-2" % ii,
                    tcn_classname(
                        input_dim,
                        input_dim,
                        t_ksize,
                        use_batchnorm=use_batchnorm,
                        use_act=use_act,
                        dilation=4,
                    ),
                )
                self.add_module(
                    "tcn-conv%d-3" % ii,
                    tcn_classname(
                        input_dim,
                        input_dim,
                        t_ksize,
                        use_batchnorm=use_batchnorm,
                        use_act=use_act,
                        dilation=8,
                    ),
                )
                self.add_module(
                    "tcn-conv%d-4" % ii,
                    tcn_classname(
                        input_dim,
                        input_dim,
                        t_ksize,
                        use_batchnorm=use_batchnorm,
                        use_act=use_act,
                        dilation=16,
                    ),
                )
                self.add_module(
                    "tcn-conv%d-5" % ii,
                    tcn_classname(
                        input_dim,
                        input_dim,
                        t_ksize,
                        use_batchnorm=use_batchnorm,
                        use_act=use_act,
                        dilation=32,
                    ),
                )

        self.output_activation = output_activation
        self.rmax = rmax
        if masking_or_mapping == 0:
            # masking
            initial_bias = 0.0
        else:
            # mapping
            initial_bias = 0.0

        #
        #              257,   1                                   2/4
        # (257-5)/2+1 = 127,  32 ---5*32--> 32  +  32  ---5*32---> 32
        # (127-3)/2+1 =  63,  32 ---5*32--> 32  +  32  ---5*32---> 32
        # (63-3)/2+1  =  31,  32 ---5*32--> 32  +  32  ---5*32---> 32
        # (31-3)/2+1  =  15,  64 ---5*64--> 64  +  64  ---5*64---> 64
        # (15-3)/2+1  =   7, 128                +  128
        # (7-3)/2+1   =   3, 256                +  256
        # (3-3)/1+1   =   1, 512                +  512
        #
        self.deconv0 = Conv2dBNReLUBlock(
            encoder_dim + input_dim,
            256,
            (t_ksize, 3),
            stride=(1, 1),
            padding=(t_ksize // 2, 0),
            use_deconv=1,
            use_batchnorm=use_batchnorm,
            use_act=use_act,
            use_convbnrelu=use_convbnrelu,
            memory_efficient=memory_efficient,
        )
        self.deconv1 = Conv2dBNReLUBlock(
            2 * 256,
            128,
            (t_ksize, 3),
            stride=(1, 2),
            padding=(t_ksize // 2, 0),
            use_deconv=1,
            use_batchnorm=use_batchnorm,
            use_act=use_act,
            use_convbnrelu=use_convbnrelu,
            memory_efficient=memory_efficient,
        )
        self.deconv2 = Conv2dBNReLUBlock(
            2 * 128,
            64,
            (t_ksize, 3),
            stride=(1, 2),
            padding=(t_ksize // 2, 0),
            use_deconv=1,
            use_batchnorm=use_batchnorm,
            use_act=use_act,
            use_convbnrelu=use_convbnrelu,
            memory_efficient=memory_efficient,
        )
        self.dden2 = DenseBlockNoOutCatFM(
            64 + 64,
            64,
            (t_ksize, 3),
            15,
            n_layers=5,
            use_batchnorm=use_batchnorm,
            use_act=use_act,
            use_convbnrelu=use_convbnrelu,
            memory_efficient=memory_efficient,
        )
        self.deconv3 = Conv2dBNReLUBlock(
            64,
            32,
            (t_ksize, 3),
            stride=(1, 2),
            padding=(t_ksize // 2, 0),
            use_deconv=1,
            use_batchnorm=use_batchnorm,
            use_act=use_act,
            use_convbnrelu=use_convbnrelu,
            memory_efficient=memory_efficient,
        )
        self.dden3 = DenseBlockNoOutCatFM(
            32 + 32,
            32,
            (t_ksize, 3),
            31,
            n_layers=5,
            use_batchnorm=use_batchnorm,
            use_act=use_act,
            use_convbnrelu=use_convbnrelu,
            memory_efficient=memory_efficient,
        )
        self.deconv4 = Conv2dBNReLUBlock(
            32,
            32,
            (t_ksize, 3),
            stride=(1, 2),
            padding=(t_ksize // 2, 0),
            use_deconv=1,
            use_batchnorm=use_batchnorm,
            use_act=use_act,
            use_convbnrelu=use_convbnrelu,
            memory_efficient=memory_efficient,
        )
        self.dden4 = DenseBlockNoOutCatFM(
            32 + 32,
            32,
            (t_ksize, 3),
            63,
            n_layers=5,
            use_batchnorm=use_batchnorm,
            use_act=use_act,
            use_convbnrelu=use_convbnrelu,
            memory_efficient=memory_efficient,
        )
        self.deconv5 = Conv2dBNReLUBlock(
            32,
            32,
            (t_ksize, 3),
            stride=(1, 2),
            padding=(t_ksize // 2, 0),
            use_deconv=1,
            use_batchnorm=use_batchnorm,
            use_act=use_act,
            use_convbnrelu=use_convbnrelu,
            memory_efficient=memory_efficient,
        )
        self.dden5 = DenseBlockNoOutCatFM(
            32 + 32,
            32,
            (t_ksize, 3),
            127,
            n_layers=5,
            use_batchnorm=use_batchnorm,
            use_act=use_act,
            use_convbnrelu=use_convbnrelu,
            memory_efficient=memory_efficient,
        )
        self.deconv6 = nn.ConvTranspose2d(
            32, n_outputs, (t_ksize, 5), stride=(1, 2), padding=(t_ksize // 2, 0)
        )
        self.deconv6.bias.data[:] = initial_bias

        self.target_dim = target_dim
        self.nlayer = nlayer
        self.n_units = n_units
        self.use_seqmodel = use_seqmodel
        self.masking_or_mapping = masking_or_mapping
        self.n_outputs = n_outputs

    def forward(
        self,
        x_ins,
        device,
        input_dropout_rate=0.0,
        hidden_dropout_rate=0.0,
        dilation_dropout_rate=0.0,
    ):
        """
        x_ins: list of 2D tensors
        """

        # set all sequence to equal length
        batchsize = len(x_ins)
        ilenvec = [x_in.shape[0] for x_in in x_ins]
        N = max(ilenvec)

        batch = np.concatenate(x_ins, axis=0)
        batch = torch.from_numpy(batch)
        batch = torch.split(batch, ilenvec, dim=0)
        batch = pad_sequence(
            batch, batch_first=True, padding_value=0
        )  # [batchsize, N, -1]

        batch = batch.to(device)

        if input_dropout_rate > 0.0:
            batch = F.dropout(batch, p=input_dropout_rate, training=True, inplace=False)

        batch = batch.view(
            [batchsize, N, -1, self.target_dim]
        )  # [batchsize, N, -1, n_freqs]
        batch = batch.transpose(1, 2)  # [batchsize, -1, N, n_freqs]

        batch = self.apply_neural_network(batch)

        batch = batch.transpose(1, 2)  # [batchsize, N, -1, self.target_dim]
        batch = batch.reshape(
            [batchsize, N, -1]
        )  # [batchsize, N, num_speakers*n_outputs*self.target_dim]
        batch = [batch[bb, :utt_len] for bb, utt_len in enumerate(ilenvec)]
        batch = torch.cat(batch, dim=0)  # [-1,n_outputs*target_dim]

        self.activations = batch  # [n_frames,n_outputs*target_dim]

    def apply_neural_network(
        self,
        batch: torch.Tensor,
        input_dropout_rate=0.0,
        hidden_dropout_rate=0.0,
        dilation_dropout_rate=0.0,
    ):
        """Applies the neural network.


        Parameters
        ----------
        batch : torch.Tensor
            A batch of input samples with shape (batchsize, -1, n_frames, n_freqs_bins)
        """
        if input_dropout_rate > 0.0:
            batch = F.dropout(batch, p=input_dropout_rate, training=True, inplace=False)

        batchsize, _, N, n_freqs = batch.shape

        all_conv_batch = []
        for cc in range(10):
            conv_link_name = "conv%d" % cc
            if hasattr(self, conv_link_name):
                batch = self[conv_link_name](batch)
                eden_link_name = "eden%d" % cc
                if hasattr(self, eden_link_name):
                    batch = self[eden_link_name](batch)
                all_conv_batch.append(batch)
            else:
                break
        # batch.shape is [batchsize, self.n_units, N, 1]

        if self.use_seqmodel == 0:
            batch = batch.squeeze(dim=-1)  # [batchsize, self.n_units, N]
            batch = batch.transpose(1, 2)  # [batchsize, N, self.n_units]
            batch = self.propagate_full_sequence(
                batch, dropout_rate=hidden_dropout_rate
            )  # [batchsize, N, 2*self.n_units]
            batch = batch.transpose(1, 2)  # [batchsize, 2*self.n_units, N]
            batch = batch.unsqueeze(dim=-1)  # [batchsize, 2*self.n_units, N, 1]
        else:
            batch = batch.view(
                [batchsize, self.n_units, N]
            )  # [batchsize, self.n_units, N]
            for ii in range(1, self.nlayer + 1):
                for cc in range(20):
                    conv_link_name = "tcn-conv%d-%d" % (ii, cc)
                    if hasattr(self, conv_link_name):
                        batch = self[conv_link_name](
                            batch,
                            hidden_dropout_rate=hidden_dropout_rate,
                            dilation_dropout_rate=dilation_dropout_rate,
                        )
                    else:
                        break
            batch = batch.unsqueeze(dim=-1)  # [batchsize, self.n_units, N, 1]

        for cc in range(10):
            deconv_link_name = "deconv%d" % cc
            if hasattr(self, deconv_link_name):
                if cc - 1 >= 0 and hasattr(self, "dden%d" % (cc - 1)):
                    batch = self[deconv_link_name](batch)
                else:
                    batch = self[deconv_link_name](
                        torch.cat([batch, all_conv_batch[-1 - cc]], dim=1)
                    )
                dden_link_name = "dden%d" % cc
                if hasattr(self, dden_link_name):
                    batch = self[dden_link_name](
                        torch.cat([batch, all_conv_batch[-1 - cc - 1]], dim=1)
                    )
            else:
                break
        # batch.shape is [batchsize, -1, N, self.target_dim]

        if self.masking_or_mapping == 0:
            # masking
            if self.pitactivation == "linear":
                batch = torch.clamp(batch, -self.rmax, self.rmax)
            else:
                raise
        else:
            # mapping
            if self.output_activation == "linear":
                batch = batch
            else:
                raise

        return batch

    def get_loss(self, ins, device):

        if self.loss_function.startswith("l2"):
            raise
        else:
            loss_type = torch.abs

        y_reals = np.concatenate(ins[0][1], axis=0)
        y_imags = np.concatenate(ins[0][2], axis=0)
        y_reals = torch.from_numpy(y_reals).to(device)
        y_imags = torch.from_numpy(y_imags).to(device)

        activations_reals, activations_imags = torch.chunk(
            self.activations, self.n_outputs, dim=-1
        )

        if self.masking_or_mapping == 0:
            x_reals = np.concatenate(ins[1][1], axis=0)
            x_imags = np.concatenate(ins[1][2], axis=0)
            x_reals = torch.from_numpy(x_reals).to(device)
            x_imags = torch.from_numpy(x_imags).to(device)
            # (a+b*i)*(c+d*i) = ac-bd + (ad+bc)*i
            activations_reals, activations_imags = (
                x_reals * activations_reals - x_imags * activations_imags,
                x_reals * activations_imags + x_imags * activations_reals,
            )
        else:
            pass

        est_y_reals, est_y_imags = activations_reals, activations_imags

        ret = [torch.tensor(0.0, device=device)]

        if "MSA" in self.approx_method:
            y_mags = torch.sqrt(y_reals ** 2 + y_imags ** 2 + 1e-5)
            est_y_mags = torch.sqrt(est_y_reals ** 2 + est_y_imags ** 2 + 1e-5)
            loss_mags = torch.mean(loss_type(est_y_mags - y_mags))
            ret[0] += loss_mags
            ret.append(loss_mags)

        if "RIx2" in self.approx_method:
            loss_reals = torch.mean(loss_type(y_reals - est_y_reals))
            loss_imags = torch.mean(loss_type(y_imags - est_y_imags))
            ret[0] += loss_reals + loss_imags
            ret.append(loss_reals)
            ret.append(loss_imags)

        return ret

    def propagate_one_layer(self, batch, layer, dropout_rate=0.0):
        batch, (_, _) = self["bilstm-%d" % layer](batch)
        return (
            F.dropout(batch, p=dropout_rate, training=True, inplace=False)
            if dropout_rate > 0.0
            else batch
        )

    def propagate_full_sequence(self, batch, dropout_rate=0.0):
        for ll in range(self.nlayer - 1):
            batch = self.propagate_one_layer(batch, ll, dropout_rate=dropout_rate)
        batch = self.propagate_one_layer(batch, self.nlayer - 1, dropout_rate=0.0)
        return batch


class CSeqUNetDense_Singlechannel(
    CSeqUNetDense,
):
    def __init__(
        self,
        ref_mic_idx: int = 0,
        n_srcs: int = 1,
        nlayer: int = 2,
        n_units: int = 512,
        use_seqmodel: int = 1,
        masking_or_mapping: int = 1,
        output_activation: str = "linear",
        rmax: int = 5,
        use_batchnorm: int = 3,
        use_convbnrelu: int = 2,
        use_act: int = 2,
        memory_efficient: int = 1,
        n_out_srcs: int = 1,
        **kwargs
    ):
        """
        Interface of Zhang-Qiu Wang's model from [1] with our system. The default parameters are for [1].

        The microphone channels are stacked on batch dimension.

        Parameters
        ----------
        ref_mic_idx: int
            The reference microphone index, by default 0.
        n_srcs : int
            The number of sources. By default 1.
        nlayer : int, optional
            The number of layers in the sequence model. +1 for temporal convolution network, by default 2
        n_units: int, optional
            The number of units in the sequence model. By default 512.
        use_seqmodel : int, optional
            Set to 0 for bi-directional LSTM else employs temporal convolutional network, by default 1
        masking_or_mapping : int, optional
            Set to 0 for masking and 1 for mapping. Changes the loss and the output layer. If masking, clamps the output layer to [-rmax, rmax], by default 1
        output_activation : str, optional
            Only linear is supported, by default "linear"
        rmax : int, optional
            The clamp value if masking is employed, by default 5
        use_batchnorm : int, optional
            Normalization scheme. Set to 1 for batch norm, set to 2 for groupnorm with single group, where 3 is an implementation of instance norm via groupnorm, by default 3
        use_convbnrelu : int, optional
            The order of operation in each convolution blocks. Set to 0 for Norm-Act-Conv, 1 for Conv-Norm-Act and 2 for Conv-Act-Norm, by default 2
        use_act : int, optional
            The activation to employ. Set to 1 for Exponential Linear Unit, and 2 for Swish function, by default 2
        memory_efficient : int, optional
            Employs the checkpoint function from torch.utils.checkpoint to reduce memory usage during forward pass. This function remove the intermediate activations during forward pass and recomputes them during backward pass. Set to 0 to disable and 1 to enable, by default 1
        n_out_srcs : int, optional
            The number of output sources. By default 1.
        """

        n_freq_bins = 257
        target_dim = n_freq_bins
        n_inputs = 2 * n_srcs
        n_outputs = 2 * n_out_srcs

        super().__init__(
            input_dim=target_dim * n_inputs,
            nlayer=nlayer,
            n_units=n_units,
            target_dim=n_freq_bins,
            use_seqmodel=use_seqmodel,
            masking_or_mapping=masking_or_mapping,
            output_activation=output_activation,
            rmax=rmax,
            use_batchnorm=use_batchnorm,
            use_convbnrelu=use_convbnrelu,
            use_act=use_act,
            memory_efficient=memory_efficient,
            n_outputs=n_outputs,
        )
        window_length = 512
        hop_length = 128
        window_func = lambda window_length: torch.sqrt(torch.hann_window(window_length))
        STFT = STFT_module(
            n_fft=window_length,
            window_func=window_func,
            hop_length=hop_length,
            return_complex=False,
        )

        self.STFT = STFT
        self.ref_mic_idx = ref_mic_idx

    def forward_in_STFT(self, Y):
        batch_size, _, _RI, n_frames, n_freq = Y.shape

        # stack microphone channel with batch_dim
        # Y = Y.reshape(batch_size, _RI, n_frames, n_freq)
        Y = Y[:, self.ref_mic_idx]

        X_hat = self.apply_neural_network(Y)

        X_hat = X_hat.reshape(batch_size, 1, _RI, n_frames, n_freq)

        return X_hat

    def forward(self, input: Dict[str, torch.Tensor]):
        """
        Apply the model in

        [extended_summary]

        Parameters
        ----------
        input : Dict[str,Tensor]
            Contains time-domain signals of the mixture of shape (batch_size, n_chns, length), and other components

        Returns
        -------
        Dict[str,Tensor]
            Contains the est_target signal, and other components
        """

        y = input["mixture"]
        batch_size, n_chns, length = y.shape
        y = y[:, self.ref_mic_idx : self.ref_mic_idx + 1]
        Y = self.STFT(y)

        X_hat = self.forward_in_STFT(Y)

        x_hat = self.STFT.backward(X_hat, length)  # batch_size, n_chns, length

        return {
            # "est_target_multi_channel": x_hat,
            "est_target": x_hat[:, self.ref_mic_idx],
            "STFT": self.STFT,
            "ref_mic_idx": self.ref_mic_idx,
        }

    def apply_neural_network(
        self,
        input: torch.Tensor,
        input_dropout_rate=0.0,
        hidden_dropout_rate=0.0,
        dilation_dropout_rate=0.0,
    ):
        """
        Apply the neural network (extracted from Zhang-Qiu's code).

        Parameters
        ----------
        input : torch.Tensor
            (batchsize, 2 x n_srcs , n_frames, n_freqs_bins)

        Returns
        -------
        torch.Tensor
            Enhanced signal with the shape of (batchsize, 2 x n_out_srcs , n_frames, n_freqs_bins)
        """
        batch = input
        if input_dropout_rate > 0.0:
            batch = F.dropout(batch, p=input_dropout_rate, training=True, inplace=False)

        batchsize, _, N, n_freqs = batch.shape

        all_conv_batch = []
        for cc in range(10):
            conv_link_name = "conv%d" % cc
            if hasattr(self, conv_link_name):
                batch = self[conv_link_name](batch)
                eden_link_name = "eden%d" % cc
                if hasattr(self, eden_link_name):
                    batch = self[eden_link_name](batch)
                all_conv_batch.append(batch)
            else:
                break
        # batch.shape is [batchsize, self.n_units, N, 1]

        if self.use_seqmodel == 0:
            batch = batch.squeeze(dim=-1)  # [batchsize, self.n_units, N]
            batch = batch.transpose(1, 2)  # [batchsize, N, self.n_units]
            batch = self.propagate_full_sequence(
                batch, dropout_rate=hidden_dropout_rate
            )  # [batchsize, N, 2*self.n_units]
            batch = batch.transpose(1, 2)  # [batchsize, 2*self.n_units, N]
            batch = batch.unsqueeze(dim=-1)  # [batchsize, 2*self.n_units, N, 1]
        else:
            batch = batch.view(
                [batchsize, self.n_units, N]
            )  # [batchsize, self.n_units, N]
            for ii in range(1, self.nlayer + 1):
                for cc in range(20):
                    conv_link_name = "tcn-conv%d-%d" % (ii, cc)
                    if hasattr(self, conv_link_name):
                        batch = self[conv_link_name](
                            batch,
                            hidden_dropout_rate=hidden_dropout_rate,
                            dilation_dropout_rate=dilation_dropout_rate,
                        )
                    else:
                        break
            batch = batch.unsqueeze(dim=-1)  # [batchsize, self.n_units, N, 1]

        for cc in range(10):
            deconv_link_name = "deconv%d" % cc
            if hasattr(self, deconv_link_name):
                if cc - 1 >= 0 and hasattr(self, "dden%d" % (cc - 1)):
                    batch = self[deconv_link_name](batch)
                else:
                    batch = self[deconv_link_name](
                        torch.cat([batch, all_conv_batch[-1 - cc]], dim=1)
                    )
                dden_link_name = "dden%d" % cc
                if hasattr(self, dden_link_name):
                    batch = self[dden_link_name](
                        torch.cat([batch, all_conv_batch[-1 - cc - 1]], dim=1)
                    )
            else:
                break
        # batch.shape is [batchsize, -1, N, self.target_dim]

        if self.masking_or_mapping == 0:
            # masking
            if self.pitactivation == "linear":
                batch = torch.clamp(batch, -self.rmax, self.rmax)
            else:
                raise
        else:
            # mapping
            if self.output_activation == "linear":
                batch = batch
            else:
                raise

        return batch


class CSeqUNetDense_Multichannel(
    CSeqUNetDense,
):
    def __init__(
        self,
        ref_mic_idx: int = 0,
        n_srcs: int = 1,
        nlayer: int = 2,
        n_units: int = 512,
        use_seqmodel: int = 1,
        masking_or_mapping: int = 1,
        output_activation: str = "linear",
        rmax: int = 5,
        use_batchnorm: int = 3,
        use_convbnrelu: int = 2,
        use_act: int = 2,
        memory_efficient: int = 1,
        n_out_srcs: int = 1,
        **kwargs
    ):
        """
        Interface of Zhang-Qiu Wang's model from [1] with our system. The default parameters are for [1].

        The microphone channels are stacked on batch dimension.

        Parameters
        ----------
        ref_mic_idx: int
            The reference microphone index, by default 0.
        n_srcs : int
            The number of sources. By default 1.
        nlayer : int, optional
            The number of layers in the sequence model. +1 for temporal convolution network, by default 2
        n_units: int, optional
            The number of units in the sequence model. By default 512.
        use_seqmodel : int, optional
            Set to 0 for bi-directional LSTM else employs temporal convolutional network, by default 1
        masking_or_mapping : int, optional
            Set to 0 for masking and 1 for mapping. Changes the loss and the output layer. If masking, clamps the output layer to [-rmax, rmax], by default 1
        output_activation : str, optional
            Only linear is supported, by default "linear"
        rmax : int, optional
            The clamp value if masking is employed, by default 5
        use_batchnorm : int, optional
            Normalization scheme. Set to 1 for batch norm, set to 2 for groupnorm with single group, where 3 is an implementation of instance norm via groupnorm, by default 3
        use_convbnrelu : int, optional
            The order of operation in each convolution blocks. Set to 0 for Norm-Act-Conv, 1 for Conv-Norm-Act and 2 for Conv-Act-Norm, by default 2
        use_act : int, optional
            The activation to employ. Set to 1 for Exponential Linear Unit, and 2 for Swish function, by default 2
        memory_efficient : int, optional
            Employs the checkpoint function from torch.utils.checkpoint to reduce memory usage during forward pass. This function remove the intermediate activations during forward pass and recomputes them during backward pass. Set to 0 to disable and 1 to enable, by default 1
        n_out_srcs : int, optional
            The number of output sources. By default 1.
        """

        n_freq_bins = 257
        target_dim = n_freq_bins
        n_inputs = 2 * n_srcs
        n_outputs = 2 * n_out_srcs

        super().__init__(
            input_dim=target_dim * n_inputs,
            nlayer=nlayer,
            n_units=n_units,
            target_dim=n_freq_bins,
            use_seqmodel=use_seqmodel,
            masking_or_mapping=masking_or_mapping,
            output_activation=output_activation,
            rmax=rmax,
            use_batchnorm=use_batchnorm,
            use_convbnrelu=use_convbnrelu,
            use_act=use_act,
            memory_efficient=memory_efficient,
            n_outputs=n_outputs,
        )
        window_length = 512
        hop_length = 128
        window_func = lambda window_length: torch.sqrt(torch.hann_window(window_length))
        STFT = STFT_module(
            n_fft=window_length,
            window_func=window_func,
            hop_length=hop_length,
            return_complex=False,
        )

        self.STFT = STFT
        self.ref_mic_idx = ref_mic_idx

    def forward_in_STFT(self, Y):
        batch_size, n_chns, _RI, n_frames, n_freq = Y.shape

        # stack microphone channel with batch_dim
        Y = Y.reshape(batch_size * n_chns, _RI, n_frames, n_freq)

        X_hat = self.apply_neural_network(Y)

        X_hat = X_hat.reshape(batch_size, n_chns, _RI, n_frames, n_freq)

        return X_hat

    def forward(self, input: Dict[str, torch.Tensor]):
        """
        Apply the model in

        [extended_summary]

        Parameters
        ----------
        input : Dict[str,Tensor]
            Contains time-domain signals of the mixture of shape (batch_size, n_chns, length), and other components

        Returns
        -------
        Dict[str,Tensor]
            Contains the est_target signal, and other components
        """

        y = input["mixture"]
        batch_size, n_chns, length = y.shape
        Y = self.STFT(y)

        X_hat = self.forward_in_STFT(Y)

        x_hat = self.STFT.backward(X_hat, length)  # batch_size, n_chns, length

        return {
            "est_target_multi_channel": x_hat,
            "est_target": x_hat[:, self.ref_mic_idx],
            "STFT": self.STFT,
            "ref_mic_idx": self.ref_mic_idx,
        }

    def apply_neural_network(
        self,
        input: torch.Tensor,
        input_dropout_rate=0.0,
        hidden_dropout_rate=0.0,
        dilation_dropout_rate=0.0,
    ):
        """
        Apply the neural network (extracted from Zhang-Qiu's code).

        Parameters
        ----------
        input : torch.Tensor
            (batchsize, 2 x n_srcs , n_frames, n_freqs_bins)

        Returns
        -------
        torch.Tensor
            Enhanced signal with the shape of (batchsize, 2 x n_out_srcs , n_frames, n_freqs_bins)
        """
        batch = input
        if input_dropout_rate > 0.0:
            batch = F.dropout(batch, p=input_dropout_rate, training=True, inplace=False)

        batchsize, _, N, n_freqs = batch.shape

        all_conv_batch = []
        for cc in range(10):
            conv_link_name = "conv%d" % cc
            if hasattr(self, conv_link_name):
                batch = self[conv_link_name](batch)
                eden_link_name = "eden%d" % cc
                if hasattr(self, eden_link_name):
                    batch = self[eden_link_name](batch)
                all_conv_batch.append(batch)
            else:
                break
        # batch.shape is [batchsize, self.n_units, N, 1]

        if self.use_seqmodel == 0:
            batch = batch.squeeze(dim=-1)  # [batchsize, self.n_units, N]
            batch = batch.transpose(1, 2)  # [batchsize, N, self.n_units]
            batch = self.propagate_full_sequence(
                batch, dropout_rate=hidden_dropout_rate
            )  # [batchsize, N, 2*self.n_units]
            batch = batch.transpose(1, 2)  # [batchsize, 2*self.n_units, N]
            batch = batch.unsqueeze(dim=-1)  # [batchsize, 2*self.n_units, N, 1]
        else:
            batch = batch.view(
                [batchsize, self.n_units, N]
            )  # [batchsize, self.n_units, N]
            for ii in range(1, self.nlayer + 1):
                for cc in range(20):
                    conv_link_name = "tcn-conv%d-%d" % (ii, cc)
                    if hasattr(self, conv_link_name):
                        batch = self[conv_link_name](
                            batch,
                            hidden_dropout_rate=hidden_dropout_rate,
                            dilation_dropout_rate=dilation_dropout_rate,
                        )
                    else:
                        break
            batch = batch.unsqueeze(dim=-1)  # [batchsize, self.n_units, N, 1]

        for cc in range(10):
            deconv_link_name = "deconv%d" % cc
            if hasattr(self, deconv_link_name):
                if cc - 1 >= 0 and hasattr(self, "dden%d" % (cc - 1)):
                    batch = self[deconv_link_name](batch)
                else:
                    batch = self[deconv_link_name](
                        torch.cat([batch, all_conv_batch[-1 - cc]], dim=1)
                    )
                dden_link_name = "dden%d" % cc
                if hasattr(self, dden_link_name):
                    batch = self[dden_link_name](
                        torch.cat([batch, all_conv_batch[-1 - cc - 1]], dim=1)
                    )
            else:
                break
        # batch.shape is [batchsize, -1, N, self.target_dim]

        if self.masking_or_mapping == 0:
            # masking
            if self.pitactivation == "linear":
                batch = torch.clamp(batch, -self.rmax, self.rmax)
            else:
                raise
        else:
            # mapping
            if self.output_activation == "linear":
                batch = batch
            else:
                raise

        return batch


class CSeqUNetDense_Multichannel_TI_Oracle_MVDR_2nd_stage(
    CSeqUNetDense,
):
    def __init__(
        self,
        first_stage_checkpoint,
        ref_mic_idx: int = 0,
        n_srcs: int = 1,
        nlayer: int = 2,
        n_units: int = 512,
        use_seqmodel: int = 1,
        masking_or_mapping: int = 1,
        output_activation: str = "linear",
        rmax: int = 5,
        use_batchnorm: int = 3,
        use_convbnrelu: int = 2,
        use_act: int = 2,
        memory_efficient: int = 1,
        n_out_srcs: int = 1,
        **kwargs
    ):
        """
        Interface of Zhang-Qiu Wang's model from [1] with our system. The default parameters are for [1].

        The microphone channels are stacked on batch dimension.

        Parameters
        ----------
        ref_mic_idx: int
            The reference microphone index, by default 0.
        n_srcs : int
            The number of sources. By default 1.
        nlayer : int, optional
            The number of layers in the sequence model. +1 for temporal convolution network, by default 2
        n_units: int, optional
            The number of units in the sequence model. By default 512.
        use_seqmodel : int, optional
            Set to 0 for bi-directional LSTM else employs temporal convolutional network, by default 1
        masking_or_mapping : int, optional
            Set to 0 for masking and 1 for mapping. Changes the loss and the output layer. If masking, clamps the output layer to [-rmax, rmax], by default 1
        output_activation : str, optional
            Only linear is supported, by default "linear"
        rmax : int, optional
            The clamp value if masking is employed, by default 5
        use_batchnorm : int, optional
            Normalization scheme. Set to 1 for batch norm, set to 2 for groupnorm with single group, where 3 is an implementation of instance norm via groupnorm, by default 3
        use_convbnrelu : int, optional
            The order of operation in each convolution blocks. Set to 0 for Norm-Act-Conv, 1 for Conv-Norm-Act and 2 for Conv-Act-Norm, by default 2
        use_act : int, optional
            The activation to employ. Set to 1 for Exponential Linear Unit, and 2 for Swish function, by default 2
        memory_efficient : int, optional
            Employs the checkpoint function from torch.utils.checkpoint to reduce memory usage during forward pass. This function remove the intermediate activations during forward pass and recomputes them during backward pass. Set to 0 to disable and 1 to enable, by default 1
        n_out_srcs : int, optional
            The number of output sources. By default 1.
        """

        n_freq_bins = 257
        target_dim = n_freq_bins
        n_inputs = 2 * n_srcs * 2
        n_outputs = 2 * n_out_srcs

        super().__init__(
            input_dim=target_dim * n_inputs,
            nlayer=nlayer,
            n_units=n_units,
            target_dim=n_freq_bins,
            use_seqmodel=use_seqmodel,
            masking_or_mapping=masking_or_mapping,
            output_activation=output_activation,
            rmax=rmax,
            use_batchnorm=use_batchnorm,
            use_convbnrelu=use_convbnrelu,
            use_act=use_act,
            memory_efficient=memory_efficient,
            n_outputs=n_outputs,
        )
        window_length = 512
        hop_length = 128
        window_func = lambda window_length: torch.sqrt(torch.hann_window(window_length))
        STFT = STFT_module(
            n_fft=window_length,
            window_func=window_func,
            hop_length=hop_length,
            return_complex=False,
        )

        self.STFT = STFT
        self.ref_mic_idx = ref_mic_idx

        self.bf1 = TI_MVDR(STFT, ref_mic_idx)
        # self.nn1 = self.load_first_stage(first_stage_checkpoint)

    def load_first_stage(self, checkpoint):

        raise NotImplementedError
        return model

    def forward(self, input: Dict[str, torch.Tensor]):
        """
        Apply the model in


        Parameters
        ----------
        input : Dict[str,Tensor]
            Contains time-domain signals of the mixture of shape (batch_size, n_chns, length), and other components

        Returns
        -------
        Dict[str,Tensor]
            Contains the est_target signal, and other components
        """

        y = input["mixture"]

        batch_size, n_chns, length = y.shape
        Y = self.STFT(y)

        batch_size, n_chns, _RI, n_frames, n_freq = Y.shape

        # X_nn_1 = self.nn1(Y)
        with torch.no_grad():
            X_nn_1 = self.STFT(input["target"])

            X_nbf_1 = self.bf1.forward_in_STFT(Y, X_hat=X_nn_1, V_hat=Y - X_nn_1)
            # stack microphone channel with batch_dim

        X_nbf_1 = X_nbf_1.reshape(batch_size * n_chns, _RI, n_frames, n_freq)
        Y = Y.reshape(batch_size * n_chns, _RI, n_frames, n_freq)

        Y_X_nbf_1_concat = torch.cat(
            [X_nbf_1, Y], dim=1
        )  # batch_size * n_chns, 2 * _RI, n_frames, n_freq

        X_nn_2 = self.apply_neural_network(Y_X_nbf_1_concat)

        X = X_nn_2.reshape(batch_size, n_chns, _RI, n_frames, n_freq)

        x = self.STFT.backward(X, length)  # batch_size, n_chns, length

        return {
            "est_target_multi_channel": x,
            "est_target": x[:, self.ref_mic_idx],
            "STFT": self.STFT,
            "ref_mic_idx": self.ref_mic_idx,
        }


class CSeqUNetDense_Multichannel_TI_MVDR_2nd_stage(
    CSeqUNetDense,
):
    def __init__(
        self,
        first_stage_ckpt_path,
        first_stage_model,
        ref_mic_idx: int = 0,
        n_srcs: int = 1,
        nlayer: int = 2,
        n_units: int = 512,
        use_seqmodel: int = 1,
        masking_or_mapping: int = 1,
        output_activation: str = "linear",
        rmax: int = 5,
        use_batchnorm: int = 3,
        use_convbnrelu: int = 2,
        use_act: int = 2,
        memory_efficient: int = 1,
        n_out_srcs: int = 1,
        **kwargs
    ):
        """
        Interface of Zhang-Qiu Wang's model from [1] with our system. The default parameters are for [1].

        The microphone channels are stacked on batch dimension.

        Parameters
        ----------
        ref_mic_idx: int
            The reference microphone index, by default 0.
        n_srcs : int
            The number of sources. By default 1.
        nlayer : int, optional
            The number of layers in the sequence model. +1 for temporal convolution network, by default 2
        n_units: int, optional
            The number of units in the sequence model. By default 512.
        use_seqmodel : int, optional
            Set to 0 for bi-directional LSTM else employs temporal convolutional network, by default 1
        masking_or_mapping : int, optional
            Set to 0 for masking and 1 for mapping. Changes the loss and the output layer. If masking, clamps the output layer to [-rmax, rmax], by default 1
        output_activation : str, optional
            Only linear is supported, by default "linear"
        rmax : int, optional
            The clamp value if masking is employed, by default 5
        use_batchnorm : int, optional
            Normalization scheme. Set to 1 for batch norm, set to 2 for groupnorm with single group, where 3 is an implementation of instance norm via groupnorm, by default 3
        use_convbnrelu : int, optional
            The order of operation in each convolution blocks. Set to 0 for Norm-Act-Conv, 1 for Conv-Norm-Act and 2 for Conv-Act-Norm, by default 2
        use_act : int, optional
            The activation to employ. Set to 1 for Exponential Linear Unit, and 2 for Swish function, by default 2
        memory_efficient : int, optional
            Employs the checkpoint function from torch.utils.checkpoint to reduce memory usage during forward pass. This function remove the intermediate activations during forward pass and recomputes them during backward pass. Set to 0 to disable and 1 to enable, by default 1
        n_out_srcs : int, optional
            The number of output sources. By default 1.
        """

        n_freq_bins = 257
        target_dim = n_freq_bins
        n_inputs = 2 * n_srcs * 2
        n_outputs = 2 * n_out_srcs

        super().__init__(
            input_dim=target_dim * n_inputs,
            nlayer=nlayer,
            n_units=n_units,
            target_dim=n_freq_bins,
            use_seqmodel=use_seqmodel,
            masking_or_mapping=masking_or_mapping,
            output_activation=output_activation,
            rmax=rmax,
            use_batchnorm=use_batchnorm,
            use_convbnrelu=use_convbnrelu,
            use_act=use_act,
            memory_efficient=memory_efficient,
            n_outputs=n_outputs,
        )
        window_length = 512
        hop_length = 128
        window_func = lambda window_length: torch.sqrt(torch.hann_window(window_length))
        STFT = STFT_module(
            n_fft=window_length,
            window_func=window_func,
            hop_length=hop_length,
            return_complex=False,
        )

        self.STFT = STFT

        self.ref_mic_idx = ref_mic_idx
        
        self.bf1 = TI_MVDR(STFT, ref_mic_idx)
        self.nn1 = self.load_first_stage(first_stage_ckpt_path, first_stage_model())

    def load_first_stage(self, first_stage_ckpt_path, first_stage_model):

        checkpoint = torch.load(first_stage_ckpt_path, map_location=torch.device("cpu"))
        try:
            first_stage_model.load_state_dict(checkpoint["state_dict"])
        except RuntimeError:
            try:
                # removes model.
                state_dict = checkpoint["state_dict"]
                new_state_dict = {}
                for key, value in state_dict.items():
                    if "model." in key:
                        key = key.replace("model.", "")

                    new_state_dict[key] = value
                first_stage_model.load_state_dict(new_state_dict)
            except Exception as e:
                raise e

        return first_stage_model

    def forward_in_STFT(self, Y):

        batch_size, n_chns, _RI, n_frames, n_freq = Y.shape

        with torch.no_grad():

            X_nn_1 = self.nn1.forward_in_STFT(Y)

            X_nbf_1 = self.bf1.forward_in_STFT(Y, X_hat=X_nn_1, V_hat=Y - X_nn_1)
            # stack microphone channel with batch_dim

        X_nbf_1 = X_nbf_1.reshape(batch_size * n_chns, _RI, n_frames, n_freq)
        Y = Y.reshape(batch_size * n_chns, _RI, n_frames, n_freq)

        Y_X_nbf_1_concat = torch.cat(
            [X_nbf_1, Y], dim=1
        )  # batch_size * n_chns, 2 * _RI, n_frames, n_freq

        X_nn_2 = self.apply_neural_network(Y_X_nbf_1_concat)

        X_nn_2 = X_nn_2.reshape(batch_size, n_chns, _RI, n_frames, n_freq)

        return X_nn_2

    def forward(self, input: Dict[str, torch.Tensor]):
        """
        Apply the model in


        Parameters
        ----------
        input : Dict[str,Tensor]
            Contains time-domain signals of the mixture of shape (batch_size, n_chns, length), and other components

        Returns
        -------
        Dict[str,Tensor]
            Contains the est_target signal, and other components
        """

        y = input["mixture"]
        batch_size, n_chns, length = y.shape
        Y = self.STFT(y)

        X_hat = self.forward_in_STFT(Y)
        x_hat = self.STFT.backward(X_hat, length)  # batch_size, n_chns, length

        self.ref_mic_idx = 0
        return {
            "est_target_multi_channel": x_hat,
            "est_target": x_hat[:, self.ref_mic_idx],
            "STFT": self.STFT,
            "ref_mic_idx": self.ref_mic_idx,
        }


class CSeqUNetDense_Multichannel_TI_MVDR_2nd_stage_wo_stft(
    CSeqUNetDense,
):
    def __init__(
        self,
        first_stage_ckpt_path,
        first_stage_model,
        ref_mic_idx: int = 0,
        n_srcs: int = 1,
        nlayer: int = 2,
        n_units: int = 512,
        use_seqmodel: int = 1,
        masking_or_mapping: int = 1,
        output_activation: str = "linear",
        rmax: int = 5,
        use_batchnorm: int = 3,
        use_convbnrelu: int = 2,
        use_act: int = 2,
        memory_efficient: int = 1,
        n_out_srcs: int = 1,
        **kwargs
    ):
        """
        Interface of Zhang-Qiu Wang's model from [1] with our system. The default parameters are for [1].

        The microphone channels are stacked on batch dimension.

        Parameters
        ----------
        ref_mic_idx: int
            The reference microphone index, by default 0.
        n_srcs : int
            The number of sources. By default 1.
        nlayer : int, optional
            The number of layers in the sequence model. +1 for temporal convolution network, by default 2
        n_units: int, optional
            The number of units in the sequence model. By default 512.
        use_seqmodel : int, optional
            Set to 0 for bi-directional LSTM else employs temporal convolutional network, by default 1
        masking_or_mapping : int, optional
            Set to 0 for masking and 1 for mapping. Changes the loss and the output layer. If masking, clamps the output layer to [-rmax, rmax], by default 1
        output_activation : str, optional
            Only linear is supported, by default "linear"
        rmax : int, optional
            The clamp value if masking is employed, by default 5
        use_batchnorm : int, optional
            Normalization scheme. Set to 1 for batch norm, set to 2 for groupnorm with single group, where 3 is an implementation of instance norm via groupnorm, by default 3
        use_convbnrelu : int, optional
            The order of operation in each convolution blocks. Set to 0 for Norm-Act-Conv, 1 for Conv-Norm-Act and 2 for Conv-Act-Norm, by default 2
        use_act : int, optional
            The activation to employ. Set to 1 for Exponential Linear Unit, and 2 for Swish function, by default 2
        memory_efficient : int, optional
            Employs the checkpoint function from torch.utils.checkpoint to reduce memory usage during forward pass. This function remove the intermediate activations during forward pass and recomputes them during backward pass. Set to 0 to disable and 1 to enable, by default 1
        n_out_srcs : int, optional
            The number of output sources. By default 1.
        """

        n_freq_bins = 257
        target_dim = n_freq_bins
        n_inputs = 2 * n_srcs * 2
        n_outputs = 2 * n_out_srcs

        super().__init__(
            input_dim=target_dim * n_inputs,
            nlayer=nlayer,
            n_units=n_units,
            target_dim=n_freq_bins,
            use_seqmodel=use_seqmodel,
            masking_or_mapping=masking_or_mapping,
            output_activation=output_activation,
            rmax=rmax,
            use_batchnorm=use_batchnorm,
            use_convbnrelu=use_convbnrelu,
            use_act=use_act,
            memory_efficient=memory_efficient,
            n_outputs=n_outputs,
        )
        window_length = 512
        hop_length = 128
        window_func = lambda window_length: torch.sqrt(torch.hann_window(window_length))
        STFT = STFT_module(
            n_fft=window_length,
            window_func=window_func,
            hop_length=hop_length,
            return_complex=False,
        )

        # self.STFT = STFT

        self.ref_mic_idx = ref_mic_idx
        
        self.bf1 = TI_MVDR(STFT, ref_mic_idx)
        self.nn1 = self.load_first_stage(first_stage_ckpt_path, first_stage_model())

    def load_first_stage(self, first_stage_ckpt_path, first_stage_model):

        checkpoint = torch.load(first_stage_ckpt_path, map_location=torch.device("cpu"))
        try:
            first_stage_model.load_state_dict(checkpoint["state_dict"])
        except RuntimeError:
            try:
                # removes model.
                state_dict = checkpoint["state_dict"]
                new_state_dict = {}
                for key, value in state_dict.items():
                    if "model." in key:
                        key = key.replace("model.", "")

                    new_state_dict[key] = value
                first_stage_model.load_state_dict(new_state_dict)
            except Exception as e:
                raise e

        return first_stage_model

    def forward_in_STFT(self, Y):

        batch_size, n_chns, _RI, n_frames, n_freq = Y.shape

        with torch.no_grad():

            X_nn_1 = self.nn1.forward_in_STFT(Y)

            X_nbf_1 = self.bf1.forward_in_STFT(Y, X_hat=X_nn_1, V_hat=Y - X_nn_1)
            # stack microphone channel with batch_dim

        X_nbf_1 = X_nbf_1.reshape(batch_size * n_chns, _RI, n_frames, n_freq)
        Y = Y.reshape(batch_size * n_chns, _RI, n_frames, n_freq)

        Y_X_nbf_1_concat = torch.cat(
            [X_nbf_1, Y], dim=1
        )  # batch_size * n_chns, 2 * _RI, n_frames, n_freq

        X_nn_2 = self.apply_neural_network(Y_X_nbf_1_concat)

        X_nn_2 = X_nn_2.reshape(batch_size, n_chns, _RI, n_frames, n_freq)

        return X_nn_2

    def forward(self, input: Dict[str, torch.Tensor]):
        """
        Apply the model in


        Parameters
        ----------
        input : Dict[str,Tensor]
            Contains time-domain signals of the mixture of shape (batch_size, n_chns, length), and other components

        Returns
        -------
        Dict[str,Tensor]
            Contains the est_target signal, and other components
        """

        # y = input["mixture"]
        Y = input
        # batch_size, n_chns, length = y.shape
        # Y = self.STFT(y)

        X_hat = self.forward_in_STFT(Y)
        # x_hat = self.STFT.backward(X_hat, length)  # batch_size, n_chns, length
        return X_hat
        self.ref_mic_idx = 0
        # return {
        #     "est_target_multi_channel": X_hat,
        #     # "est_target": x_hat[:, self.ref_mic_idx],
        #     # "STFT": self.STFT,
        #     "ref_mic_idx": self.ref_mic_idx,
        # }



class CSeqUNetDense_Multichannel_TV_MVDR_2nd_stage(
    CSeqUNetDense,
):
    def __init__(
        self,
        first_stage_ckpt_path,
        first_stage_model,
        ref_mic_idx: int = 0,
        n_srcs: int = 1,
        nlayer: int = 2,
        n_units: int = 512,
        use_seqmodel: int = 1,
        masking_or_mapping: int = 1,
        output_activation: str = "linear",
        rmax: int = 5,
        use_batchnorm: int = 3,
        use_convbnrelu: int = 2,
        use_act: int = 2,
        memory_efficient: int = 1,
        n_out_srcs: int = 1,
        **kwargs
    ):
        """
        Interface of Zhang-Qiu Wang's model from [1] with our system. The default parameters are for [1].

        The microphone channels are stacked on batch dimension.

        Parameters
        ----------
        ref_mic_idx: int
            The reference microphone index, by default 0.
        n_srcs : int
            The number of sources. By default 1.
        nlayer : int, optional
            The number of layers in the sequence model. +1 for temporal convolution network, by default 2
        n_units: int, optional
            The number of units in the sequence model. By default 512.
        use_seqmodel : int, optional
            Set to 0 for bi-directional LSTM else employs temporal convolutional network, by default 1
        masking_or_mapping : int, optional
            Set to 0 for masking and 1 for mapping. Changes the loss and the output layer. If masking, clamps the output layer to [-rmax, rmax], by default 1
        output_activation : str, optional
            Only linear is supported, by default "linear"
        rmax : int, optional
            The clamp value if masking is employed, by default 5
        use_batchnorm : int, optional
            Normalization scheme. Set to 1 for batch norm, set to 2 for groupnorm with single group, where 3 is an implementation of instance norm via groupnorm, by default 3
        use_convbnrelu : int, optional
            The order of operation in each convolution blocks. Set to 0 for Norm-Act-Conv, 1 for Conv-Norm-Act and 2 for Conv-Act-Norm, by default 2
        use_act : int, optional
            The activation to employ. Set to 1 for Exponential Linear Unit, and 2 for Swish function, by default 2
        memory_efficient : int, optional
            Employs the checkpoint function from torch.utils.checkpoint to reduce memory usage during forward pass. This function remove the intermediate activations during forward pass and recomputes them during backward pass. Set to 0 to disable and 1 to enable, by default 1
        n_out_srcs : int, optional
            The number of output sources. By default 1.
        """

        n_freq_bins = 257
        target_dim = n_freq_bins
        n_inputs = 2 * n_srcs * 2
        n_outputs = 2 * n_out_srcs

        super().__init__(
            input_dim=target_dim * n_inputs,
            nlayer=nlayer,
            n_units=n_units,
            target_dim=n_freq_bins,
            use_seqmodel=use_seqmodel,
            masking_or_mapping=masking_or_mapping,
            output_activation=output_activation,
            rmax=rmax,
            use_batchnorm=use_batchnorm,
            use_convbnrelu=use_convbnrelu,
            use_act=use_act,
            memory_efficient=memory_efficient,
            n_outputs=n_outputs,
        )
        window_length = 512
        hop_length = 128
        window_func = lambda window_length: torch.sqrt(torch.hann_window(window_length))
        STFT = STFT_module(
            n_fft=window_length,
            window_func=window_func,
            hop_length=hop_length,
            return_complex=False,
        )

        self.STFT = STFT

        self.ref_mic_idx = ref_mic_idx
        
        self.bf1 = TV_MVDR(STFT, ref_mic_idx)
        self.nn1 = self.load_first_stage(first_stage_ckpt_path, first_stage_model())

    def load_first_stage(self, first_stage_ckpt_path, first_stage_model):

        checkpoint = torch.load(first_stage_ckpt_path, map_location=torch.device("cpu"))
        try:
            first_stage_model.load_state_dict(checkpoint["state_dict"])
        except RuntimeError:
            try:
                # removes model.
                state_dict = checkpoint["state_dict"]
                new_state_dict = {}
                for key, value in state_dict.items():
                    if "model." in key:
                        key = key.replace("model.", "")

                    new_state_dict[key] = value
                first_stage_model.load_state_dict(new_state_dict)
            except Exception as e:
                raise e

        return first_stage_model

    def forward_in_STFT(self, Y):

        batch_size, n_chns, _RI, n_frames, n_freq = Y.shape

        with torch.no_grad():

            X_nn_1 = self.nn1.forward_in_STFT(Y)

            X_nbf_1 = self.bf1.forward_in_STFT(Y, X_hat=X_nn_1, V_hat=Y - X_nn_1)
            # stack microphone channel with batch_dim

        X_nbf_1 = X_nbf_1.reshape(batch_size * n_chns, _RI, n_frames, n_freq)
        Y = Y.reshape(batch_size * n_chns, _RI, n_frames, n_freq)

        Y_X_nbf_1_concat = torch.cat(
            [X_nbf_1, Y], dim=1
        )  # batch_size * n_chns, 2 * _RI, n_frames, n_freq

        X_nn_2 = self.apply_neural_network(Y_X_nbf_1_concat)

        X_nn_2 = X_nn_2.reshape(batch_size, n_chns, _RI, n_frames, n_freq)

        return X_nn_2

    def forward(self, input: Dict[str, torch.Tensor]):
        """
        Apply the model in


        Parameters
        ----------
        input : Dict[str,Tensor]
            Contains time-domain signals of the mixture of shape (batch_size, n_chns, length), and other components

        Returns
        -------
        Dict[str,Tensor]
            Contains the est_target signal, and other components
        """

        y = input["mixture"]
        batch_size, n_chns, length = y.shape
        Y = self.STFT(y)

        X_hat = self.forward_in_STFT(Y)
        x_hat = self.STFT.backward(X_hat, length)  # batch_size, n_chns, length

        self.ref_mic_idx = 0
        return {
            "est_target_multi_channel": x_hat,
            "est_target": x_hat[:, self.ref_mic_idx],
            "STFT": self.STFT,
            "ref_mic_idx": self.ref_mic_idx,
        }



def type_of_script():
    try:
        ipy_str = str(type(get_ipython()))
        if "zmqshell" in ipy_str:
            return "jupyter"
        if "terminal" in ipy_str:
            return "ipython"
    except:
        return "terminal"


if type_of_script() == "jupyter" and __name__ == "__main__":

    from torchinfo import summary

    model = CSeqUNetDense_interfaced(n_srcs=1, n_out_srcs=1, ref_mic_idx=0)
    # [batchsize, N, -1, n_freqs]
    n_chns = 7
    batch_size = 8
    n_frames = 10240

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    # compute STFT size
    # n_freq_bins = window_length // 2 + 1
    # n_freq_bins = (
    #     n_freq_bins + 1 if not (n_frames - window_length) % hop_length else n_freq_bins
    # )
    # n_freq_frames = n_frames // hop_length
    # hop_length = 128
    # window_length = 512
    # input_size = (batch_size * n_chns, 2, n_freq_frames, n_freq_bins)
    input = {"mixture": torch.zeros(batch_size, n_chns, n_frames)}
    print(
        summary(
            model,
            input_data=input,
            device="cpu",
            depth=2,
            col_names=[
                # "kernel_size",
                # "input_size",
                "output_size",
                "num_params",
            ],  # , "mult_adds"],
        )
    )

    # input = torch.zeros(input_size)
    _out = model(input)
# print(_out["est_target"].shape)

# %%
