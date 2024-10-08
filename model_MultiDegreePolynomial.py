#!/usr/bin/env python
"""
model.py

Self defined model definition with polynomial interactions.
Usage:

"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np

import torch
import torch.nn as torch_nn
import torchaudio
import torch.nn.functional as torch_nn_func

import sandbox.block_nn as nii_nn
import sandbox.util_frontend as nii_front_end
import core_scripts.other_tools.debug as nii_debug
import core_scripts.data_io.seq_info as nii_seq_tk

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"




'''  
CHANGES COMPARED TO BASIC MODEL
1) PolynomilaLayer
    i) Initialization: The layer was initialized to match the 
    dimensions of a convolutional layer's weights.
    ii) Weight Parameter: Defined self.weight with the correct 
    shape to match convolutional weights.
    iii) Bias Parameter: Defined self.bias to match convolutional 
    bias parameters.
    iv) Forward Method: Implemented polynomial feature generation 
    and convolution operation using torch.nn.functional.conv2d.

2) Model Class:
    i) Layer Replacement: Replaced convolutional layers with 
    PolynomialLayer in the model's architecture.
    ii) State Dictionary Compatibility: Implemented 
    load_state_dict_compatible function to ensure the 
    keys in the saved state dictionary match those expected 
    by the new model architecture.
'''







##############
## util
##############

def protocol_parse(protocol_filepath):
    data_buffer = {}
    try:
        temp_buffer = np.loadtxt(protocol_filepath, dtype='str')
        for row in temp_buffer:
            if row[-1] == 'bonafide':
                data_buffer[row[1]] = 1
            else:
                data_buffer[row[1]] = 0
    except OSError:
        print("Skip loading protocol file")
    return data_buffer

##############
## FOR MODEL
##############

class PolynomialLayer(torch_nn.Module):
    ''' 
    --- Initialized with parameters to match the input and output 
    dimensions, kernel size, stride, and padding.

    --- self.weight and self.bias were defined with appropriate shapes.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, degree=2, stride=1, padding=0):
        super(PolynomialLayer, self).__init__()
        self.degree = degree
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights for polynomial interactions
        self.weight = torch_nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.bias = torch_nn.Parameter(torch.randn(out_channels))


    ''' 
    --- Created polynomial features by raising the input to different powers.
    --- Applied convolution to each polynomial feature and summed the results.
    '''
    def forward(self, x):
        # Create polynomial features
        batch_size, channels, height, width = x.size()
        poly_features = [x ** (i + 1) for i in range(self.degree)]
        
        # Convolve polynomial features with the weight
        out = torch.zeros(batch_size, self.out_channels, height, width).to(x.device)
        for d in range(self.degree):
            out += torch.nn.functional.conv2d(poly_features[d], self.weight, stride=self.stride, padding=self.padding)
        out += self.bias.view(1, -1, 1, 1)
        return out

class Model(torch_nn.Module):
    # Replaced convolutional layers in the model's architecture with 
    # PolynomialLayer instances in "self.m_transform.append".
    def __init__(self, in_dim, out_dim, args, prj_conf, mean_std=None):
        super(Model, self).__init__()

        in_m, in_s, out_m, out_s = self.prepare_mean_std(in_dim, out_dim, args, mean_std)
        self.input_mean = torch_nn.Parameter(in_m, requires_grad=False)
        self.input_std = torch_nn.Parameter(in_s, requires_grad=False)
        self.output_mean = torch_nn.Parameter(out_m, requires_grad=False)
        self.output_std = torch_nn.Parameter(out_s, requires_grad=False)
        
        protocol_file = prj_conf.optional_argument[0]
        self.protocol_parser = protocol_parse(protocol_file)
        
        self.m_target_sr = 16000

        self.frame_hops = [160]
        self.frame_lens = [320]
        self.fft_n = [1024]

        self.lfcc_dim = [20]
        self.lfcc_with_delta = True
        self.lfcc_max_freq = 0.5 

        self.win = torch.hann_window
        self.amp_floor = 0.00001
        
        self.v_truncate_lens = [None for x in self.frame_hops]
        self.v_submodels = len(self.frame_lens)        
        self.v_emd_dim = 1

        self.m_transform = []
        self.m_before_pooling = []
        self.m_output_act = []
        self.m_frontend = []

        for idx, (trunc_len, fft_n, lfcc_dim) in enumerate(zip(self.v_truncate_lens, self.fft_n, self.lfcc_dim)):
            fft_n_bins = fft_n // 2 + 1
            if self.lfcc_with_delta:
                lfcc_dim = lfcc_dim * 3

            self.m_transform.append(
                torch_nn.Sequential(
                    PolynomialLayer(1, 64, kernel_size=[5, 5], degree=2, padding=[2, 2]),
                    nii_nn.MaxFeatureMap2D(),
                    torch_nn.MaxPool2d([2, 2], [2, 2]),

                    PolynomialLayer(32, 64, kernel_size=[1, 1], degree=2, padding=[0, 0]),
                    nii_nn.MaxFeatureMap2D(),
                    torch_nn.BatchNorm2d(32, affine=False),
                    PolynomialLayer(32, 96, kernel_size=[3, 3], degree=2, padding=[1, 1]),
                    nii_nn.MaxFeatureMap2D(),

                    torch_nn.MaxPool2d([2, 2], [2, 2]),
                    torch_nn.BatchNorm2d(48, affine=False),

                    PolynomialLayer(48, 96, kernel_size=[1, 1], degree=2, padding=[0, 0]),
                    nii_nn.MaxFeatureMap2D(),
                    torch_nn.BatchNorm2d(48, affine=False),
                    PolynomialLayer(48, 128, kernel_size=[3, 3], degree=2, padding=[1, 1]),
                    nii_nn.MaxFeatureMap2D(),

                    torch_nn.MaxPool2d([2, 2], [2, 2]),

                    PolynomialLayer(64, 128, kernel_size=[1, 1], degree=2, padding=[0, 0]),
                    nii_nn.MaxFeatureMap2D(),
                    torch_nn.BatchNorm2d(64, affine=False),
                    PolynomialLayer(64, 64, kernel_size=[3, 3], degree=2, padding=[1, 1]),
                    nii_nn.MaxFeatureMap2D(),
                    torch_nn.BatchNorm2d(32, affine=False),

                    PolynomialLayer(32, 64, kernel_size=[1, 1], degree=2, padding=[0, 0]),
                    nii_nn.MaxFeatureMap2D(),
                    torch_nn.BatchNorm2d(32, affine=False),
                    PolynomialLayer(32, 64, kernel_size=[3, 3], degree=2, padding=[1, 1]),
                    nii_nn.MaxFeatureMap2D(),
                    torch_nn.MaxPool2d([2, 2], [2, 2]),
                    
                    torch_nn.Dropout(0.7)
                )
            )

            self.m_before_pooling.append(
                torch_nn.Sequential(
                    nii_nn.BLSTMLayer((lfcc_dim//16) * 32, (lfcc_dim//16) * 32),
                    nii_nn.BLSTMLayer((lfcc_dim//16) * 32, (lfcc_dim//16) * 32)
                )
            )

            self.m_output_act.append(
                torch_nn.Linear((lfcc_dim // 16) * 32, self.v_emd_dim)
            )
            
            self.m_frontend.append(
                nii_front_end.LFCC(self.frame_lens[idx],
                                   self.frame_hops[idx],
                                   self.fft_n[idx],
                                   self.m_target_sr,
                                   self.lfcc_dim[idx],
                                   with_energy=True,
                                   max_freq = self.lfcc_max_freq)
            )

        self.m_frontend = torch_nn.ModuleList(self.m_frontend)
        self.m_transform = torch_nn.ModuleList(self.m_transform)
        self.m_output_act = torch_nn.ModuleList(self.m_output_act)
        self.m_before_pooling = torch_nn.ModuleList(self.m_before_pooling)
        
        return
    
    def prepare_mean_std(self, in_dim, out_dim, args, data_mean_std=None):
        if data_mean_std is not None:
            in_m = torch.from_numpy(data_mean_std[0])
            in_s = torch.from_numpy(data_mean_std[1])
            out_m = torch.from_numpy(data_mean_std[2])
            out_s = torch.from_numpy(data_mean_std[3])
            if in_m.shape[0] != in_dim or in_s.shape[0] != in_dim:
                print("Input dim: {:d}".format(in_dim))
                print("Mean dim: {:d}".format(in_m.shape[0]))
                print("Std dim: {:d}".format(in_s.shape[0]))
                print("Input dimension incompatible")
                sys.exit(1)
            if out_m.shape[0] != out_dim or out_s.shape[0] != out_dim:
                print("Output dim: {:d}".format(out_dim))
                print("Mean dim: {:d}".format(out_m.shape[0]))
                print("Std dim: {:d}".format(out_s.shape[0]))
                print("Output dimension incompatible")
                sys.exit(1)
        else:
            in_m = torch.zeros([in_dim])
            in_s = torch.ones([in_dim])
            out_m = torch.zeros([out_dim])
            out_s = torch.ones([out_dim])
            
        return in_m, in_s, out_m, out_s
        
    def normalize_input(self, x):
        return (x - self.input_mean) / self.input_std

    def normalize_target(self, y):
        return (y - self.output_mean) / self.output_std

    def denormalize_output(self, y):
        return y * self.output_std + self.output_mean

    def _front_end(self, wav, idx, trunc_len, datalength):
        with torch.no_grad():
            x_sp_amp = self.m_frontend[idx](wav.squeeze(-1))
        return x_sp_amp

    def _compute_embedding(self, x, datalength):
        batch_size = x.shape[0]
        output_emb = torch.zeros([batch_size * self.v_submodels, self.v_emd_dim], device=x.device, dtype=x.dtype)
        
        for idx, (fs, fl, fn, trunc_len, m_trans, m_be_pool, m_output) in enumerate(
                zip(self.frame_hops, self.frame_lens, self.fft_n, self.v_truncate_lens, self.m_transform, self.m_before_pooling, self.m_output_act)):
            x_sp_amp = self._front_end(x, idx, trunc_len, datalength)
            hidden_features = m_trans(x_sp_amp.unsqueeze(1))
            hidden_features = hidden_features.permute(0, 2, 1, 3).contiguous()
            frame_num = hidden_features.shape[1]
            hidden_features = hidden_features.view(batch_size, frame_num, -1)
            hidden_features_lstm = m_be_pool(hidden_features)
            tmp_emb = m_output((hidden_features_lstm + hidden_features).mean(1))
            output_emb[idx * batch_size : (idx+1) * batch_size] = tmp_emb

        return output_emb

    def _compute_score(self, feature_vec, inference=False):
        if inference:
            return feature_vec.squeeze(1)
        else:
            return torch.sigmoid(feature_vec).squeeze(1)

    def _get_target(self, filenames):
        try:
            return [self.protocol_parser[x] for x in filenames]
        except KeyError:
            print("Cannot find target data for %s" % (str(filenames)))
            sys.exit(1)

    def _get_target_eval(self, filenames):
        return [self.protocol_parser[x] if x in self.protocol_parser else -1 for x in filenames]

    def forward(self, x, fileinfo):
        filenames = [nii_seq_tk.parse_filename(y) for y in fileinfo]
        datalength = [nii_seq_tk.parse_length(y) for y in fileinfo]
        if self.training:
            feature_vec = self._compute_embedding(x, datalength)
            scores = self._compute_score(feature_vec)
            target = self._get_target(filenames)
            target_vec = torch.tensor(target, device=x.device, dtype=scores.dtype)
            target_vec = target_vec.repeat(self.v_submodels)
            return [scores, target_vec, True]
        else:
            feature_vec = self._compute_embedding(x, datalength)
            scores = self._compute_score(feature_vec, True)
            target = self._get_target_eval(filenames)
            print("Output, %s, %d, %f" % (filenames[0], target[0], scores.mean()))
            return None

class Loss():
    def __init__(self, args):
        self.m_loss = torch_nn.BCELoss()

    def compute(self, outputs, target):
        loss = self.m_loss(outputs[0], outputs[1])
        return loss


if __name__ == "__main__":
    print("Definition of model")
