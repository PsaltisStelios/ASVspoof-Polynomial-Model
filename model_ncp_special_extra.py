#!/usr/bin/env python
"""
model.py

Self defined model definition.
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

##############
## util
##############

def protocol_parse(protocol_filepath):
    """ Parse protocol of ASVspoof2019 and get bonafide/spoof for each trial
    
    input:
    -----
      protocol_filepath: string, path to the protocol file
        for convenience, I put train/dev/eval trials into a single protocol file
    
    output:
    -------
      data_buffer: dic, data_bufer[filename] -> 1 (bonafide), 0 (spoof)
    """ 
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



def get_norm(norm_type):
    """ Define normalization layer """
    if norm_type == 'batch':
        return torch_nn.BatchNorm2d
    elif norm_type == 'instance':
        return torch_nn.InstanceNorm2d
    else:
        return lambda x: torch.nn.Identity()

class SinglePoly(torch_nn.Module):
    def __init__(self, in_planes, planes, stride=1, kernel_size=3, norm_layer=None, **kwargs):
        super(SinglePoly, self).__init__()
        norm_layer = get_norm(norm_layer)

        self.conv1 = torch_nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = norm_layer(planes)

        self.conv2 = torch_nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = norm_layer(planes)

        # Special convolution
        kernel_size_S = 1
        pad = kernel_size_S // 2
        self.conv_S = torch_nn.Conv2d(in_planes, planes, kernel_size=kernel_size_S, stride=stride, padding=pad, bias=False)
        self.bnS = norm_layer(planes)  # Using the same normalization layer for simplicity


    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        
        # Special convolution
        out_S = self.bnS(self.conv_S(x))
        
        # Polynomial interaction
        return out_S * out + out_S


class Model(torch_nn.Module):
    """ Model definition integrating SinglePoly block """
    def __init__(self, in_dim, out_dim, args, prj_conf, mean_std=None):
        super(Model, self).__init__()

        # mean and std for input and output normalization
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
        self.v_truncate_lens = [None for _ in self.frame_hops]
        self.v_submodels = len(self.frame_lens)
        self.v_emd_dim = 1

        self.m_transform = []
        self.m_before_pooling = []
        self.m_output_act = []
        self.m_frontend = []

        for idx, (trunc_len, fft_n, lfcc_dim) in enumerate(zip(self.v_truncate_lens, self.fft_n, self.lfcc_dim)):
            fft_n_bins = fft_n // 2 + 1
            if self.lfcc_with_delta:
                lfcc_dim *= 3

            # Using SinglePoly as part of feature extraction
            self.m_transform.append(
                torch_nn.Sequential(
                    SinglePoly(1, 64, kernel_size=5, norm_layer='batch'),
                    torch_nn.MaxPool2d([2, 2], [2, 2]),
                    SinglePoly(64, 64, kernel_size=3, norm_layer='batch'),
                    torch_nn.MaxPool2d([2, 2], [2, 2]),
                    SinglePoly(64, 64, kernel_size=3, norm_layer='batch'),
                    torch_nn.MaxPool2d([2, 2], [2, 2]),
                    SinglePoly(64, 64, kernel_size=3, norm_layer='batch'),
                    torch_nn.MaxPool2d([2, 2], [2, 2])
                )
            )

            '''  FIRST MAJOR CHANGE TO FIX THE DIM
            self.m_before_pooling.append(   
                torch_nn.Sequential(
                    # 1st change we make to the dim
                    # nii_nn.BLSTMLayer(960, 960) 
                    nii_nn.BLSTMLayer((lfcc_dim // 16) * 64, (lfcc_dim // 16) * 64),
                    nii_nn.BLSTMLayer((lfcc_dim // 16) * 64, (lfcc_dim // 16) * 64)
                )
            )
            '''

            
            self.m_before_pooling.append(
                torch_nn.Sequential(
                    nii_nn.BLSTMLayer(192, 192),
                    nii_nn.BLSTMLayer(192, 192),
                    nii_nn.BLSTMLayer(192, 192),
                )
            )
            



            ''' 
            shape of hidden_features before "m_before_pooltorch" is 
            torch.Size([64, 51, 960]). Expected 192, got 960 meaning
            m_trans outputs a tensor with 960 as its last dimension 
            but LSTM expects 192

            '''


            ''' SECOND MAJOR CHANGE TO FIX THE DIM
            self.m_output_act.append(
                torch_nn.Linear((lfcc_dim // 16) * 64, self.v_emd_dim)
            )
            '''

            self.m_output_act.append(
                torch_nn.Linear(192, self.v_emd_dim)
            )

            self.m_frontend.append(
                nii_front_end.LFCC(self.frame_lens[idx], self.frame_hops[idx], self.fft_n[idx], self.m_target_sr, self.lfcc_dim[idx], with_energy=True, max_freq=self.lfcc_max_freq)
            )

        self.m_frontend = torch_nn.ModuleList(self.m_frontend)
        self.m_transform = torch_nn.ModuleList(self.m_transform)
        self.m_before_pooling = torch_nn.ModuleList(self.m_before_pooling)
        self.m_output_act = torch_nn.ModuleList(self.m_output_act)

    def prepare_mean_std(self, in_dim, out_dim, args, data_mean_std=None):
        if data_mean_std is not None:
            in_m = torch.from_numpy(data_mean_std[0])
            in_s = torch.from_numpy(data_mean_std[1])
            out_m = torch.from_numpy(data_mean_std[2])
            out_s = torch.from_numpy(data_mean_std[3])
            return in_m, in_s, out_m, out_s
        else:
            return torch.zeros([in_dim]), torch.ones([in_dim]), torch.zeros([out_dim]), torch.ones([out_dim])




    # NOT USED BUT MANDATORY TP HAVE AS AN OPTION
    def normalize_input(self, x):
            """ normalizing the input data
            This is required for the Pytorch project, but not relevant to this code
            """
            return (x - self.input_mean) / self.input_std

    def normalize_target(self, y):
        """ normalizing the target data
        This is required for the Pytorch project, but not relevant to this code
        """
        return (y - self.output_mean) / self.output_std

    def denormalize_output(self, y):
        """ denormalizing the generated output from network
        This is required for the Pytorch project, but not relevant to this code
        """
        return y * self.output_std + self.output_mean





    def forward(self, x, fileinfo):
        filenames = [nii_seq_tk.parse_filename(y) for y in fileinfo]
        datalength = [nii_seq_tk.parse_length(y) for y in fileinfo]

        feature_vec = self._compute_embedding(x, datalength)
        scores = self._compute_score(feature_vec)

        target = self._get_target(filenames)
        target_vec = torch.tensor(target, device=x.device, dtype=scores.dtype)
        target_vec = target_vec.repeat(self.v_submodels)

        return [scores, target_vec, True]

    def _compute_embedding(self, x, datalength):
        batch_size = x.shape[0]
        output_emb = torch.zeros([batch_size * self.v_submodels, self.v_emd_dim], device=x.device, dtype=x.dtype)

        for idx, (trunc_len, m_trans, m_be_pool, m_output) in enumerate(zip(self.v_truncate_lens, self.m_transform, self.m_before_pooling, self.m_output_act)):
            x_sp_amp = self.m_frontend[idx](x.squeeze(-1)).unsqueeze(1)
            #print(f"After frontend transformation: {x_sp_amp.shape}")

            hidden_features = m_trans(x_sp_amp)
            #print(f"After m_transform: {hidden_features.shape}")

            hidden_features = hidden_features.permute(0, 2, 1, 3).contiguous()
            #print(f"After permute and contiguous: {hidden_features.shape}")

            frame_num = hidden_features.shape[1]
            hidden_features = hidden_features.view(batch_size, frame_num, -1)
            #print(f"After view reshape: {hidden_features.shape}")

            hidden_features_lstm = m_be_pool(hidden_features)
            #print(f"After BLSTM layers: {hidden_features_lstm.shape}")

            tmp_emb = m_output((hidden_features_lstm + hidden_features).mean(1))
            #print(f"After m_output (Linear): {tmp_emb.shape}")
            output_emb[idx * batch_size: (idx + 1) * batch_size] = tmp_emb
        return output_emb

    def _compute_score(self, feature_vec, inference=False):
        if inference:
            return feature_vec.squeeze(1)
        else:
            return torch.sigmoid(feature_vec).squeeze(1)

    def _get_target(self, filenames):
        return [self.protocol_parser[x] for x in filenames]

    def _get_target_eval(self, filenames):
        return [self.protocol_parser[x] if x in self.protocol_parser else -1 for x in filenames]

    def forward(self, x, fileinfo):

        #with torch.no_grad():
        #    vad_waveform = self.m_vad(x.squeeze(-1))
        #    vad_waveform = self.m_vad(torch.flip(vad_waveform, dims=[1]))
        #    if vad_waveform.shape[-1] > 0:
        #        x = torch.flip(vad_waveform, dims=[1]).unsqueeze(-1)
        #    else:
        #        pass
        filenames = [nii_seq_tk.parse_filename(y) for y in fileinfo]
        datalength = [nii_seq_tk.parse_length(y) for y in fileinfo]
        if self.training:
            
            feature_vec = self._compute_embedding(x, datalength)
            scores = self._compute_score(feature_vec)
            
            # target
            target = self._get_target(filenames)
            target_vec = torch.tensor(target, 
                                      device=x.device, dtype=scores.dtype)
            target_vec = target_vec.repeat(self.v_submodels)
            
            return [scores, target_vec, True]

        else:
            feature_vec = self._compute_embedding(x, datalength)
            scores = self._compute_score(feature_vec, True)
            
            target = self._get_target_eval(filenames)
            print("Output, %s, %d, %f" % (filenames[0], 
                                          target[0], scores.mean()))
            # don't write output score as a single file
            return None


class Loss():
    """ Wrapper to define loss function 
    """
    def __init__(self, args):
        """
        """
        self.m_loss = torch_nn.BCELoss()


    def compute(self, outputs, target):
        """ 
        """
        loss = self.m_loss(outputs[0], outputs[1])
        return loss

    
if __name__ == "__main__":
    print("Definition of model")