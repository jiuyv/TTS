import torch
import torch.nn as nn
import torch.nn.functional as F
from tacotron2.layers import ConvNorm

class Postnet(nn.Module):
    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size = hparams.postnet_kernel_size, stride = 1,
                         padding = int((hparams.postnet_kernel_size - 1) / 2),
                         dilation = 1, w_init_gain = 'tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size = hparams.postnet_kernel_size, stride = 1,
                             padding = int((hparams.postnet_kernel_size - 1) / 2),
                             dilation = 1, w_init_gain = 'tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size = hparams.postnet_kernel_size, stride = 1,
                         padding = int((hparams.postnet_kernel_size - 1) / 2),
                         dilation = 1, w_init_gain = 'linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x