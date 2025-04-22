import torch
import torch.nn as nn
from .base_module import *
from thop import profile


class S2SHSID(nn.Module):
    def __init__(self, opt):
        super(S2SHSID, self).__init__()
        self.encoder = Encoder(input_dim=1,hidden_dim=opt['hidden_dim'],local_range = opt['local_range'],region_num =opt['region_num'],cuda=opt['cpu'],device=opt['gpu_ids'])
        self.decoder = Decoder(hidden_dim=opt['hidden_dim'],local_range = opt['local_range'],region_num =opt['region_num'])

    def forward(self, x):
        '''
        :param x: [n,b,c,h,w]
        :return: out: [n,b,c,h,w]
        '''
        # encoder
        encoder_out = self.encoder(x)
        # decoder
        out = self.decoder(x, encoder_out)
        return out
