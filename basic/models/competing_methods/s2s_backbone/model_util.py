import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import os

def make_layer(block, num_layers, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_layers):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)

def get_adjacent(input_fea, pos, range):
    c = input_fea.shape[1]
    if pos+range <= c:
        adjacent = input_fea[:,pos:pos+range]
    else:
        adjacent = input_fea[:,c-range:c]
    return adjacent

def get_back(input_fea, pos, range):
    c = len(input_fea)
    if pos <= range:
        back = input_fea[0: pos]
    elif pos > range and pos < c - range:
        back = input_fea[pos - range: pos]
    else:
        back = input_fea[c - 2 * range - 1: pos]
    back = torch.stack([torch.as_tensor(item) for item in back],dim=1)
    return back

def get_forth(input_fea, pos, range):
    c = len(input_fea)
    if pos == c-1:
        return None
    if pos <= range:
        forth = input_fea[ pos + 1: 2 * range + 1]
    elif pos > range and pos < c - range:
        forth = input_fea[pos + 1: pos + range + 1]
    else:
        forth = input_fea[pos + 1: c]
    forth = torch.stack([torch.as_tensor(item) for item in forth],dim=1)
    return forth