from os import path as osp
from collections import OrderedDict
import yaml
import argparse
import random
import numpy as np

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def  parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default='train', help='train or test')
    parser.add_argument('-method', type=str, default='mamba', help='which method to use')
    parser.add_argument('-type', type=str, default='None', help='which method\'s config to use')
    
    args = parser.parse_args()
    
    if args.type == "None":
        opt_path = '/home/jiahua/s2s_release/options/' + args.method + '_hsid.yml'
    else:
        opt_path = '/home/jiahua/s2s_release/options/' + args.method + '_' + args.type + '_hsid.yml'
    
    # parse yml to dict
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])

    opt['mode'] = args.mode

    return opt
