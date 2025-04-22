from .competing_methods import *

""" Models
"""

def s2s(opt):
    net = S2SHSID(opt)
    net.use_2dconv = True
    net.bandwise = False
    return net