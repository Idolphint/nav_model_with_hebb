"""
Construct a HD-OVC circuit model that converts egocentric sensory inputs into allocentric representation.

1. Convert egocentric sensory inputs into a egocentric object vector representation.
2. Merge the egocentric object vector representation with the head direction representation to generate an allocentric object vector representation.

"""
import torch 
import torch.nn as nn 
import numpy as np 


device = "cuda" if torch.cuda.is_available() else "cpu"


class HDModel(nn.Module):

    def __init__(self,):
        pass 

    def forward(self, train=0):
        # train=0, not learn
        # train=1, learn via Hebbian
        pass 


def sens2egoOVC():
    pass 


class HdOVCModel(nn.Module):

    def __init__(self,):
        pass 

    def forward(self, train=0):
        # train=0, not learn
        # train=1, learn via Hebbian

        pass

    def perform_hebb(self,):
        pass  









#