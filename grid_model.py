""".DS_Store

build the circuit to transform egocentric sensory inputs into allocentric inputs.

1. transform egocentric sensory inputs into egocentric object vector representation 
2. integrate egocenric object vector representation and 

"""

import torch 
import torch.nn as nn 
import numpy as np 


device = "cuda" if torch.cuda.is_available() else "cpu"


class GridModel(nn.Module):

    def __init__(self,):
        pass 

    def forward(self, train=0):
        # train=0, not learn
        # train=1, learn via Hebbian
        pass 
