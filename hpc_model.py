import torch 
import torch.nn as nn 
import numpy as np 


device = "cuda" if torch.cuda.is_available() else "cpu"


class HPCModel(nn.Module):

    def __init__(self,):
        pass 

    def forward(self, train=0):
        # train=0, not learn
        # train=1, learn via Hebbian

        pass

    def perform_hebb(self,):
        pass  









#