import torch 
import torch.nn as nn 
import numpy as np 
from util import *
from torchdiffeq import odeint

device = "cuda" if torch.cuda.is_available() else "cpu"


class HPCModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # neuron number
        self.num = config.num_hpc
        self.num_sen = self.config.OVC["N_dis"] * self.config.OVC["N_theta"] * self.config.env.loc_land.shape[0]

        # dynamical parameters
        self.tau = 1.  # The synaptic time constant
        self.tau_v = 5.
        self.k = 10 / self.num
        self.m = self.tau / self.tau_v

        # Learning parameters
        self.spike_num_exc = 100
        self.tau_W = 500
        self.tau_Sen = 100
        self.norm_sen_exc = 0.5
        self.lr_sen_exc = 10.
        self.lr_exc = 2.5 / self.num
        self.lr_back = self.lr_exc
        # variables
        # Neural state
        self.r = nn.Parameter(torch.zeros([self.num, ]).to(device))
        self.u = nn.Parameter(torch.zeros([self.num, ]).to(device))
        self.v = nn.Parameter(torch.zeros([self.num, ]).to(device))
        self.r_learn = torch.zeros([self.num, ]).to(device)  # Neuron waiting to learn
        self.E_total = torch.zeros([self.num, ]).to(device)
        # Conn matrix
        self.W_sen_back = nn.Parameter(torch.zeros([self.num_sen, self.num]).to(device))
        self.W_E2E = nn.Parameter(torch.zeros([self.num, self.num]).to(device))
        self.corr = torch.ones((self.num, self.num_sen)).to(device)
        self.W_sen = nn.Parameter(torch.randn((self.num, self.num_sen)).to(device) * 0.1)
        # self.W_sen = randomize_matrix(self.W_sen, percent_zeros=percentage)
        self.W_sen.data = normalize_rows(self.W_sen) * self.norm_sen_exc

    def dist(self, d):
        dis = torch.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        return dis

    def reset_state(self):
        self.r.data = torch.zeros(self.num, ).to(device)
        self.u.data = torch.zeros(self.num, ).to(device)
        self.v.data = torch.zeros(self.num, ).to(device)

    def conn_update_forward(self, r_sen):
        # 这里试图将HPC的发放和I_sen绑定在一起
        # 竞争通过的top_K完成
        self.r_learn = keep_top_n(self.r, self.spike_num_exc)
        r_learn = self.r_learn.reshape(-1, 1)  # r_learn就是HPC细胞的发放，选取topK个
        r_sen = r_sen.reshape(1, -1)  # r_sen是I_ovc, r_sen* W_sen后 shape与hpc_num一致
        corr_mat_sen = torch.matmul(r_learn, r_sen)
        # corr_mat_sen = torch.outer(r_learn, r_sen)
        # I_sen = torch.matmul(self.W_sen, r_sen)
        # corr_mat_sen = torch.outer(r_learn, I_sen)
        self.corr = corr_mat_sen
        # tau * dW/dt = r_learn * r_sen * W_sen * lr = r_learn * I_sen
        dW = corr_mat_sen * self.W_sen * self.lr_sen_exc / self.tau_Sen * self.config.dt
        # print("check dw", torch.max(corr_mat_sen), torch.max(self.r_learn), torch.max(r_sen))
        W_sen = self.W_sen + dW
        W_sen = torch.where(W_sen > 0., W_sen, 0.)
        self.W_sen.data = normalize_rows(W_sen) * self.norm_sen_exc

    def conn_update_rec(self, thres=3.5):
        r_exc = self.r.reshape(-1, 1)
        thres = torch.mean(r_exc)
        r_exc = torch.where(r_exc > thres, r_exc, 0)
        corr_mat_E2E = torch.matmul(r_exc, r_exc.transpose(1,0))
        u_rec = torch.matmul(self.W_E2E, self.r).reshape(-1, 1)
        dW_E = (self.lr_exc * corr_mat_E2E - self.W_E2E * (r_exc * u_rec)) / self.tau_W
        # print("check WE2E", torch.max(dW_E))
        W_E2E = self.W_E2E + dW_E
        self.W_E2E.data = torch.where(W_E2E > 0, W_E2E, 0)

    def conn_update_back(self, r_sen, thres=3.5):
        # Oja学习，第一项保证噪音干扰下的OVC输入和hpc发放趋于一致，所以这项可以跟踪W_sen，后面是为了正则化。总之是为了在噪音下追踪W_sen
        # 目前怀疑是两个权重占比差别太多了
        r_sen = r_sen + torch.randn((self.num_sen,)).to(device)*0.01
        r_sen = r_sen.reshape(-1, 1)
        r_exc = self.r.reshape(1, -1)
        thres = torch.mean(r_exc)
        r_exc = torch.where(r_exc > thres, r_exc, 0)
        r_sen = torch.where(r_sen > 0.3, r_sen, 0)
        corr_mat_sen = torch.matmul(r_sen, r_exc)
        # print("check sen back w", torch.max(corr_mat_sen))
        u_back = torch.matmul(self.W_sen_back, self.r).reshape(-1, 1)
        dW_sen_back = (self.lr_back * corr_mat_sen - self.W_sen_back * (r_sen * u_back)) / self.tau_W
        W_sen_back = self.W_sen_back + dW_sen_back
        self.W_sen_back.data = torch.where(W_sen_back > 0, W_sen_back, 0)

    def derivative(self, t, state):
        ue = state[0]
        ve = state[1]
        due = (-ue + self.E_total - self.v) / self.tau
        dve = (-ve + self.m * self.u) / self.tau_v
        # due = lambda ue, t, E_total: (-ue + E_total - self.v) / self.tau
        # dve = lambda ve, t: (-ve + self.m * self.u) / self.tau_v
        return torch.stack([due, dve])

    def update(self, I_mec, I_sen, I_OVCs, shared_t, train=0):
        # Calculate the feedforward input
        noise = torch.randn(self.num).to(device) * 0.1
        # Calculate the recurrent current
        I_rec = torch.matmul(self.W_E2E, self.r)
        # Calculate total input
        self.E_total = I_rec + I_mec + noise + I_sen
        # update neuron state
        state0 = torch.stack([self.u, self.v])
        solution = odeint(self.derivative, state0, shared_t)
        ue = solution[1, 0]  # 如果t是[n,]那么solution[n,2]一般n=1
        ve = solution[1, 1]
        # ue, ve = self.integral(self.u, self.v, share_t, E_total, self.config.dt)

        self.u.data = torch.where(ue > 0, ue, 0)
        self.v.data = ve
        r1 = torch.square(self.u)
        r2 = 1.0 + self.k * torch.sum(r1)
        self.r.data = r1 / r2
        # print("HPC r", self.r)

        if train == 1:
            self.conn_update_rec()  # from HPC to HPC
        elif train == 2:
            self.conn_update_forward(r_sen=I_OVCs)  # from LV to HPC
            self.conn_update_rec()  # from HPC to HPC
            self.conn_update_back(r_sen=I_OVCs)  # from HPC to LV

    def forward(self, train=0):
        # train=0, not learn
        # train=1, learn via Hebbian

        pass

    def perform_hebb(self,):
        pass  



if __name__ == '__main__':
    pass