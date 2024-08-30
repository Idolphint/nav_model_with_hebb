import time

import torch
import numpy as np
from util import *
from hd_obv_system import landmark_bearing, sens2egoOVC
device = "cuda" if torch.cuda.is_available() else "cpu"

class Coupled_Net_HHG(torch.nn.Module):
    def __init__(self, HPC_model, MEC_model_list, HD_net, config,
                 mec2hpc_stre=1., sen2hpc_stre=1., hpc2mec_stre=0., sen2HD_stre=0.):
        super().__init__()
        self.config = config
        self.HPC_model = HPC_model
        self.HD_net = HD_net
        self.MEC_model_list = torch.nn.ModuleList()   #MEC_model_list
        for each_model in MEC_model_list:
            self.MEC_model_list.append(each_model)
        self.num_module = config.num_mec_module

        self.num_sen = config.OVC["N_dis"] * config.OVC["N_theta"] * config.env.loc_land.shape[0]
        num_mec = MEC_model_list[0].num

        # initialize dynamical variables
        self.r_sen = torch.zeros(self.num_sen, )
        self.input_back = torch.zeros(self.num_sen, )
        self.u_mec_module = torch.zeros([num_mec, self.num_module])
        self.input_hpc_module = torch.zeros([num_mec, self.num_module])
        self.I_mec = torch.zeros(HPC_model.num, )  # MEC 输入给HPC的部分
        self.I_sen = torch.zeros(HPC_model.num, )
        self.mec2hpc_stre = mec2hpc_stre
        self.sen2hpc_stre = sen2hpc_stre
        self.hpc2mec_stre = hpc2mec_stre
        self.sen2HD_stre = sen2HD_stre

    def reset_state(self, loc, HD_truth):
        self.HPC_model.reset_state()
        for MEC_model in self.MEC_model_list:
            MEC_model.reset_state(loc)
        self.HD_net.reset_state(HD_truth)

    def update(self, velocity, angular_velocity, loc, HD_gt, shared_t, get_loc=0, get_view=0, get_HD=1, train=0):
        """
        :param velocity:
        :param angular_velocity:
        :param loc: 真实的loc
        :param HD_gt: 就是头朝向本身，并非角速度，一定注意！
        :param get_loc: =1代表使用真实的pos而不是积分得到的pos
        :param get_view: =1代表使用（部分）120°视野 且使用sen_back
        :param get_HD: =1代表使用真实的HD，而不是积分得到的HD
        :param train:
        :return:
        """
        bound_effect = 1
        # Update MEC states
        r_hpc = self.HPC_model.r
        I_mec = torch.zeros(self.HPC_model.num, ).to(device)
        t0 = time.time()
        for i in range(self.num_module):
            MEC_model = self.MEC_model_list[i]
            MEC_model.update(pos=loc, velocity=velocity, r_hpc=r_hpc, shared_t=shared_t,
                             hpc2mec_stre=self.hpc2mec_stre, train=train,
                             get_loc=get_loc)
            r_mec = MEC_model.r
            self.u_mec_module[:, i] = r_mec
            I_mec_module = torch.matmul(MEC_model.conn_out, r_mec - 0.5 * torch.max(r_mec))  # r_mec自然地分成一半正一半负
            I_mec += I_mec_module
            input_hpc = torch.matmul(MEC_model.conn_in, r_hpc) - 0.1  # 从hpc输入到MEC的量
            input_hpc = torch.where(input_hpc > 0, input_hpc, 0)
            self.input_hpc_module[:, i] = input_hpc
        print("MEC update using", time.time() - t0)
        t0 = time.time()
        I_mec = torch.where(I_mec > 0, I_mec, 0)
        # Update HD cells states
        r_lb = landmark_bearing(loc=loc, HD_vec=HD_gt, loc_land=self.config.env.global_land)
        HD_truth = angle_between_vectors(HD_gt, torch.Tensor([1, 0]).to(device))
        self.HD_net.update(angular_velocity=angular_velocity, HD=HD_truth, r_hpc=r_hpc, r_bearing=r_lb,
                           shared_t=shared_t, get_HD=get_HD,
                           sen2HD_stre=self.sen2HD_stre, train=train)
        HD_direction = self.HD_net.center
        print("HD update using", time.time()-t0)
        t0 = time.time()
        # Update Sensory inputs

        view_size = 2 / 3 * np.pi if get_view == 1 else 2 * np.pi
        I_OVC_1, _ = sens2egoOVC(config=self.config, loc_gt=loc, HD_gt=HD_gt, HD_signal=HD_direction,
                                 agent_view=view_size)
        I_OVCs = I_OVC_1

        # Feedback conn
        W_sen_back = self.HPC_model.W_sen_back
        self.input_back = torch.matmul(W_sen_back, r_hpc)
        r_sen = I_OVCs.flatten() + get_view * self.input_back
        r_sen = torch.where(r_sen > 1, 1, r_sen)

        self.r_sen = bound_effect * I_OVCs.flatten()
        self.I_mec = bound_effect * self.mec2hpc_stre * I_mec
        self.I_sen = torch.matmul(self.HPC_model.W_sen, r_sen) * self.sen2hpc_stre  # I_sen是经过HPC输入权重调制的
        # Update Hippocampus states
        self.HPC_model.update(I_mec=self.I_mec, I_sen=self.I_sen, I_OVCs=I_OVCs.flatten(), shared_t=shared_t, train=train)
        print("HPC update using", time.time()-t0)
        t0 = time.time()