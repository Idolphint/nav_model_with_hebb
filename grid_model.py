""".DS_Store

build the circuit to transform egocentric sensory inputs into allocentric inputs.

1. transform egocentric sensory inputs into egocentric object vector representation 
2. integrate egocenric object vector representation and 

"""

import torch
import torch.nn as nn
import numpy as np
from util import *
from torchdiffeq import odeint

device = "cuda" if torch.cuda.is_available() else "cpu"


class GridModel(nn.Module):

    def __init__(self, ratio, angle, config):
        super().__init__()
        self.config = config
        # grid module parameters
        self.ratio = ratio
        self.pos2phase_constant = ratio * 2 * np.pi / config.env.diameter
        self.angle = angle
        # self.module_strength = strength

        # dynamics parameters
        self.tau = 1.  # The synaptic time constant
        self.tau_v = 10.
        self.tau_W_out = 50.
        self.tau_W_in = 500.
        self.spike_num = 100
        self.tau_e = 1000

        # neuro num
        self.num_mec_x = config.num_mec  # number of excitatory neurons for x dimension
        self.num_mec_y = config.num_mec  # number of excitatory neurons for y dimension
        self.num = self.num_mec_x * self.num_mec_y
        self.num_hpc = config.num_hpc

        # spike control parameter
        self.k = 1 / self.num * 20  # Degree of the rescaled inhibition
        self.gauss_width = np.pi / 2
        self.gauss_height = 10
        self.conn_matrix_max = 1. / self.num * 20
        self.m = 1. * self.tau / self.tau_v

        # Learning parameters
        self.lr_in = 2.5 / config.num_hpc
        self.lr_out = self.lr_in
        self.norm_fac = 30

        # feature space
        self.x = np.linspace(-np.pi, np.pi, self.num_mec_x, endpoint=False)
        self.y = np.linspace(-np.pi, np.pi, self.num_mec_y, endpoint=False)
        x_grid, y_grid = np.meshgrid(self.x, self.y)
        self.x_grid = torch.Tensor(x_grid).flatten().to(device)
        self.y_grid = torch.Tensor(y_grid).flatten().to(device)
        self.value_grid = torch.stack([self.x_grid, self.y_grid]).T
        self.neural_density = self.num / (4 * np.pi ** 2)
        self.dxy = 1. / self.neural_density
        self.coor_transform = torch.Tensor([[1, -1 / np.sqrt(3)], [0, 2 / np.sqrt(3)]]).to(device)
        self.rot = torch.Tensor(
            [[np.cos(self.angle), -np.sin(self.angle)], [np.sin(self.angle), np.cos(self.angle)]]).to(device)

        # initialize conn matrix
        self.conn_mat = self.make_conn()
        conn_out = self.make_conn_out()
        self.conn_out = nn.Parameter(self.norm_fac * normalize_rows(conn_out))
        self.conn_in = nn.Parameter(torch.zeros([self.num, self.num_hpc]).to(device))

        # initialize dynamical variables
        self.r = nn.Parameter(torch.zeros(self.num).to(device))
        self.u = nn.Parameter(torch.zeros(self.num).to(device))
        self.v = nn.Parameter(torch.zeros(self.num).to(device))
        self.input = torch.zeros(self.num).to(device)
        self.input_total = torch.zeros(self.num).to(device)
        self.motion_input = torch.zeros(self.num).to(device)
        self.center = torch.zeros(2, ).to(device)
        self.centerI = torch.zeros(2, ).to(device)
        self.center_ideal = torch.zeros(2, ).to(device)
        self.phase_estimate = torch.zeros(2, ).to(device)

    def dist(self, d):
        d = period_bound(d)
        delta_x = d[:, 0] + d[:, 1] / 2
        delta_y = d[:, 1] * np.sqrt(3) / 2
        dis = torch.sqrt(delta_x ** 2 + delta_y ** 2)
        return dis

    def make_conn_out(self):
        matrix = torch.randn((self.num_hpc, self.num)).to(device) * 0.01
        random_indices = torch.randint(self.num, size=(self.num_hpc,)).to(device)
        matrix[torch.LongTensor(np.arange(self.num_hpc)), random_indices] = 1
        return matrix

    def make_conn(self):
        # 根据value grid任意两点间的距离得到的矩阵
        value_grid = self.value_grid.unsqueeze(0)
        value_gridT = value_grid.transpose(1, 0)
        d = torch.sum(torch.square(value_gridT - value_grid), dim=-1)
        matrix = self.conn_matrix_max * torch.exp(-0.5 * d / (self.gauss_width ** 2)) / (
                    np.sqrt(2 * np.pi) * self.gauss_width)
        return matrix

    def Postophase(self, pos):
        # 将pos旋转self.angle角，就得到了在本grid cell坐标系下的坐标，*pi / radius = 相位
        Loc = torch.matmul(self.rot, pos) * self.pos2phase_constant  # （旋转self.angle角 - 0） * 放缩因子
        phase = torch.matmul(self.coor_transform, Loc) + np.pi  # 坐标变换
        phase_x = torch.fmod(phase[0], 2 * np.pi) - np.pi
        phase_y = torch.fmod(phase[1], 2 * np.pi) - np.pi
        Phase = torch.Tensor([phase_x, phase_y]).to(device)
        return Phase

    def get_stimulus_by_pos(self, pos):
        assert len(pos.shape) == 1 and pos.shape[0] == 2
        Phase = self.Postophase(pos)
        d = self.dist(Phase - self.value_grid)
        return self.gauss_height * torch.exp(-0.25 * torch.square(d / self.gauss_width))

    def get_stimulus_by_motion(self, velocity):
        # integrate self motion
        noise_v = torch.randn(1) * 0.
        v_rot = torch.matmul(self.rot, velocity)
        v_phase = torch.matmul(self.coor_transform, v_rot * self.pos2phase_constant)
        dis_phase = period_bound(self.center - self.phase_estimate)
        pos_e = self.phase_estimate + (dis_phase / self.tau_e + v_phase) * self.config.dt
        # 新版本中不用积分器了，因为可能出错了？
        # pos_e = self.path_integral(self.phase_estimate, bp.share['t'], v_phase, bm.dt)
        self.phase_estimate = period_bound(pos_e)
        d = self.dist(self.phase_estimate - self.value_grid)
        # d = self.dist(self.center_ideal - self.value_grid)
        return self.gauss_height * torch.exp(-0.25 * torch.square(d / self.gauss_width))

    def get_center(self):
        exppos_x = torch.exp(1j * self.x_grid)
        exppos_y = torch.exp(1j * self.y_grid)
        r = torch.where(self.r > torch.max(self.r) * 0.1, self.r, 0)
        self.center[0] = torch.angle(torch.sum(exppos_x * r))
        self.center[1] = torch.angle(torch.sum(exppos_y * r))
        self.centerI[0] = torch.angle(torch.sum(exppos_x * self.input))
        self.centerI[1] = torch.angle(torch.sum(exppos_y * self.input))

    def derivative(self, t, state):
        u = state[0]
        v = state[1]
        du = (-u + self.input_total - self.v) / self.tau
        dv = (-v + self.m * self.u) / self.tau_v
        # du = lambda u, t, Irec: (-u + Irec + self.input_total - self.v) / self.tau
        # dv = lambda v, t: (-v + self.m * self.u) / self.tau_v
        # return bp.JointEq([du, dv])

        return torch.stack([du, dv])

    def conn_out_update(self, r_learn_hpc):
        r_learn = r_learn_hpc  # .reshape(-1, 1)
        r_grid = self.r  # .reshape(-1, 1)
        corr_out = torch.outer(r_learn, r_grid)
        conn_out = self.conn_out + (self.lr_out * corr_out * self.conn_out) / self.tau_W_out * self.config.dt
        conn_out = torch.where(conn_out > -0.1, conn_out, -0.1)
        # TODO 或需要测试ratio更好还是1.5~11的strength更好
        self.conn_out.data = self.norm_fac * normalize_rows(conn_out) * np.exp(-self.ratio ** 2 / 5 ** 2)

    def conn_in_update(self, r_hpc, thres=3.5):
        r_hpc = r_hpc.reshape(-1, 1)
        r_grid = self.r.reshape(-1, 1)
        r_hpc = torch.where(r_hpc > thres, r_hpc, 0)
        r_grid = torch.where(r_grid > 0.08, r_grid, 0)
        input_hpc = torch.matmul(self.conn_in, r_hpc)
        corr_in = torch.matmul(r_grid, r_hpc.transpose(1, 0))  # op as outer
        # TODO shape似乎有错误
        conn_in = self.conn_in + (
                    self.lr_in * corr_in - self.conn_in * (r_grid * input_hpc)) / self.tau_W_in * self.config.dt
        self.conn_in.data = torch.where(conn_in > 0, conn_in, 0)

    def reset_state(self, loc):
        self.r.data = torch.zeros(self.num).to(device)
        self.u.data = torch.zeros(self.num).to(device)
        self.v.data = torch.zeros(self.num).to(device)
        self.input = torch.zeros(self.num).to(device)
        self.input_total = torch.zeros(self.num).to(device)
        # self.center = 2 * bm.pi * bm.random.rand(2) - bm.pi
        Phase = self.Postophase(loc)  # center 是相位的确切中心
        self.phase_estimate = Phase
        self.center = Phase
        self.center_ideal = Phase

    def sampling_by_pos(self, r_hpc, pos, shared_t, noise_stre=0.01):
        noise = torch.randn((self.num,)).to(device) * noise_stre
        self.motion_input = self.get_stimulus_by_pos(pos) + noise
        input_hpc = torch.matmul(self.conn_in, r_hpc)
        self.input = self.motion_input + input_hpc
        Irec = torch.matmul(self.conn_mat, self.r)
        self.input_total = self.input + Irec
        # Update neural state
        state0 = torch.stack([self.u, self.v])
        solution = odeint(self.derivative, state0, shared_t)
        u = solution[1, 0]  # 如果t是[n,]那么solution[n,2]一般n=1
        v = solution[1, 1]
        self.u.data = torch.where(u > 0, u, 0)
        self.v.data = v
        r1 = torch.square(self.u)
        r2 = 1.0 + self.k * torch.sum(r1)
        self.r.data = r1 / r2
        self.get_center()

    def update(self, pos, velocity, r_hpc, shared_t, hpc2mec_stre=0., train=0, get_loc=1):
        self.get_center()
        # if self.ratio < 2:
        #     print("grid center of ", self.ratio, self.center)
        v_rot = torch.matmul(self.rot, velocity)
        v_phase = torch.matmul(self.coor_transform, v_rot * self.pos2phase_constant)
        center_i = self.center_ideal + v_phase * self.config.dt
        self.center_ideal = period_bound(center_i)  # Ideal phase using path integration
        self.centerI = self.Postophase(pos)  # Ideal phase using mapping function

        input_pos = self.get_stimulus_by_pos(pos)
        input_motion = self.get_stimulus_by_motion(velocity)
        if get_loc == 1:
            self.input = input_pos
        else:
            self.input = input_motion
        input_hpc = torch.matmul(self.conn_in, r_hpc) - 0.1
        input_hpc = torch.where(input_hpc > 0, input_hpc, 0)

        self.input += input_hpc * hpc2mec_stre
        Irec = torch.matmul(self.conn_mat, self.r)
        # Update neural state
        self.input_total = self.input + Irec
        # Update neural state
        state0 = torch.stack([self.u, self.v])
        solution = odeint(self.derivative, state0, shared_t)
        u = solution[1, 0]  # 如果t是[n,]那么solution[n,2]一般n=1
        v = solution[1, 1]
        # u, v = self.integral(self.u, self.v, bp.share['t'], Irec, bm.dt)
        self.u.data = torch.where(u > 0, u, 0)
        self.v.data = v
        r1 = torch.square(self.u)
        r2 = 1.0 + self.k * torch.sum(r1)
        self.r.data = r1 / r2
        r_learn_hpc = keep_top_n(r_hpc, self.spike_num)
        if train > 0:
            self.conn_out_update(r_learn_hpc=r_learn_hpc)
            self.conn_in_update(r_hpc=r_hpc)

    def forward(self, train=0):
        # train=0, not learn
        # train=1, learn via Hebbian
        pass 