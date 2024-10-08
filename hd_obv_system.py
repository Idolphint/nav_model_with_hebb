"""
Construct a HD-OVC circuit model that converts egocentric sensory inputs into allocentric representation.

1. Convert egocentric sensory inputs into a egocentric object vector representation.
2. Merge the egocentric object vector representation with the head direction representation to generate an allocentric object vector representation.
通过维护一个巨大的表格hpc_num, ego_hD_num, allo_HD_num, 记录在具体的地点当ego_HD为某个值时，对应的allo_HD该是什么？
"""
import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
from util import *
from model_config import ConfigParam, Env

device = "cuda" if torch.cuda.is_available() else "cpu"


class HdOVCModel(nn.Module):
    def __init__(self, config, sample_place_num=2000):
        super().__init__()
        self.config = config
        self.num = config.num_HD
        # parameters
        self.tau = 1.  # The synaptic time constant
        self.tau_v = 10.
        self.tau_conn = 10.
        self.tau_e = 1000.  # ? 似乎是MEC相关的参数
        self.tau_W = self.tau_conn * 1000
        self.k = 0.8 / self.num * 20  # Degree of the rescaled inhibition
        self.J0 = 1 / self.num * 20  # maximum connection value
        self.m = 0.84 * self.tau / self.tau_v
        self.norm_fac = 10
        self.norm_fac_HD = 1
        self.gauss_width = np.pi / 4
        self.gauss_height = 3
        self.lr = 1e4

        # config for ovc
        self.ovc_fr = 1.0

        # cell nums
        self.spike_num = 4
        self.sample_place_cell_num = sample_place_num
        self.num_land = 30  # landmark bearing cells of one module

        # feature space
        self.x = torch.linspace(-np.pi, np.pi, self.num + 1)[:-1].to(device)
        self.neural_density = self.num / (2 * np.pi)
        self.dx = 2 * np.pi / self.num

        # connection matrix
        conn_mat = self.make_conn()
        self.conn_fft = torch.fft.fft(conn_mat)  # 1d fft
        self.conn_bearing = nn.Parameter(torch.zeros([self.sample_place_cell_num, self.num, self.num_land]).to(device) + torch.abs(
            torch.randn((self.sample_place_cell_num, self.num, self.num_land)) * 0.1).to(device))

        # neuron state variables
        self.r = nn.Parameter(torch.zeros(self.num).to(device))
        self.u = nn.Parameter(torch.zeros(self.num).to(device))  # head direction cell
        self.v = nn.Parameter(torch.zeros(self.num).to(device))
        self.input_bearing = torch.zeros(self.num).to(device)
        self.input_total = torch.zeros(self.num).to(device)
        # self.u_place = torch.zeros(self.sample_place_cell_num).to(device)  # place cell
        self.r_place = torch.zeros(self.sample_place_cell_num).to(device)  # place cell
        # self.u_bearing = torch.zeros([self.num_land]).to(device)  # conjunctive cell
        self.index = torch.arange(self.sample_place_cell_num).to(device)

        # other variables
        self.motion_input = torch.zeros(self.num).to(device)
        self.center = torch.zeros(1).to(device)
        self.center_ideal = torch.zeros(1).to(device)
        self.pos_estimate = torch.zeros(1).to(device)

        # 积分器使用torchdiffeq中的

    def dist(self, d):
        d = period_bound(d)
        d = torch.where(d > np.pi, d - 2 * np.pi, d)
        return d

    def make_conn(self):
        d = self.dist(torch.abs(self.x[0] - self.x))
        Jxx = self.J0 * torch.exp(-0.5 * torch.square(d / self.gauss_width)) / (2 * np.pi * self.gauss_width ** 2)
        return Jxx

    def derivative(self, t, state):
        u = state[0]
        v = state[1]
        du = (-u + self.input_total - self.v) / self.tau
        dv = (-v + self.m * self.u) / self.tau_v
        return torch.stack([du, dv])

    def get_center(self, r, x):  # 复数加权角度，可很好处理周期数据
        exppos = torch.exp(1j * x)
        center = torch.angle(torch.sum(exppos * r))
        return center.reshape(-1, )

    def self_motion_integrate(self, velocity, noise_stre=0.):  # 手动积分计算头朝向
        # TODO 角速度量级差了一点
        # 根据角速度定积分计算下一个可能的头朝向
        # integrate self motion
        noise_v = noise_stre * torch.randn(1).to(device)
        pos_e = self.pos_estimate + (
                    period_bound(self.center - self.pos_estimate) / self.tau_e + velocity + noise_v) * self.config.dt
        self.pos_estimate = period_bound(pos_e)
        center_i = self.center_ideal + velocity * self.config.dt
        # self.pos_estimate = self.period_bound(pos_e)
        self.center_ideal = period_bound(center_i)
        self.motion_input = self.gauss_height * torch.exp(
            -0.25 * torch.square(self.dist(self.x - self.pos_estimate) / self.gauss_width))

    def input_HD(self, HD):  # 真实头朝向在n个方向分类上的响应
        # TODO 为什么角度是反着的？
        # integrate self motion
        return self.gauss_height * torch.exp(-0.25 * torch.square(self.dist(self.x - HD) / self.gauss_width))

    def check1(self):  # 证明从encode HD到decode没有问题
        hd = 0
        for i in range(100):
            hd += 0.1
            hd = hd if hd < np.pi else hd - np.pi * 2
            r = self.input_HD(hd)
            center = self.get_center(r, self.x)
            print("ideal a, cal a=", hd, center)

    def conn_update(self, r_bearing, index):
        # 让r_bearing(相对固定global landmark的角度)和学习到的量保持一致
        conn_bearing_update = self.conn_bearing[index]  # Activate only one group
        r_hd = self.r * (1 + torch.abs(torch.randn((self.num,)).to(device) * 0.01))
        ####### Competitive Hebbian learning
        r_learn_hd = keep_top_n(r_hd, self.spike_num)
        r_bearing = r_bearing + torch.abs(torch.randn((self.num_land,)).to(device) * 0.01)
        corr_bearing = torch.outer(r_learn_hd, r_bearing)
        d_bearing = (self.lr * corr_bearing * conn_bearing_update) / self.tau_conn * self.config.dt
        conn_bearing = conn_bearing_update + d_bearing
        self.conn_bearing.data[index] = self.norm_fac_HD * normalize_rows(conn_bearing)

    def reset_state(self, HD_truth):
        self.r.data = torch.zeros(self.num).to(device)
        self.u.data = torch.zeros(self.num).to(device)  # head direction cell
        self.v.data = torch.zeros(self.num).to(device)
        self.motion_input = torch.zeros(self.num).to(device)
        self.center = torch.tensor(HD_truth).to(device)
        self.center_ideal = torch.zeros(1).to(device)
        self.pos_estimate = torch.zeros(1).to(device)

    def update(self, angular_velocity, HD, r_hpc, r_bearing, shared_t, sen2HD_stre, get_HD=0, noise_stre=0., train=0):
        """
        :param angular_velocity:
        :param HD:
        :param r_hpc: 难道是被排序的最高相应的hpc吗，否则怎么能只用前2000个呢？
        :param r_bearing: 相对global lndmark的角度[30]
        :param shared_t:
        :param sen2HD_stre:
        :param get_HD:
        :param noise_stre:
        :param train:
        :return:
        """
        # Calculate all inputs
        # Calculate path-integrate input
        self.center = self.get_center(r=self.r, x=self.x)  # 适用所有头朝向细胞计算出的波包中心,x代表每一位实际指代的角度是谁，r代表每一位的实际发放情况
        # print("HD get center", self.center)
        self.self_motion_integrate(angular_velocity, noise_stre)
        input_HD = self.input_HD(HD)  # 计算head direction映射到128维分量的值,这里是精确值
        if get_HD == 1:
            Iext = input_HD
        else:
            Iext = self.motion_input
        # print("check Iext", Iext)
        # Calculate sensory input
        self.r_place = r_hpc[:self.sample_place_cell_num]
        index = torch.argmax(self.r_place)
        normalize_r = torch.where(self.r_place == 0, 0, self.r_place / torch.sum(self.r_place))
        # r_bearing相当于真实全局HD，属于用来修正的量？
        input_bearing_all = torch.matmul(self.conn_bearing, r_bearing)
        self.input_bearing = torch.sum(input_bearing_all * normalize_r[:, np.newaxis], dim=0)
        # print(("check hpc and hd mix input bearing", self.r_place, self.input_bearing))

        # Calculate input
        r_fft = torch.fft.fft(self.r)
        Irec = torch.real(torch.fft.ifft(r_fft * self.conn_fft))  # ==self.r * self.conn 获得自循环输入
        # print("check Irec", Irec)
        # 当sen2HD为1时才会将global HD加入计算
        self.input_total = Iext + Irec + self.input_bearing * sen2HD_stre  # + torch.abs(torch.random.normal(0,0.01,(self.num,)))
        # Update neural state
        state0 = torch.stack([self.u, self.v], dim=0)
        solution = odeint(self.derivative, state0, shared_t)  # 此处积分应当是不为0
        # print("after odeint, solution", self.u, solution[1]) # time, 2, 128
        u = solution[1, 0]  # 如果t是[n,]那么solution[n,2]一般n=1
        v = solution[1, 1]
        # u, v = self.integral(self.u, self.v, bp.share.load('t'), input_total, self.config.dt)
        self.u.data = torch.where(u > 0, u, 0)
        self.v.data = v
        r1 = torch.square(self.u)
        r2 = 1.0 + self.k * torch.sum(r1)
        self.r.data = r1 / r2
        # print("check after odient", self.r)
        if train == 2:
            # self.conn_update(r_bearing, top_n_indices)
            self.conn_update(r_bearing, index)  # 只更新最好的？


def landmark_bearing(loc, loc_land, a_bearing=torch.pi / 4, num=30, HD_vec=torch.Tensor([0, 1])):
    # 头朝向细胞以环境外的landmark为标准，计算30等分的头朝向的响应程度
    # 使用环境外的新的landmark计算夹角，这样与HD夹角不会是0
    # if isinstance(loc, torch.Tensor):
    #     loc = loc.detach().cpu().numpy()
    #     HD_vec = HD_vec.detach().cpu().numpy()
    fr_Ampl = 1.
    theta_l = torch.linspace(-torch.pi, torch.pi, num+1)[:-1].to(device)
    loc_land = torch.Tensor(loc_land).to(device)
    # vec_land = loc_land[0,:]-loc
    vec_land = loc_land - loc
    angle = angle_between_vectors(HD_vec, vec_land)
    dis = torch.abs(angle - theta_l)
    dis = torch.Tensor(torch.where(dis > torch.pi, 2 * torch.pi - dis, dis)).to(device)

    I = fr_Ampl * torch.exp(- (dis ** 2) / (2 * a_bearing ** 2))
    return I.flatten()


def sens2egoOVC(config, loc_gt, HD_gt, HD_signal, agent_view=2 / 3 * torch.pi):
    """
    该函数与类中的函数作用一致，暂时保留
    :param config: 记录了环境信息，landmark信息等
    :param loc_gt: 当前agent的真实位置
    :param HD_gt: 当前Agent的真实头朝向
    :param HD_signal: HD_net的发放中心
    :param agent_view: agent视野，默认120°
    :return: agent根据观察到的信息对landmark的响应
    可以考虑对探测到的object产生高斯响应并复制多份进行旋转和平移
    """
    fr_Ampl = 1.0
    loc_land = torch.Tensor(config.env.loc_land).to(device)
    land_num = loc_land.shape[0]
    I_allo = torch.zeros((land_num, config.OVC["N_dis"], config.OVC["N_theta"])).to(device)
    I_ego = torch.zeros((land_num, config.OVC["N_dis"], config.OVC["N_theta"])).to(device)

    dis_l = torch.linspace(0.1, config.env.diameter / 2 - 0.1, config.OVC["N_dis"]).to(device)
    theta_l = torch.linspace(-np.pi, np.pi, config.OVC["N_theta"]+1)[:-1].to(device)
    HD_gt_angle = angle_between_vectors(HD_gt, torch.Tensor([1, 0]).to(device))

    theta_ego = theta_l + HD_gt_angle  # 相对自身当前朝向的360度
    theta_e = period_bound(theta_ego)

    # ego center的角度相对与该点自身相对global_landmark的角度 = 相对global_landmark的角度
    theta_allo = theta_l + HD_gt_angle - HD_signal  # 相对自身的朝向 - 自身相对某个世界的朝向 = allo_theta
    theta_a = period_bound(theta_allo)

    [Theta_Allo, Dis] = torch.meshgrid(theta_a, dis_l)
    [Theta_Ego, Dis] = torch.meshgrid(theta_e, dis_l)

    for i in range(land_num):
        angle = angle_between_vectors(HD_gt, loc_land[i, :] - loc_gt)  # HD_vec难道不是角速度嘛，还是根据线速度算出来的角度
        is_in_view = torch.where(torch.abs(angle) <= agent_view / 2, 1, 0)
        target_x_allo = loc_land[i, 0] + Dis * torch.cos(Theta_Allo)  # 距离loc_land指定距离的点坐标是什么，用x,y表示，以gloabl出发点的方向为0方向
        target_y_allo = loc_land[i, 1] + Dis * torch.sin(Theta_Allo)
        dis_allo = torch.sqrt((loc_gt[0] - target_x_allo) ** 2 + (loc_gt[1] - target_y_allo) ** 2)
        I_allo[i] = fr_Ampl * torch.exp(- (dis_allo ** 2) / (
                    2 * (config.OVC["gauss_width"] * config.env.diameter) ** 2)) * is_in_view  # 在global方向下排列的细胞对当前位置的响应

        target_x_ego = loc_land[i, 0] + Dis * torch.cos(Theta_Ego)
        target_y_ego = loc_land[i, 1] + Dis * torch.sin(Theta_Ego)
        dis_ego = torch.sqrt((loc_gt[0] - target_x_ego) ** 2 + (loc_gt[1] - target_y_ego) ** 2)
        # I_ego是以自身朝向为极坐标，距离landmark等距排布的cell，对agent位置产生的高斯响应
        I_ego[i] = fr_Ampl * torch.exp(
            - (dis_ego ** 2) / (2 * (config.OVC["gauss_width"] * config.env.diameter) ** 2)) * is_in_view

    # I_allo = torch.Tensor(I_allo).to(device)
    # I_ego = torch.Tensor(I_ego).to(device)
    # 确实可能存在盲区，导致ovc的输入全为0
    return I_allo, I_ego


if __name__ == '__main__':
    # 当不使用HPC输入时，HD预测正确
    env = Env()
    config = ConfigParam(env)
    HD_ovc_model = HdOVCModel(config, 1000)
    gt_HD = 0
    ang_vec = 0.2
    for i in range(100):
        loc_gt = np.array([i * 0.01, i * 0.01])
        gt_HD += ang_vec * config.dt
        gt_HD = gt_HD - 2 * np.pi if gt_HD >= np.pi else gt_HD
        gt_HD_vec = np.array([np.cos(gt_HD), np.sin(gt_HD)])
        r_hpc = torch.randn(1000).to(device)  # 先用随机值代表hpc的发放
        r_lb = landmark_bearing(loc=loc_gt, HD_vec=gt_HD_vec, loc_land=env.global_land)
        HD_ovc_model.update(angular_velocity=ang_vec, HD=gt_HD, r_hpc=r_hpc, r_bearing=r_lb,
                            shared_t=torch.Tensor([i * config.dt, (i + 1) * config.dt]), get_HD=0,
                            sen2HD_stre=0.0, train=2)

        I_allo, I_ego = sens2egoOVC(config, loc_gt, gt_HD_vec, HD_ovc_model.center)
        print("check HD model center and truth HD", HD_ovc_model.center, gt_HD)
