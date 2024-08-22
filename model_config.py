import numpy as np


class Env:
    def __init__(self):
        self.loc_land = np.array([
            [0.3, 0.3],
            [0.3, 0.6],
            [0.6, 0.3],
            [0.6, 0.6]
        ])  # shape = n_land * 2 记录每个landmark的坐标
        self.global_land = np.array([1.2, 1.1])
        self.diameter = 1


class ConfigParam:
    def __init__(self, env: Env):
        self.env = env
        self.OVC = {  # 描述OVC的关键参数
            "N_dis": 10,  # 每个landamrk有多少个描述距离的细胞，将N_dis平埔到整个环境中
            "N_theta": 10,  # 每个landmark有多少个描述角度的细胞，N_theta平铺360度
            "gauss_width": 0.07,   # 每单位宽度的env需要对应OVC的响应域
        }
        self.dt = 0.1

