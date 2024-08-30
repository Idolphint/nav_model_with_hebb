import numpy as np
from util import *

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

    def trajectory_train(self, lapnum=1, v_HD=np.pi/20, v_abs=0.01, dy=0.01, HD_init=0, dt=1):
        traj_use_diameter = 0.99  # 必须是奇数
        current_x, current_y = 0, 0  # agent的初始位置
        current_HD = HD_init
        positions = []
        head_direction = []
        angular_velocities = []
        # 生成轨迹
        for i in range(lapnum):
            while current_y <= traj_use_diameter:
                while current_x <= traj_use_diameter - v_abs * dt: # x+
                    positions.append([current_x, current_y])
                    # velocities.append([v_abs, 0.0])  # 在x方向上移动，vx=v_abs，vy=0
                    head_direction.append([np.cos(current_HD), np.sin(current_HD)])
                    angular_velocities.append(v_HD)
                    current_x += v_abs * dt
                    current_HD += v_HD * dt
                hist_y = current_y
                # print("check how stop", hist_y, dy, traj_use_diameter)
                if hist_y + dy <= traj_use_diameter:
                    while current_y <= hist_y + dy:  #
                        positions.append([current_x, current_y])
                        # velocities.append([0.0, v_abs])  # 在y方向上移动，vx=0，vy=v_abs
                        head_direction.append([np.cos(current_HD), np.sin(current_HD)])
                        angular_velocities.append(v_HD)
                        # 向上移动delta y
                        current_y += v_abs * dt
                        current_HD += v_HD * dt
                    else:
                        current_y = hist_y + dy
                else:
                    break

                # 反方向移动直到碰到左侧边界
                while current_x >= v_abs * dt:
                    positions.append([current_x, current_y])
                    # velocities.append([-v_abs, 0.0])  # 在x方向上移动，vx=-v_abs，vy=0
                    head_direction.append([np.cos(current_HD), np.sin(current_HD)])
                    angular_velocities.append(v_HD)
                    current_x -= v_abs * dt
                    current_HD += v_HD * dt
                hist_y = current_y
                # print("this time end x?", hist_y, dy, traj_use_diameter)
                if hist_y + dy <= traj_use_diameter:
                    while current_y <= hist_y + dy:
                        # 向上移动delta y
                        positions.append([current_x, current_y])
                        # velocities.append([0.0, v_abs])  # 在y方向上移动，vx=0，vy=v_abs
                        head_direction.append([np.cos(current_HD), np.sin(current_HD)])
                        angular_velocities.append(v_HD)
                        current_y += v_abs * dt
                        current_HD += v_HD * dt
                    else:
                        current_y = hist_y + dy
                else:
                    break
            # print("finish heng", current_x, current_y)
            while current_x >= 0:
                # print("begin shu", current_y, v_abs* dt)
                while current_y >= v_abs * dt:
                    positions.append([current_x, current_y])
                    # velocities.append([0.0, -v_abs])
                    head_direction.append([np.cos(current_HD), np.sin(current_HD)])
                    angular_velocities.append(v_HD)
                    current_y -= v_abs * dt
                    current_HD += v_HD * dt
                hist_x = current_x
                # print(current_x, dy)
                if hist_x - dy >= 0:
                    # print("hist", hist_x-dy)
                    while current_x >= hist_x - dy:
                        # 向上移动delta y
                        positions.append([current_x, current_y])
                        # velocities.append([-v_abs, 0.0])  # 在y方向上移动，vx=0，vy=v_abs
                        head_direction.append([np.cos(current_HD), np.sin(current_HD)])
                        angular_velocities.append(v_HD)
                        current_x -= v_abs * dt
                        current_HD += v_HD * dt
                    else:
                        current_x = hist_x - dy
                else:
                    break

                while current_y <= traj_use_diameter - v_abs * dt:
                    positions.append([current_x, current_y])
                    # velocities.append([0.0, v_abs])  # 在x方向上移动，vx=-v_abs，vy=0
                    head_direction.append([np.cos(current_HD), np.sin(current_HD)])
                    angular_velocities.append(v_HD)
                    current_y += v_abs * dt
                    current_HD += v_HD * dt
                hist_x = current_x
                if hist_x - dy >= 0:
                    while current_x >= hist_x - dy:
                        # 向上移动delta y
                        positions.append([current_x, current_y])
                        # velocities.append([-v_abs, 0.0])  # 在y方向上移动，vx=0，vy=v_abs
                        head_direction.append([np.cos(current_HD), np.sin(current_HD)])
                        angular_velocities.append(v_HD)
                        current_x -= v_abs * dt
                        current_HD += v_HD * dt
                    else:
                        current_x = hist_x - dy
                else:
                    break

        # 将位置和速度矩阵转换为NumPy数组
        positions_matrix = np.array(positions)
        head_direction_matrix = np.array(head_direction)
        angular_velocities = np.array(angular_velocities)
        total_time = len(positions) * dt
        velocities_matrix = np.gradient(positions_matrix, axis=0)

        return positions_matrix, velocities_matrix, head_direction_matrix, angular_velocities, total_time

    def trajectory_test(self, T=6400, v_max=0.01, scale=1.0, v_HD=0, dt=1):
        # 设置时间参数 t 的范围和步长
        t = np.arange(0, T, dt)
        # 使用调整后的振幅和频率定义 x(t) 和 y(t)
        x = 0.4 * np.sin(v_max / 10 * t) + 0.4
        y = 0.4 * np.cos(v_max * 3 / 10 * t) + 0.4

        # 将位置和速度矩阵转换为NumPy数组
        positions_matrix = np.column_stack((x, y))
        # 重新计算速度向量 v = (vx, vy)
        vx = np.gradient(x, t)
        vy = np.gradient(y, t)
        velocities_matrix = np.column_stack((vx, vy))
        # if self.norm:
        #     positions_matrix = (positions_matrix-self.min_point) / (self.max_point-self.min_point)
        # velocities_matrix = np.diff(positions_matrix, axis=0)

        total_time = int(len(x) * dt)
        print("check vel shape", len(x), velocities_matrix.shape, positions_matrix.shape, T)

        speed = np.sqrt(velocities_matrix[:, 0] ** 2 + velocities_matrix[:, 1] ** 2)
        head_direction = velocities_matrix / (speed[:, np.newaxis])  # normalizing velocity vector
        HD_truth = np.zeros(total_time)
        for i in range(total_time):
            HD_truth[i] = angle_between_vectors(velocities_matrix[i], np.array([1,0]))
        angular_velocity = np.diff(HD_truth)
        angular_velocity = np.append(angular_velocity, 0)
        return positions_matrix, velocities_matrix, head_direction, angular_velocity, int(total_time)


class ConfigParam:
    def __init__(self, env: Env):
        self.env = env
        self.OVC = {  # 描述OVC的关键参数
            "N_dis": 10,  # 每个landamrk有多少个描述距离的细胞，将N_dis平埔到整个环境中
            "N_theta": 10,  # 每个landmark有多少个描述角度的细胞，N_theta平铺360度
            "gauss_width": 0.07,   # 每单位宽度的env需要对应OVC的响应域
        }
        self.num_mec = 10
        self.num_mec_module = 7
        self.mec_max_ratio = 9
        self.num_hpc = 8000
        self.num_HD = 128
        self.dt = 1.  # 在轨迹采样函数没改变时，dt只能是1，否则速度，loc关系将会不对应

