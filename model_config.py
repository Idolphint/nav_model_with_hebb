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

    def trajectory_train(self, lapnum=1, v_HD=np.pi/20, v_abs=0.01, dy=0.01, HD_init=0, dt=1):
        current_x, current_y = 0, 0  # agent的初始位置
        current_HD = HD_init
        positions = []
        head_direction = []
        angular_velocities = []
        # 生成轨迹
        for i in range(lapnum):
            while current_y <= self.diameter:
                while current_x <= self.diameter - v_abs * dt:
                    positions.append([current_x, current_y])
                    # velocities.append([v_abs, 0.0])  # 在x方向上移动，vx=v_abs，vy=0
                    head_direction.append([np.cos(current_HD), np.sin(current_HD)])
                    angular_velocities.append(v_HD)
                    current_x += v_abs * dt
                    current_HD += v_HD * dt
                hist_y = current_y
                if hist_y + dy <= self.diameter:
                    while current_y <= hist_y + dy:
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
                if hist_y + dy <= self.diameter:
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

            while current_x >= 0:
                while current_y >= v_abs * dt:
                    positions.append([current_x, current_y])
                    # velocities.append([0.0, -v_abs])
                    head_direction.append([np.cos(current_HD), np.sin(current_HD)])
                    angular_velocities.append(v_HD)
                    current_y -= v_abs * dt
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

                while current_y <= self.diameter - v_abs * dt:
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
        self.dt = 0.1

