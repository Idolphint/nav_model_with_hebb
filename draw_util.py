import os
import datetime

import numpy as np

from util import *
import matplotlib.pyplot as plt
from model_config import Env, ConfigParam
from matplotlib.animation import FuncAnimation


# base env config
env = Env()
config = ConfigParam(env)
save_dir = "./visual_res/"
os.makedirs(save_dir, exist_ok=True)

def draw_place_field_distribution(hpc_u, hpc_fr, loc, thres=3.5, title=""):
    # 输入是一段时间内全部细胞的活动 shape= num_cell, timestep
    hpc_fr = hpc_fr.detach().cpu().numpy()
    hpc_u = hpc_u.detach().cpu().numpy()
    loc = loc.detach().cpu().numpy()
    max_r = np.max(hpc_fr, axis=0)
    thres = np.mean(hpc_u)
    print("check max r", max_r, thres, max_r.shape)
    max_u, place_index, place_num = place_cell_select_fr(hpc_u, thres=thres)
    place_score = max_r[place_index.reshape(-1, )]
    Center = place_center(hpc_fr, place_index, loc)
    print("check center", Center, place_score)
    plt.scatter(Center[:, 0], Center[:, 1], c=place_score)

    # plt.plot(env.x_bound, env.y_bound)
    plt.plot(env.loc_land[:, 0], env.loc_land[:, 1], 'r.', markersize=10, label='Landmarks')
    # plt.xlim([env.x1, env.x2])
    # plt.ylim([env.y1, env.y2])
    plt.axis('equal')
    plt.title("place field distribution")
    plt.legend(loc='lower right')
    plt.savefig(save_dir+"place_field_distribution_"+title+".png")
    # plt.show()
    print("fig drawn!")
    mat_dict = {'Max fr': max_r, 'place_index': place_index.reshape(-1), 'Center': Center}
    np.save(save_dir+"place_information_"+title, mat_dict)  # ?
    print("data saved!")
    return max_r, place_index, Center

def draw_population_activity(hpc_fr, I_mec, I_sen, loc, place_info_path, title=""):
    hpc_fr = hpc_fr.detach().cpu().numpy()
    I_mec = I_mec.detach().cpu().numpy()
    I_sen = I_sen.detach().cpu().numpy()
    loc = loc.detach().cpu().numpy()
    data = np.load(place_info_path, allow_pickle=True).item()
    place_index = data['place_index']  # 可能是place index顺序有问题？
    place_cell_coordinates = data['Center']

    place_fr = hpc_fr[:, place_index.reshape(-1, )]
    place_mec = I_mec[:, place_index.reshape(-1, )]
    place_sen = I_sen[:, place_index.reshape(-1, )]

    # Decode population activities of place cells
    # 各个模块建议的hpc的中心，也就是从mec猜测的hpc该发放哪个，从sen猜测的hpc该发放哪个cell
    decoded_x, decoded_y = get_center_hpc(place_fr, place_cell_coordinates[:, 0], place_cell_coordinates[:, 1])
    decoded_x_mec, decoded_y_mec = get_center_hpc(place_mec, place_cell_coordinates[:, 0], place_cell_coordinates[:, 1],
                                                  thres=0.4)  # Grid cell input
    decoded_x_sen, decoded_y_sen = get_center_hpc(place_sen, place_cell_coordinates[:, 0],
                                                  place_cell_coordinates[:, 1])  # LV cell input

    x = loc[:, 0]
    y = loc[:, 1]

    fig = plt.figure()
    plt.plot(x, y, label="Groudtruth Trajectory")
    plt.plot(decoded_x[10:-10], decoded_y[10:-10], label="Decoded Trajectory HPC")
    plt.plot(decoded_x_mec[10:-10], decoded_y_mec[10:-10], label="Decoded Trajectory MEC")
    plt.plot(decoded_x_sen[10:-10], decoded_y_sen[10:-10], label="Decoded Trajectory SEN")

    plt.plot(env.loc_land[:, 0], env.loc_land[:, 1], 'r.', markersize=10)
    plt.legend()
    plt.savefig(save_dir+'Place_decoding'+title+'.png')

    n_step = 2
    gif_plaec_data = place_fr[::n_step, :]
    T = gif_plaec_data.shape[0]
    scatter = plt.scatter([], [], c=[], cmap='viridis', s=25)
    def update(frame):
        if frame % 20 == 0:
            print(frame)
        data = gif_plaec_data[frame].flatten()
        scatter.set_offsets(place_cell_coordinates)
        scatter.set_array(data)
        return scatter

    ani = FuncAnimation(fig, update, frames=T, interval=20, blit=False)
    ani_filename = save_dir + 'test_Population_activities'+title+'.gif'
    ani.save(ani_filename, writer='Pillow', fps=30)


def visual_traj():
    env = Env()
    traj, _, _, _, T = env.trajectory_train(1, v_HD=0.1, v_abs=0.01, dy=0.01)
    fig = plt.figure()
    plt.plot(env.loc_land[:, 0], env.loc_land[:, 1], 'r.', markersize=10)
    plt.plot(traj[:,0], traj[:,1])
    plt.savefig(save_dir + 'traj_train.jpg')
    # scatter = plt.scatter([], [], c=[], cmap='viridis', s=25)
    # n_step = 2
    # traj = traj[::n_step, :]
    # # print(traj)
    # T = traj.shape[0]-5
    # def update(frame):
    #     data = traj[frame:frame+5]
    #     scatter.set_offsets(data)
    #     data_c = np.ones((data.shape[0]))
    #     scatter.set_array(data_c)
    #     return scatter
    #
    # ani = FuncAnimation(fig, update, frames=T, interval=20, blit=False)
    # ani_filename = save_dir + 'traj_train.gif'
    # ani.save(ani_filename, writer='Pillow', fps=30)


if __name__ == '__main__':
    data = torch.load("./ckpt/test_data_lap2_2024-08-30_2.pt")
    hpc_data_u = data["u_HPC"]  # .detach().cpu().numpy()
    hpc_data_r = data["r_HPC"]  # .detach().cpu().numpy()
    loc = data["loc"]  # .detach().cpu().numpy()
    couple_I_mec = data["I_sen"]  # .detach().cpu().numpy()
    couple_I_sen = data["I_mec"]  # .detach().cpu().numpy()
    title = datetime.datetime.now().strftime("%Y-%m-%d")

    # max_r, place_indice, place_coordinates = draw_place_field_distribution(
    #     hpc_data_u, hpc_data_r, loc, title=title)

    draw_population_activity(hpc_data_r, couple_I_mec, couple_I_sen, loc,
                             place_info_path="./visual_res/place_information_2024-08-30.npy", title=title)
