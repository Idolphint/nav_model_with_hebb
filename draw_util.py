import os

from util import *
import matplotlib.pyplot as plt
from model_config import Env, ConfigParam

# base env config
env = Env()
config = ConfigParam(env)
save_dir = "./visual_res/"
os.makedirs(save_dir, exist_ok=True)

def draw_place_field_distribution(hpc_u, hpc_fr, loc, thres=3.5, title=""):
    # 输入是一段时间内全部细胞的活动 shape= num_cell, timestep
    hpc_fr = hpc_fr.cpu().numpy()
    hpc_u = hpc_u.cpu().numpy()
    max_r = np.max(hpc_fr, axis=0)
    max_u, place_index, place_num = place_cell_select_fr(hpc_u, thres=thres)
    place_score = max_r[place_index.reshape(-1, )]
    Center = place_center(hpc_fr, place_index, loc)
    sc = plt.scatter(Center[:, 0], Center[:, 1], c=place_score)

    # plt.plot(env.x_bound, env.y_bound)
    plt.plot(env.loc_land[:, 0], env.loc_land[:, 1], 'r.', markersize=10, label='Landmarks')
    # plt.xlim([env.x1, env.x2])
    # plt.ylim([env.y1, env.y2])
    plt.axis('equal')
    plt.title("place field distribution")
    plt.legend(loc='lower right')
    plt.savefig(save_dir+"place_field_distribution_"+title+".png")
    plt.show()

