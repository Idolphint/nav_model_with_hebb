import datetime
import os
import time
import numpy as np
import torch
from couple_hd_hpc_grid import Coupled_Net_HHG
from grid_model import GridModel
from hpc_model import HPCModel
from hd_obv_system import HdOVCModel
from model_config import Env, ConfigParam
from draw_util import draw_place_field_distribution, draw_population_activity
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(Coupled_Model: Coupled_Net_HHG,
          lapnum,
          config,
          train=1,
          get_loc=0., get_HD=1, get_view=1):
    v_abs = 0.01
    v_HD = np.pi / 20 - 0.001 if train==1 else 0.06
    dy = 0.02
    t0 = time.time()
    loc, velocity, head_direction, angular_velocities, total_time = config.env.trajectory_train(v_abs=v_abs, v_HD=v_HD,
                                                                                                dy=dy, lapnum=lapnum,
                                                                                                dt=config.dt)
    loc = torch.Tensor(loc).to(device)
    velocity = torch.Tensor(velocity).to(device)
    head_direction = torch.Tensor(head_direction).to(device)
    angular_velocities = torch.Tensor(angular_velocities).to(device)
    print("ready data using", time.time()-t0)
    t0 = time.time()
    # init net
    Coupled_Model.sen2hpc_stre = 0.
    Coupled_Model.sen2HD_stre = 0.  # No sensory input
    Coupled_Model.hpc2mec_stre = 0.
    init_shared_t = torch.arange(1001).to(device)*config.dt
    for i in tqdm(range(300)):
        Coupled_Model.update(velocity=torch.zeros(2).to(device), loc=loc[0], HD_gt=torch.Tensor([1,0]).to(device),
                             angular_velocity=0,
                             shared_t=init_shared_t[i:i+2],
                             get_view=0,
                             get_loc=1,
                             get_HD=1,
                             train=0)

    # TODO 由于没办法通过shared_t将多个model同时计算微分方程，所以这里可能计算比较慢，必须一步一步算
    total_shared_t = torch.arange(loc.shape[0]+1).to(device)*config.dt
    Coupled_Model.hpc2mec_stre = 1.
    Coupled_Model.sen2hpc_stre = 1.
    for i in tqdm(range(loc.shape[0])):
        Coupled_Model.update(velocity=velocity[i], loc=loc[i], HD_gt=head_direction[i],
                             angular_velocity=angular_velocities[i],
                             shared_t=total_shared_t[i:i+2],
                             get_view=get_view,
                             get_loc=get_loc,
                             get_HD=get_HD,
                             train=train)
        # print("HD", Coupled_Model.HD_net.center, (i*angular_velocities[0].item())% (np.pi * 2),
        #       "hpc", torch.argmax(Coupled_Model.HPC_model.r))
              # "Grid", Coupled_Model.MEC_model_list[0].center, loc[i])


def test(Coupled_Model: Coupled_Net_HHG,
        config,
        get_loc=0., get_HD=1, get_view=1, draw_step=1):
    # 用于保存place_information的应当traj密集一些，画图用的traj可以宽松一些
    v_abs = 0.02
    v_HD = 0.1
    dy = 0.02
    HD_init = 0.
    if draw_step==1:
        loc, velocity, head_direction, angular_velocities, total_time = config.env.trajectory_train(v_abs=v_abs, v_HD=v_HD,
                                                                                                    dy=dy, lapnum=1, dt=config.dt)
    elif draw_step == 2:
        loc, velocity, head_direction, angular_velocities, total_time = config.env.trajectory_test(T=900, dt=config.dt)
    loc = torch.Tensor(loc).to(device)
    velocity = torch.Tensor(velocity).to(device)
    head_direction = torch.Tensor(head_direction).to(device)
    angular_velocities = torch.Tensor(angular_velocities).to(device)

    HD_vec_init = torch.Tensor([np.cos(HD_init), np.sin(HD_init)]).to(device)
    Coupled_Model.reset_state(loc[0], HD_init)
    Coupled_Model.mec2hpc_stre = 0.  # 初始化不做路径积分！
    Coupled_Model.sen2hpc_stre = 10.
    Coupled_Model.hpc2mec_stre = 10.

    for i in tqdm(range(400)):
        Coupled_Model.update(velocity=torch.zeros(2).to(device), loc=loc[0], HD_gt=HD_vec_init,
                             angular_velocity=0,
                             shared_t=torch.Tensor([i * config.dt, (i + 1) * config.dt]).to(device),
                             get_view=0,
                             get_loc=get_loc,
                             get_HD=get_HD,
                             train=0)

    Coupled_Model.mec2hpc_stre = 1.
    Coupled_Model.sen2hpc_stre = 1.
    Coupled_Model.hpc2mec_stre = 1.

    hpc_data_u = []
    hpc_data_r = []
    couple_I_mec = []
    couple_I_sen = []

    for i in tqdm(range(loc.shape[0])):
        Coupled_Model.update(velocity=velocity[i], loc=loc[i], HD_gt=head_direction[i],
                             angular_velocity=angular_velocities[i],
                             shared_t=torch.Tensor([i * config.dt, (i + 1) * config.dt]).to(device),
                             get_view=get_view,
                             get_loc=get_loc,
                             get_HD=get_HD,
                             train=0)
        # 下面酌情考虑保存并查看
        u_HPC = Coupled_Model.HPC_model.u
        r_HPC = Coupled_Model.HPC_model.r
        I_mec = Coupled_Model.I_mec  # mec 输入到hpc(place cell)的信号
        I_sen = Coupled_Model.I_sen  # ovc 输入到place cell的信号
        u_mec = Coupled_Model.u_mec_module
        input_hpc_module = Coupled_Model.input_hpc_module

        hpc_data_u.append(u_HPC.detach())
        hpc_data_r.append(r_HPC.detach())
        couple_I_mec.append(I_mec.detach())
        couple_I_sen.append(I_sen.detach())

    hpc_data_u = torch.stack(hpc_data_u, dim=0)
    hpc_data_r = torch.stack(hpc_data_r, dim=0)
    couple_I_sen = torch.stack(couple_I_sen, dim=0)
    couple_I_mec = torch.stack(couple_I_mec, dim=0)
    # print(hpc_data_u.shape, hpc_data_u[:10, 0])
    title = f"lap{train_lap}_" + datetime.datetime.now().strftime("%Y-%m-%d")
    torch.save({'u_HPC':hpc_data_u, 'r_HPC': hpc_data_r, 'I_sen':couple_I_sen, 'I_mec':couple_I_mec, "loc":loc},
               ckpt_path+"test_data_"+title+'_'+str(draw_step)+".pt")
    if draw_step == 1:
        max_r, place_indice, place_coordinates = draw_place_field_distribution(
            hpc_data_u, hpc_data_r, loc, title=title)
    elif draw_step == 2:
        draw_population_activity(hpc_data_r, couple_I_mec, couple_I_sen, loc,
                                 place_info_path="./visual_res/place_information_2024-08-30.npy", title=title)


if __name__ == '__main__':
    env = Env()
    config = ConfigParam(env)

    ckpt_path = "./ckpt/"
    os.makedirs(ckpt_path, exist_ok=True)
    pre_train_lap = 2
    train_lap = 3
    part_view = 1
    HPC_model = HPCModel(config=config)
    MEC_model_list = []
    ratio = np.linspace(1.2, config.mec_max_ratio, config.num_mec_module)  # TODO 确认是否0，1长度的环境这个周期可以？
    angle = np.linspace(0, np.pi / 3, config.num_mec_module)
    for i in range(config.num_mec_module):
        MEC_model_list.append(GridModel(ratio=ratio[i], angle=angle[i], config=config))
    HD_net = HdOVCModel(config=config)
    Coupled_Model = Coupled_Net_HHG(HPC_model=HPC_model, MEC_model_list=MEC_model_list, HD_net=HD_net,
                                    config=config).to(device)
    print("###finish init model")
    if True:
        train(Coupled_Model, lapnum=pre_train_lap, config=config, get_HD=1, get_view=part_view, get_loc=1, train=1)
        train(Coupled_Model, lapnum=train_lap, config=config, get_HD=0, get_loc=0, get_view=part_view, train=2)
        view = "" if part_view else "fullV_"
        torch.save({"state_dict": Coupled_Model.state_dict()}, ckpt_path+f"lap{train_lap}_"+ view + datetime.datetime.now().strftime("%Y-%m-%d") + ".pth")
    else:
        para = torch.load("./ckpt/lap2_2024-08-30.pth")
        print(para.keys(), para["state_dict"]['HD_net.conn_bearing'].shape)
        Coupled_Model.load_state_dict(para["state_dict"])
    print("### begin test trained model")
    test(Coupled_Model, config=config, get_loc=0, get_view=part_view, get_HD=0, draw_step=2)
