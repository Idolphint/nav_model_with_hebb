import numpy as np
import torch


def angle_between_vectors(v1, v2):
    if isinstance(v1, torch.Tensor):
        v1_norm = torch.nn.functional.normalize(v1, dim=-1, p=2)
        v2_norm = torch.nn.functional.normalize(v2, dim=-1, p=2)

        dot_product = torch.clamp(torch.sum(v1_norm*v2_norm, dim=-1), -1.0, 1.0)
        angle = torch.arccos(dot_product)
        return angle
    else:
        # 计算向量的模
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        # 计算向量的点积
        dot_product = np.dot(v1, v2)
        norm_mul = v1_norm*v2_norm
        # 计算夹角的弧度值
        cos_theta = np.where(norm_mul == 0., 0, dot_product / norm_mul)
        # 通过反余弦函数计算夹角（弧度）
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        angle_cross = np.cross(v1, v2)
        angle = np.where(angle_cross>0, -angle, angle)
        return angle


def period_bound(A):
    if isinstance(A, torch.Tensor):
        B = torch.where(A > np.pi, A - 2 * np.pi, A)
        C = torch.where(B < -np.pi, B + 2 * np.pi, B)
    else:
        B = np.where(A > np.pi, A - 2 * np.pi, A)
        C = np.where(B < -np.pi, B + 2 * np.pi, B)
    return C


def keep_top_n(mat, n):
    n = int(n)
    mat = mat.flatten()
    topk_values, topk_indices = torch.topk(mat, n)
    mask = torch.zeros_like(mat)  # 创建一个与展平后数组大小相同的全零数组
    mask[topk_indices] = topk_values
    return mask


def normalize_rows(matrix, dim=1):
    ori_shape = [1]*len(matrix.shape)
    ori_shape[dim] = matrix.shape[dim]
    column_sums = torch.sum(matrix, dim=dim, keepdim=True).repeat(ori_shape)
    nonzero_mask = (column_sums != 0)
    normalized_matrix = matrix.clone()
    normalized_matrix[nonzero_mask] = matrix[nonzero_mask] / column_sums[nonzero_mask]
    return normalized_matrix


def place_cell_select_fr(HPC_fr, thres=0.002):
    # HPC_fr = HPC_fr
    max_fr = np.max(HPC_fr, axis=0)
    place_index = np.argwhere(max_fr > thres)  # 只要发放最大的时候超过了阈值，就被选作place cell
    place_num = place_index.shape[0]
    return max_fr, place_index, place_num


def place_center(HPC_fr, place_index, loc):
    # 根据最大发放计算place center, 一个cell的center由它发放最大时刻的traj真实loc决定
    place_num = int(place_index.shape[0])
    Center = np.zeros([place_num, 2])
    fr_probe = HPC_fr[:,place_index.reshape(-1,)]
    for i in range(place_num):
        max_time = np.argmax(fr_probe[:,i],axis=0)
        Center[i,:] = loc[max_time,:]
    return Center


def get_center_hpc(fr, center_x, center_y, thres=0.2):
    # max_values = fr.max(axis=1).reshape(-1, 1)
    max_index = np.argmax(fr, axis=1)
    T = fr.shape[0]
    Cx = np.zeros(T,)
    Cy = np.zeros(T,)
    for i in range(T):
        Cx[i] = center_x[max_index[i]]
        Cy[i] = center_y[max_index[i]]
    return Cx, Cy


class Lambda(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_total = torch.Tensor([0.5, 0.5])

    # def forwardo(self, t, y):
    #     return torch.mm(y ** 3, true_A)
    def forward(self, t, state):
        u = state[0]
        v = state[1]
        du = (-u + self.input_total - v) / 1.
        dv = (-v + 0.085 * u) / 10.
        return torch.stack([du, dv])

# def Lambda(t, y):
#     return torch.mm(y ** 3, true_A)
# def derivate()


if __name__ == '__main__':
    a = torch.randn((2,2,5))
    norm_a = normalize_rows(a, dim=2)
    print(a, norm_a)