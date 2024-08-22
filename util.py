import numpy as np
import torch


def angle_between_vectors(v1, v2):
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

def normalize_rows(matrix):
    column_sums = torch.sum(matrix, dim=1)
    nonzero_mask = (column_sums != 0)
    normalized_matrix = matrix.clone()
    normalized_matrix[nonzero_mask] = matrix[nonzero_mask] / column_sums[nonzero_mask][:, None]
    return normalized_matrix


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
    from torchdiffeq import odeint
    u0 = torch.tensor([[0., 0.]])
    v0 = torch.tensor([[-0., 0.]])
    # true_y0 = torch.tensor([[2., 0.]])
    t = torch.linspace(0., 25., 100)
    true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])

    true_y0 = torch.stack([u0,v0])
    with torch.no_grad():
        true_y = odeint(Lambda(), true_y0, torch.Tensor([0,0.1]))
    # t = torch.linspace(0., 25., 100)
    # with torch.no_grad():
    #     true_y = odeint(Lambda, true_y0, t, method='dopri5')
    print(true_y)
    # print(true_y0 + torch.mm(true_y0 ** 3, true_A)*0.25)