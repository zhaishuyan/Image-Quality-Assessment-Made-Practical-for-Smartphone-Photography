import itertools as it
import numpy as np
import torch
import torch.nn as nn
import itertools

class pLoss(nn.Module):
    def __init__(self, hexG):
        super(pLoss, self).__init__()
        self.legal_state = hexG

    def forward(self, f, y, mask, auto_mode=False):
        pMargin = torch.zeros((f.shape[0], f.shape[1])).to(f.device)

        S = self.legal_state.to(f.device)
        # f: bzxn S: (n+1)xn
        potential = torch.mm(S.double(), f.T.double())
        max_sf, _ = torch.max(potential, dim=0)  # to solve the overflow or underflow
        J = torch.exp(potential - max_sf)  # (n+1)xbz
        z_ = torch.sum(J, dim=0)

        # id_num = _check_abnornal(z_)

        for i in range(f.shape[1]):
            pMargin[:, i] = torch.sum(J[S[:, i] > 0, :], dim=0) / z_

        # pMargin: [B, n], y: [B,n]
        loss = []
        for j in range(y.size(1)):
            pred_i = pMargin[:, j]
            target_i = y[:, j]
            mask_i = mask[:, j]
            loss_i = nn.BCELoss(reduction='none')(pred_i, target_i) * mask_i # instantiation via cross-entropy loss function
            loss.append(loss_i.mean())
        if auto_mode:
            return loss, pMargin
        else:
            return sum(loss), pMargin

    def infer(self, f):
        pMargin = torch.zeros((f.shape[0], f.shape[1])).to(f.device)
        S = self.legal_state.to(f.device)
        potential = torch.mm(S.double(), f.T.double())

        max_sf, _ = torch.max(potential, dim=0)  # to solve the overflow or underflow
        J = torch.exp(potential - max_sf)  # (n+1)xbz
        z_ = torch.sum(J, dim=0)

        # for Z_ is inf, ignore its pMargin loss
        # id_num = _check_abnornal(z_)

        for i in range(f.shape[1]):
            pMargin[:, i] = torch.sum(J[S[:, i] > 0, :], dim=0) / z_

        return pMargin




def bin_list(nsize: int) -> torch.Tensor:
    """生成所有 0/1 组合，共 2^n 行。"""
    return torch.tensor(list(it.product(range(2), repeat=nsize)), dtype=torch.int64)


def list_legal_states(Ee, Eh, n_nodes: int = None) -> torch.Tensor:
    """
    仅基于 Ee(互斥) 与 Eh(依赖) 过滤合法状态。
    Ee: list[(a,b)]  —  互斥边：不允许 a=1 且 b=1
    Eh: list[(u,v)]  —  依赖边：v 依赖 u，不允许 v=1 且 u=0
    n_nodes: 若为 None，则用边集中最大下标+1 自动推断
    """
    # 自动推断节点数
    if n_nodes is None:
        max_idx = -1
        if Ee:
            max_idx = max(max_idx, max(max(e) for e in Ee))
        if Eh:
            max_idx = max(max_idx, max(max(e) for e in Eh))
        n_nodes = max_idx + 1

    # 穷举所有状态
    S = bin_list(n_nodes).numpy()         # (2^n, n)
    keep = np.ones(len(S), dtype=bool)

    # 互斥过滤：a=1 & b=1 违规
    for a, b in Ee:
        keep &= ~((S[:, a] == 1) & (S[:, b] == 1))

    # 依赖过滤：u=1 & v=0 违规
    for u, v in Eh:
        keep &= ~((S[:, u] == 1) & (S[:, v] == 0))

    S_legal = S[keep]
    return torch.tensor(S_legal, dtype=torch.int64)

# def graph_SO_FFSC():
#     Eh_edge = [
#         (2, 0), (2, 1), (1, 0),
#         (5, 3), (5, 4), (4, 3),
#         (8, 6), (8, 7), (7, 6),
#         (11, 9), (11, 10), (10, 9),
#         (14, 12), (14, 13), (13, 12),
#         (17, 15), (17, 16), (16, 15),
#         (20, 18), (20, 19), (19, 18),
#         (23, 21), (23, 22), (22, 21),
#         (26, 24), (26, 25), (25, 24),
#         (29, 27), (29, 28), (28, 27),
#         (32, 30), (32, 31), (31, 30),
#         (35, 33), (35, 34), (34, 33)
#     ]
#     Ee_edge = [(0, 3), (6, 9), (12, 15), (18, 21), (24, 27), (18, 24), (18, 27), (21, 24), (21, 27), (30, 33)]
#     state = list_legal_states(Ee_edge, Eh_edge, n_nodes=36)
#     # print(state.shape)
#     # print(state)
#     return state

def graph_SO_FFSC():
    # 定义两个互斥attribute的所有状态
    states2 = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0]
    ]
    # 3个互斥attributes的所有合法状态：10种
    states3 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0]
    ])
    legal_all = np.array([
        np.concatenate(combination)
        for combination in itertools.product(states2, states2, states3)
    ])
    return torch.tensor(legal_all, dtype=torch.int64)