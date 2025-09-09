import torch, pdb, copy
import torch.nn as nn
import numpy as np
import itertools as it
import torch.nn.functional as F


class pLoss_all_fidelity(nn.Module):
    def __init__(self, hexG):
        super(pLoss_all_fidelity, self).__init__()
        self.legal_state = hexG
        # self.my_loss = Multi_Fidelity_Loss(False)
        self.my_loss = FocalLoss(gamma=0.5, alpha=10, reduction='mean')


    def forward(self, f, y):
        y = y.float()
        pMargin = torch.zeros((f.shape[0], f.shape[1])).to(f.device)

        S = self.legal_state.to(f.device)
        # f: bzxn S: (n+1)xn
        potential = torch.mm(S.double(), f.T.double())
        max_sf, _ = torch.max(potential, dim=0)  # to solve the overflow or underflow
        J = torch.exp(potential - max_sf)  # (n+1)xbz
        # J = potential
        z_ = torch.sum(J, dim=0)

        id_num = _check_abnornal(z_)

        for i in range(f.shape[1]):
            pMargin[:, i] = torch.sum(J[S[:, i] > 0, :], dim=0) / z_

        all_loss = self.my_loss(pMargin, y)
        return all_loss, pMargin


    def infer(self, f):
        pMargin = torch.zeros((f.shape[0], f.shape[1])).to(f.device)
        S = self.legal_state.to(f.device)
        potential = torch.mm(S.double(), f.T.double())

        max_sf, _ = torch.max(potential, dim=0)  # to solve the overflow or underflow
        J = torch.exp(potential - max_sf)  # (n+1)xbz
        z_ = torch.sum(J, dim=0)

        # for Z_ is inf, ignore its pMargin loss
        id_num = _check_abnornal(z_)

        for i in range(f.shape[1]):
            pMargin[:, i] = torch.sum(J[S[:, i] > 0, :], dim=0) / z_

        return pMargin



class Fidelity_Loss_binary(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.esp = 1e-8

    def forward(self, p, g):
        loss = 1 - (torch.sqrt(p * g + self.esp) + torch.sqrt((1 - p) * (1 - g) + self.esp))

        return torch.mean(loss)


class Multi_Fidelity_Loss(torch.nn.Module):
    def __init__(self, loss_mask=False):
        super(Multi_Fidelity_Loss, self).__init__()
        self.esp = 1e-8
        self.loss_mask = loss_mask

    def forward(self, p, g):

        if not self.loss_mask:
            return self.forward_wo_mask_matrix(p, g)
        else:
            return self.forward_w_mask(p, g)

    def forward_wo_mask_matrix(self, p, g):
        p_i = p.view(-1, p.size(1), 1)
        g_i = g.view(-1, g.size(1), 1)
        loss_i = 1 - (torch.sqrt(p_i * g_i + self.esp) + torch.sqrt((1 - p_i) * (1 - g_i) + self.esp))

        loss = torch.mean(loss_i, dim=1)
        mean_loss = torch.mean(loss)
        return mean_loss


    def forward_w_mask(self, p, g):
        indx = torch.where(g.sum(dim=1) == 0)[0]
        mask = torch.ones((g.size(0),))
        mask[indx]=0
        mask = mask.to(p.device)

        loss = 0
        for i in range(p.size(1)):
            p_i = p[:, i] * mask
            g_i = g[:, i] * mask
            g_i = g_i.view(-1, 1)
            p_i = p_i.view(-1, 1)
            loss_i = 1 - (torch.sqrt(p_i * g_i + self.esp) + torch.sqrt((1 - p_i) * (1 - g_i) + self.esp))
            loss = loss + loss_i
        loss = loss / p.size(1)

        return torch.mean(loss)





class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        """
        Unified Focal Loss class for multi-label classification task.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction


    def forward(self, inputs, targets):
        """
        Focal loss for multi-label classification.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - multi-label: (batch_size, num_classes)
        """

        # probs = torch.sigmoid(inputs)
        probs = inputs

        # Compute binary cross entropy
        # bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        bce_loss = F.binary_cross_entropy(probs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss





# utils for pLoss
def _check_abnornal(z_):
    if np.inf in z_:
        pdb.set_trace()
        idx = z_ == np.inf
        id_num = [i for i, v in enumerate(list(idx)) if v == True]
    else:
        id_num = [-1]
    return id_num

