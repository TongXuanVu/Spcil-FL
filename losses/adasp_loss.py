import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ["AdaSPLoss"]

class AdaSPLoss(object):
    """
    自适应稀疏配对（AdaSP）损失
    """

    def __init__(self, N_id, temp=0.04, device="cuda:0", loss_type='adasp'):
        self.temp = temp
        self.loss_type = loss_type
        self._device = device
        self.N_id = N_id

    def __call__(self, feats, targets):
        # Lam sach input: neu feats co nan/inf (exploding) -> thay bang 0 truoc khi chuan hoa.
        # Chi tac dong len gia tri khong huu han; gia tri hop le giu nguyen.
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        # 归一化输入特征
        feats_n = nn.functional.normalize(feats, dim=1)
        scale = 1. / self.temp

        # 计算相似性矩阵
        sim_qq = torch.matmul(feats_n, feats_n.T)
        sf_sim_qq = sim_qq * scale

        # Y: (N, N_id) - maps each instance to its class dynamically
        Y = F.one_hot(targets, num_classes=self.N_id).float().to(self._device)
        
        # pos_mask: (N, N) - 1 if same class, 0 otherwise
        pos_mask = torch.matmul(Y, Y.T)
        
        # mask_H: -1 for positive pairs, 1 for negative pairs
        mask_H = 1.0 - 2.0 * pos_mask

        # ID_sim_HH calculation
        # Clamp so mu truoc exp de chan tran so (inf). Voi feature da chuan hoa,
        # gia tri that nam trong [-scale, scale] nen clamp o 30 khong doi ket qua binh thuong.
        exp_sim_H = torch.exp(torch.clamp(sf_sim_qq * mask_H, max=30.0))
        ID_sim_HH = torch.matmul(Y.T, torch.matmul(exp_sim_H, Y)) # (N_id, N_id)
        
        pos_mask_id = torch.eye(self.N_id).to(self._device)
        # Invert diagonal elements safely
        diag_HH = torch.diag(ID_sim_HH).clone()
        diag_HH[diag_HH == 0] = 1.0
        diag_HH = 1.0 / diag_HH
        ID_sim_HH = ID_sim_HH * (1 - pos_mask_id) + torch.diag(diag_HH)

        # Normalize
        ID_sim_HH_L1 = nn.functional.normalize(ID_sim_HH, p=1, dim=1)

        # ID_sim_HE calculation
        ID_sim_HE = torch.matmul(exp_sim_H, Y) # (N, N_id)
        
        # Invert elements where Y == 1
        pos_sim_HE = ID_sim_HE * Y
        pos_sim_HE[pos_sim_HE == 0] = 1.0
        pos_sim_HE = 1.0 / pos_sim_HE
        ID_sim_HE = ID_sim_HE * (1 - Y) + pos_sim_HE
        
        ID_sim_HE = torch.matmul(Y.T, ID_sim_HE) # (N_id, N_id)
        ID_sim_HE_L1 = nn.functional.normalize(ID_sim_HE, p=1, dim=1)

        # Both sim and Adaptive
        l_sim = torch.log(torch.diag(ID_sim_HH) + 1e-12)
        s_sim = torch.log(torch.diag(ID_sim_HE) + 1e-12)

        weight_sim_HH = l_sim.detach() / scale
        weight_sim_HE = s_sim.detach() / scale
        
        wt_l = 2 * weight_sim_HE * weight_sim_HH / (weight_sim_HH + weight_sim_HE + 1e-12)
        wt_l[weight_sim_HH < 0] = 0
        
        both_sim = l_sim * wt_l + s_sim * (1 - wt_l)
        
        # Clamp truoc exp de chan inf lam hong adaptive_sim_mat -> nan.
        adaptive_pos = torch.diag(torch.exp(torch.clamp(both_sim, max=30.0)))
        adaptive_sim_mat = adaptive_pos * pos_mask_id + ID_sim_HE * (1 - pos_mask_id)
        adaptive_sim_mat_L1 = nn.functional.normalize(adaptive_sim_mat, p=1, dim=1)

        # Determine valid classes in this batch to avoid NaN loss
        valid_classes = (torch.sum(Y, dim=0) > 0).float()
        num_valid = torch.sum(valid_classes)
        
        # Mask out invalid classes on diagonal
        diag_HH_L1 = torch.diag(ID_sim_HH_L1)
        diag_HE_L1 = torch.diag(ID_sim_HE_L1)
        diag_ada_L1 = torch.diag(adaptive_sim_mat_L1)
        
        loss_sph = -1 * torch.sum(torch.log(diag_HH_L1 + 1e-12) * valid_classes) / (num_valid + 1e-12)
        loss_splh = -1 * torch.sum(torch.log(diag_HE_L1 + 1e-12) * valid_classes) / (num_valid + 1e-12)
        loss_adasp = -1 * torch.sum(torch.log(diag_ada_L1 + 1e-12) * valid_classes) / (num_valid + 1e-12)

        if self.loss_type == 'sp-h':
            loss = loss_sph
        elif self.loss_type == 'sp-lh':
            loss = loss_splh
        elif self.loss_type == 'adasp':
            loss = loss_adasp
            
        return loss
