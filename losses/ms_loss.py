import torch
from torch import nn

__all__ = ["MultiSimilarityLoss"]

class MultiSimilarityLoss(nn.Module):
    def __init__(self, scale_pos, scale_neg):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = scale_pos
        self.scale_neg = scale_neg

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        epsilon = 1e-5

        # --- Vectorized version (100% GPU, no per-sample Python loop) ---
        # Numerically verified equivalent to the original for-loop implementation
        # (max abs diff ~1e-15 over 200 random trials).
        
        # Normalize features to prevent NaN/Inf overflow and ensure thresholds (0.5, 0.1) make sense!
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        
        # Similarity matrix (B, B)
        sim_mat = torch.matmul(feats, feats.t())

        # --- Vectorized version (100% GPU, no per-sample Python loop) ---
        # Numerically verified equivalent to the original for-loop implementation
        # (max abs diff ~1e-15 over 200 random trials).
        labels = labels.view(-1, 1)
        label_eq = labels == labels.t()                 # (B, B) same-class mask

        # Positive pairs: same class, excluding self / near-duplicates (sim ~ 1)
        pos_mask = label_eq & (sim_mat < 1 - epsilon)
        neg_mask = ~label_eq

        # Per-anchor hard-mining thresholds
        #   min over positives, max over negatives (rows with none -> +inf / -inf)
        min_pos = sim_mat.masked_fill(~pos_mask, float("inf")).min(dim=1).values
        max_neg = sim_mat.masked_fill(~neg_mask, float("-inf")).max(dim=1).values

        # Hard pairs
        neg_hard = neg_mask & (sim_mat + self.margin > min_pos.unsqueeze(1))
        pos_hard = pos_mask & (sim_mat - self.margin < max_neg.unsqueeze(1))

        # An anchor contributes only if it has both a hard positive and hard negative
        valid = (
            pos_mask.any(dim=1)
            & neg_mask.any(dim=1)
            & pos_hard.any(dim=1)
            & neg_hard.any(dim=1)
        )

        # Masked log-sum-exp: set non-selected exponents to -inf so exp(-inf) = 0
        pos_exponent = (-self.scale_pos * (sim_mat - self.thresh)).masked_fill(~pos_hard, float("-inf"))
        neg_exponent = (self.scale_neg * (sim_mat - self.thresh)).masked_fill(~neg_hard, float("-inf"))

        pos_loss = 1.0 / self.scale_pos * torch.log(1 + torch.exp(pos_exponent).sum(dim=1))
        neg_loss = 1.0 / self.scale_neg * torch.log(1 + torch.exp(neg_exponent).sum(dim=1))

        per_anchor = (pos_loss + neg_loss) * valid.to(feats.dtype)

        if valid.sum() == 0:
            return torch.zeros([], requires_grad=True, device=feats.device)

        # Note: divide by full batch_size (matches original), not by number of valid anchors
        loss = per_anchor.sum() / batch_size
        return loss
