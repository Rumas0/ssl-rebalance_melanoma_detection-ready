import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """v1 loss, unchanged. Kept for direct comparison against AdaptiveFocalLoss."""
    def __init__(self, alpha=None, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


class AdaptiveFocalLoss(nn.Module):
    """
    v2 contribution: focal loss + adaptive false-positive penalty.

    L = L_focal + lambda_t * L_FP

    where L_FP is the mean MEL-probability assigned to non-MEL samples in the
    current batch, and lambda_t is an exponentially-smoothed function of the
    running L_FP. When the classifier produces high MEL-confidence on benign
    samples, lambda_t rises automatically, increasing pressure to be precise.

    Parameters
    ----------
    alpha : torch.Tensor or None
        Per-class weights for the focal term (same as v1).
    gamma : float
        Focal focusing parameter.
    lambda_0 : float
        Base false-positive penalty.
    kappa : float
        Sensitivity of the adaptive penalty to the smoothed FP rate.
    beta_ema : float
        EMA decay for the false-positive signal.
    mel_index : int
        Index of the melanoma class in the output logit ordering.
    lambda_max : float
        Upper clamp to prevent runaway penalty scaling.
    """
    def __init__(self, alpha=None, gamma=1.5, lambda_0=0.5, kappa=2.0,
                 beta_ema=0.9, mel_index=1, lambda_max=5.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_0 = lambda_0
        self.kappa = kappa
        self.beta_ema = beta_ema
        self.mel_index = mel_index
        self.lambda_max = lambda_max
        # register_buffer so fp_ema survives state_dict save/load
        self.register_buffer('fp_ema', torch.zeros(1))

    def forward(self, logits, targets):
        # ---- focal term (identical to v1) ----
        ce = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma * ce).mean()

        # ---- false-positive surrogate ----
        probs = F.softmax(logits, dim=1)
        non_mel = (targets != self.mel_index)
        if non_mel.any():
            fp_batch = probs[non_mel, self.mel_index].mean()
        else:
            fp_batch = torch.zeros(1, device=logits.device, dtype=logits.dtype)

        # ---- EMA update (detached — this is a statistic, not a gradient path) ----
        with torch.no_grad():
            self.fp_ema.mul_(self.beta_ema).add_((1 - self.beta_ema) * fp_batch.detach())

        # ---- adaptive lambda, clamped to prevent explosion ----
        lam = self.lambda_0 * torch.exp(self.kappa * self.fp_ema)
        lam = torch.clamp(lam, max=self.lambda_max)

        return focal + lam * fp_batch

    def extra_repr(self):
        return (f'gamma={self.gamma}, lambda_0={self.lambda_0}, '
                f'kappa={self.kappa}, beta_ema={self.beta_ema}, '
                f'mel_index={self.mel_index}')