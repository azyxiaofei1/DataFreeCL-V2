import torch


def curvature_balanced_loss(
    Gi: torch.Tensor, eps: float = 1e-8, max_inv_const: "torch.Tensor | None" = None
) -> torch.Tensor:
    """
    Curvature-balanced loss on the last dimension of Gi.
    """
    Gi = torch.clamp(Gi, min=eps)
    inv = 1.0 / (Gi + eps)
    if max_inv_const is None:
        max_inv, _ = inv.max(dim=-1, keepdim=True)
    else:
        max_inv = torch.as_tensor(max_inv_const, device=inv.device, dtype=inv.dtype)
        if max_inv.dim() == 0:
            max_inv = max_inv.view(*([1] * (inv.dim() - 1)), 1)
        elif max_inv.dim() == inv.dim() - 1:
            max_inv = max_inv.unsqueeze(-1)
    ratio = inv / max_inv
    ratio = torch.clamp(ratio, min=eps)
    L_curv = (-torch.log(ratio)).sum(dim=-1).mean()
    return L_curv


def log_tau_epoch_weight(epoch: int, tau: float) -> torch.Tensor:
    """
    Compute log(epoch) / log(tau), epoch starts from 1 and tau > 1.
    """
    if tau <= 1:
        raise ValueError("tau must be greater than 1.")
    if epoch <= 1:
        return torch.tensor(0.0)
    epoch_tensor = torch.tensor(float(epoch))
    tau_tensor = torch.tensor(float(tau))
    return torch.log(epoch_tensor) / torch.log(tau_tensor)


def combine_loss(
    L_orig: torch.Tensor,
    L_curv: torch.Tensor,
    epoch: int,
    tau: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Combine original loss with curvature-balanced loss.
    """
    weight = torch.as_tensor(
        log_tau_epoch_weight(epoch, tau), device=L_orig.device, dtype=L_orig.dtype
    )
    scaled_curv = (L_curv / (L_orig + eps)).detach()
    L_total = L_orig + weight * scaled_curv * L_curv
    return L_total
