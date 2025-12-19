import os
import sys

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from lib.model.curvature_balanced import (
    combine_loss,
    curvature_balanced_loss,
    log_tau_epoch_weight,
)
from lib.continual.progressive_DFCL import progressive_DFCL


def test_curvature_loss_zero_when_equal():
    Gi = torch.ones(5)
    L_curv = curvature_balanced_loss(Gi)
    assert torch.isclose(L_curv, torch.tensor(0.0), atol=1e-7)


def test_curvature_loss_positive_when_differs():
    Gi = torch.tensor([1.0, 2.0, 4.0])
    L_curv = curvature_balanced_loss(Gi)
    assert L_curv > 0


def test_curvature_loss_handles_non_positive():
    Gi = torch.tensor([-1.0, 0.0, 2.0])
    L_curv = curvature_balanced_loss(Gi)
    assert torch.isfinite(L_curv)


def test_log_tau_epoch_weight_epoch_le_one():
    weight = log_tau_epoch_weight(epoch=1, tau=2.0)
    assert torch.isclose(weight, torch.tensor(0.0))


def test_log_tau_epoch_weight_invalid_tau():
    try:
        _ = log_tau_epoch_weight(epoch=2, tau=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError not raised when tau <= 1")


def test_combine_loss_backward_pass():
    Gi = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)
    L_curv = curvature_balanced_loss(Gi)
    L_orig = torch.tensor(0.5, requires_grad=True)

    total_loss = combine_loss(L_orig, L_curv, epoch=2, tau=2.0)
    total_loss.backward()

    assert Gi.grad is not None
    assert torch.all(torch.isfinite(Gi.grad))
    assert torch.sum(Gi.grad.abs()) > 0


def test_curvature_proxy_grad_backward():
    features = torch.randn(8, 4, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    Gi = progressive_DFCL.curvature_proxy_per_class_grad(
        features, labels, num_samples_per_class=32, k_neighbors=2
    )
    assert Gi.numel() > 0
    Gi = Gi + torch.linspace(0, 0.1, Gi.numel(), device=Gi.device)
    L_curv = curvature_balanced_loss(Gi)
    L_curv.backward()
    assert features.grad is not None
    assert torch.all(torch.isfinite(features.grad))
    assert torch.sum(features.grad.abs()) > 0


if __name__ == "__main__":
    test_curvature_loss_zero_when_equal()
    test_curvature_loss_positive_when_differs()
    test_curvature_loss_handles_non_positive()
    test_log_tau_epoch_weight_epoch_le_one()
    test_log_tau_epoch_weight_invalid_tau()
    test_combine_loss_backward_pass()
    test_curvature_proxy_grad_backward()
    print("All curvature-balanced tests passed.")
