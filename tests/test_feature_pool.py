import os
import sys

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from lib.utils.feature_pool import FeaturePool


def test_enqueue_and_len():
    pool = FeaturePool(max_samples=10)
    pool.enqueue(torch.ones(2, 3), torch.tensor([0, 1]))
    pool.enqueue(torch.ones(3, 3), torch.tensor([2, 3, 4]))
    assert len(pool) == 5


def test_fifo_trim_on_overflow():
    pool = FeaturePool(max_samples=5)
    pool.enqueue(torch.arange(3).float().unsqueeze(1), torch.arange(3))
    pool.enqueue(torch.arange(3, 7).float().unsqueeze(1), torch.arange(3, 7))

    feats, labs = pool.get_all()
    expected = torch.arange(2, 7).float().unsqueeze(1)
    assert feats.shape == (5, 1)
    assert torch.equal(feats, expected)
    assert torch.equal(labs, torch.arange(2, 7))


def test_dequeue_order():
    pool = FeaturePool(max_samples=10)
    pool.enqueue(torch.tensor([[1.0], [2.0]]), torch.tensor([1, 2]))
    pool.enqueue(torch.tensor([[3.0], [4.0]]), torch.tensor([3, 4]))

    feats, labs = pool.dequeue(3)
    assert torch.equal(feats, torch.tensor([[1.0], [2.0], [3.0]]))
    assert torch.equal(labs, torch.tensor([1, 2, 3]))
    assert len(pool) == 1

    remaining_feats, remaining_labs = pool.get_all()
    assert torch.equal(remaining_feats, torch.tensor([[4.0]]))
    assert torch.equal(remaining_labs, torch.tensor([4]))


def test_empty_shapes_after_clear():
    pool = FeaturePool(max_samples=3)
    pool.enqueue(torch.tensor([[1.0, 2.0]]), torch.tensor([0]))
    pool.dequeue(1)
    feats, labs = pool.get_all()
    assert feats.shape == (0, 2)
    assert labs.shape == (0,)


def test_get_all_device():
    pool = FeaturePool(max_samples=4)
    pool.enqueue(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([0, 1]))

    device = torch.device("cpu")
    feats, labs = pool.get_all(device=device)
    assert feats.shape == (2, 2)
    assert labs.shape == (2,)
    assert feats.device == device
    assert labs.device == device


if __name__ == "__main__":
    test_enqueue_and_len()
    test_fifo_trim_on_overflow()
    test_dequeue_order()
    test_get_all_device()
    print("All feature pool tests passed.")
