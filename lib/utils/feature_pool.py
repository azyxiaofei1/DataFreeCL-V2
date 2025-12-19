import collections
from typing import Optional, Tuple

import torch


class FeaturePool:
    def __init__(self, max_samples: int, store_device: str = "cpu") -> None:
        self.max_samples = int(max_samples)
        self.store_device = store_device
        self._features = collections.deque()
        self._labels = collections.deque()
        self._total_count = 0
        self._feat_dim = None

    def __len__(self) -> int:
        return self._total_count

    def enqueue(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        feats = features.detach().to(self.store_device)
        labs = labels.detach().to(self.store_device)

        assert feats.dim() == 2, "features must be 2D [N, D]"
        assert labs.dim() == 1 and labs.size(0) == feats.size(0), "labels must be 1D and match features batch size"

        if self._feat_dim is None:
            self._feat_dim = feats.size(1)

        self._features.append(feats)
        self._labels.append(labs)
        self._total_count += feats.size(0)

        self._trim_excess()

    def dequeue(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_samples <= 0 or self._total_count == 0:
            return self._empty_tensors()

        if num_samples >= self._total_count:
            feats, labs = self.get_all()
            self._clear()
            return feats, labs

        taken_feats = []
        taken_labs = []
        remaining = num_samples

        while remaining > 0:
            batch_feats = self._features[0]
            batch_labs = self._labels[0]
            batch_size = batch_feats.size(0)

            if batch_size <= remaining:
                taken_feats.append(batch_feats)
                taken_labs.append(batch_labs)
                self._features.popleft()
                self._labels.popleft()
                remaining -= batch_size
                self._total_count -= batch_size
            else:
                taken_feats.append(batch_feats[:remaining])
                taken_labs.append(batch_labs[:remaining])
                self._features[0] = batch_feats[remaining:]
                self._labels[0] = batch_labs[remaining:]
                self._total_count -= remaining
                remaining = 0

        feats_out = torch.cat(taken_feats, dim=0)
        labs_out = torch.cat(taken_labs, dim=0)
        return feats_out, labs_out

    def get_all(
        self, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._total_count == 0:
            return self._empty_tensors(device=device)

        feats = torch.cat(list(self._features), dim=0)
        labs = torch.cat(list(self._labels), dim=0)
        if device is not None:
            feats = feats.to(device)
            labs = labs.to(device)
        return feats, labs

    def _trim_excess(self) -> None:
        if self._total_count <= self.max_samples:
            return

        overflow = self._total_count - self.max_samples
        while overflow > 0 and self._features:
            batch_feats = self._features[0]
            batch_labs = self._labels[0]
            batch_size = batch_feats.size(0)

            if batch_size <= overflow:
                self._features.popleft()
                self._labels.popleft()
                self._total_count -= batch_size
                overflow -= batch_size
            else:
                self._features[0] = batch_feats[overflow:]
                self._labels[0] = batch_labs[overflow:]
                self._total_count -= overflow
                overflow = 0

    def _clear(self) -> None:
        self._features.clear()
        self._labels.clear()
        self._total_count = 0

    def _empty_tensors(
        self, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tgt_device = device if device is not None else self.store_device
        dim = self._feat_dim or 0
        return (
            torch.empty((0, dim), device=tgt_device),
            torch.empty(0, dtype=torch.long, device=tgt_device),
        )
