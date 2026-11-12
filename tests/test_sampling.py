"""Test sampler theo luỹ thừa và ngân sách bước cố định mỗi epoch (empirical plan mục 8.1). Không cần model, GPU."""
from collections import Counter

import pytest
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from vimmsd.data.dataset import EpochBudgetSampler, build_train_sampler, sampling_weights


class _DS(torch.utils.data.Dataset):
    def __init__(self, labels):
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return i


def test_sampling_weights_power():
    labels = [0] * 100 + [1] * 4
    w = sampling_weights(labels, 0.5)
    assert w[0] == pytest.approx(0.1) and w[-1] == pytest.approx(0.5)
    # luỹ thừa 0.5: tổng trọng số lớp tỉ lệ với sqrt(count) -> 10 : 2
    total = Counter()
    for lab, x in zip(labels, w):
        total[lab] += x
    assert total[0] / total[1] == pytest.approx(5.0)
    assert sampling_weights(labels, 1.0)[0] * 100 == pytest.approx(sampling_weights(labels, 1.0)[-1] * 4)
    assert set(sampling_weights(labels, 0.0)) == {1.0}


def test_build_train_sampler_modes():
    ds = _DS([0] * 10 + [1] * 2)
    assert build_train_sampler(ds, 2, "random") is None
    s = build_train_sampler(ds, 2, "power", power=0.5, num_samples=40)
    assert isinstance(s, WeightedRandomSampler) and len(s) == 40
    old = build_train_sampler(ds, 2, "weighted")
    assert len(old) == 12 and old.weights[0] * 10 == pytest.approx(old.weights[-1] * 2)
    with pytest.raises(ValueError, match="sampler_power"):
        build_train_sampler(ds, 2, "power")
    with pytest.raises(ValueError):
        build_train_sampler(ds, 2, "uniform")


def test_epoch_budget_sampler_covers_without_repeats_and_carries_over():
    torch.manual_seed(0)
    s = EpochBudgetSampler(10, 7)
    first, second = list(s), list(s)
    assert len(first) == len(second) == 7 and len(set(first)) == 7
    # 14 chỉ số đầu = một hoán vị đủ 10 + 4 của hoán vị sau: mỗi chỉ số xuất hiện trước khi có chỉ số nào lặp lại
    assert set((first + second)[:10]) == set(range(10))


def test_steps_per_epoch_equalises_loader_length_across_dataset_sizes():
    torch.manual_seed(0)
    budget = 5 * 4   # 5 bước × batch 4
    for n in (12, 30):   # tập gốc và tập đã thêm augmentation
        for mode, power in (("random", None), ("power", 0.5)):
            ds = _DS([i % 3 for i in range(n)])
            loader = DataLoader(ds, batch_size=4, sampler=build_train_sampler(ds, 3, mode, power, budget))
            assert len(loader) == 5 and sum(len(b) for b in loader) == budget
