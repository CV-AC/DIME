from collections import Counter
import math
import pickle

import pytest
import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from DIME_VIT.config import load_config
from DIME_VIT.data import (
    DistributedDisjointPairBatchSampler,
    DistributedIdentityPairBatchSampler,
    build_train_loader,
)
from DIME_VIT.engine import WarmupCosineScheduler, train_one_epoch


@pytest.mark.parametrize(
    "mode,sampler_type",
    [
        ("identity_uniform", DistributedIdentityPairBatchSampler),
        ("image_disjoint", DistributedDisjointPairBatchSampler),
    ],
)
def test_loader_selects_configured_pairing(tmp_path, mode, sampler_type):
    path = tmp_path / "faces.lmdb"
    env = lmdb.open(str(path), map_size=1 << 20)
    with env.begin(write=True) as transaction:
        transaction.put(
            b"__meta__", pickle.dumps({"num_samples": 4, "jpeg_encoded": False})
        )
        transaction.put(b"__identity_index__", pickle.dumps({0: [0, 1], 1: [2, 3]}))
        for index in range(4):
            transaction.put(
                f"{index:08d}".encode(),
                pickle.dumps(
                    {
                        "image": np.zeros((32, 32, 3), dtype=np.uint8),
                        "label": index // 2,
                    }
                ),
            )
    env.close()
    loader, sampler = build_train_loader(
        path,
        input_size=32,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        pair_sampling=mode,
    )
    assert isinstance(sampler, sampler_type)
    assert len(list(loader)) == 2


def test_default_pairing_is_unchanged_and_unknown_modes_are_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("{}\n")
    assert load_config(path).data.pair_sampling == "identity_uniform"
    assert (
        load_config(path, ["data.pair_sampling=image_disjoint"]).data.pair_sampling
        == "image_disjoint"
    )
    with pytest.raises(ValueError, match="pair_sampling"):
        load_config(path, ["data.pair_sampling=unknown"])


def pairs_from_batches(sampler):
    pairs = []
    for batch in sampler:
        half = len(batch) // 2
        assert half > 0 and len(batch) % 2 == 0
        pairs.extend(zip(batch[:half], reversed(batch[half:])))
    return pairs


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_even_identity_images_are_used_once_across_ranks(world_size):
    identities = {label: list(range(4 * label, 4 * label + 4)) for label in range(8)}
    labels = {image: label for label, images in identities.items() for image in images}
    used = []
    lengths = []
    for rank in range(world_size):
        sampler = DistributedDisjointPairBatchSampler(
            identities, batch_size=6, dataset_size=32, rank=rank, world_size=world_size
        )
        pairs = pairs_from_batches(sampler)
        lengths.append(len(sampler))
        assert len(pairs) == sampler.pairs_per_rank
        for first, second in pairs:
            assert first != second and labels[first] == labels[second]
            used.extend([first, second])
    assert len(set(lengths)) == 1
    assert Counter(used) == Counter(range(32))


def test_singletons_odd_images_and_distributed_tail_are_counted(caplog):
    identities = {0: [0], 1: [1, 2, 3], 2: [4, 5, 6, 7, 8], 3: [9, 10, 11, 12]}
    used = []
    caplog.set_level("INFO")
    for rank in range(2):
        sampler = DistributedDisjointPairBatchSampler(
            identities, batch_size=6, dataset_size=14, rank=rank, world_size=2
        )
        assert sampler.singleton_images == 1
        assert sampler.odd_identity_images == 2
        assert sampler.unindexed_images == 1
        assert sampler.total_pairs == 5
        used.extend(image for pair in pairs_from_batches(sampler) for image in pair)
    assert len(used) == len(set(used)) == 8
    assert 0 not in used and 13 not in used
    assert "singleton images 1" in caplog.text
    assert "odd-identity leftover images 2" in caplog.text
    assert "distributed-tail pairs 1" in caplog.text
    assert "unindexed images 1" in caplog.text


def test_epoch_seed_replays_pairs_and_rotates_odd_leftovers():
    sampler = DistributedDisjointPairBatchSampler({0: list(range(9))}, 4, 9, seed=7)
    sampler.set_epoch(3)
    first = pairs_from_batches(sampler)
    assert first == pairs_from_batches(sampler)
    missing = set()
    for epoch in range(32):
        sampler.set_epoch(epoch)
        used = {image for pair in pairs_from_batches(sampler) for image in pair}
        assert len(used) == 8
        missing.update(set(range(9)) - used)
    assert missing == set(range(9))


def test_resolution_batch_changes_preserve_pairs_and_update_count():
    identities = {label: list(range(6 * label, 6 * label + 6)) for label in range(11)}
    for rank in range(4):
        low = DistributedDisjointPairBatchSampler(identities, 8, 66, rank, 4, seed=3)
        high = DistributedDisjointPairBatchSampler(identities, 2, 66, rank, 4, seed=3)
        low.set_epoch(5)
        high.set_epoch(5)
        assert pairs_from_batches(low) == pairs_from_batches(high)
        assert math.ceil(len(low) / 1) == math.ceil(len(high) / 4)


@pytest.mark.parametrize(
    "identities,size",
    [
        ({0: [0, 0]}, 2),
        ({0: [0, 1], 1: [1, 2]}, 3),
        ({0: [-1, 0]}, 2),
        ({0: [0, 2]}, 2),
    ],
)
def test_invalid_identity_indexes_are_rejected(identities, size):
    with pytest.raises(ValueError):
        DistributedDisjointPairBatchSampler(identities, 2, size)


def test_small_data_never_duplicates_pairs_to_fill_ranks():
    with pytest.raises(ValueError, match="every rank"):
        DistributedDisjointPairBatchSampler({0: [0, 1]}, 2, 2, world_size=2)
    with pytest.raises(ValueError, match="non-repeating epoch"):
        DistributedDisjointPairBatchSampler({0: [0, 1]}, 2, 2, num_batches=2)


def test_batch_limit_only_shortens_a_non_repeating_epoch():
    sampler = DistributedDisjointPairBatchSampler(
        {0: list(range(12))}, 4, 12, num_batches=2
    )
    used = [image for pair in pairs_from_batches(sampler) for image in pair]
    assert len(used) == len(set(used)) == sampler.num_samples == 8
    assert len(sampler) == 2


def test_partial_accumulation_weights_images_equally():
    class Regression(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def forward(self, images, **options):
            loss = (self.weight - images).square().mean()
            return {"loss": loss, "loss_rgb": loss}

    sampler = DistributedDisjointPairBatchSampler({0: list(range(10))}, 4, 10)
    loader = DataLoader(
        TensorDataset(torch.arange(10).float().view(-1, 1)), batch_sampler=sampler
    )
    model = Regression()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = WarmupCosineScheduler(optimizer, total_updates=2, warmup_updates=0)
    _, updates = train_one_epoch(
        model,
        loader,
        optimizer,
        scheduler,
        torch.device("cpu"),
        0,
        accum_iter=3,
        amp_dtype="fp32",
        log_freq=0,
    )
    assert updates == 1
    torch.testing.assert_close(model.weight, torch.tensor(0.9))
