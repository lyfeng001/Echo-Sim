"""Deterministic chunk-level sampling for a selected Echo data profile."""
from __future__ import annotations

from collections.abc import Mapping
import math
import operator

import numpy as np

def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


class EchoWeightedSampler:
    """Sample frames using global type weights and deterministic rank slices.

    Each block chooses a type, a length-weighted segment, then uniform frames
    within that segment. Thus frame marginals obey the requested type mixture,
    while blocks improve the episode cache's locality. Weights are probabilities,
    not exact quotas; realized type counts are recorded for every epoch.

    Sampling uses replacement, so ranks receive disjoint sequence positions but
    may see the same source frame. Equal rank lengths preserve DDP's update count
    and keep the last short batch. Up to ``num_replicas - 1`` extra samples round
    the global epoch size up to a multiple of the replica count.
    """

    def __init__(self, dataset, num_replicas=1, rank=0, seed=42,
                 samples_per_epoch=320000, weights=None, block_size=32,
                 require_all_types=True, data_profile="three_type"):
        self.dataset = dataset
        self.num_replicas = _integer(num_replicas, "num_replicas", 1)
        self.rank = _integer(rank, "rank")
        self.seed = _integer(seed, "seed")
        self.samples_per_epoch = _integer(samples_per_epoch, "samples_per_epoch", 1)
        self.block_size = _integer(block_size, "block_size", 1)
        if self.rank >= self.num_replicas:
            raise ValueError("rank must be below num_replicas")
        if self.samples_per_epoch < self.num_replicas:
            raise ValueError("samples_per_epoch must be at least num_replicas")
        if type(require_all_types) is not bool:
            raise ValueError("require_all_types must be a boolean")
        self.data_profile = data_profile
        kinds = ("success", "failure", "recovery") if data_profile == "three_type" else ("success", "recovery")
        if data_profile not in {"three_type", "baseline"}:
            raise ValueError("Expected three_type or baseline")
        supplied = {k: {"success": 40, "failure": 1, "recovery": 20}[k] for k in kinds} if weights is None else weights
        if not isinstance(supplied, Mapping) or set(supplied) != set(kinds):
            raise ValueError("weights must name exactly the selected Echo profile types")
        for kind, weight in supplied.items():
            if (isinstance(weight, (bool, np.bool_))
                    or not isinstance(weight, (int, float, np.number))
                    or not math.isfinite(weight) or weight <= 0):
                raise ValueError(f"weight for {kind} must be finite and positive")
        # Canonical ordering prevents JSON key reordering from changing samples.
        self.weights = {kind: float(supplied[kind]) for kind in kinds}

        self.offsets = np.asarray([
            _integer(value, "dataset offset") for value in dataset.offsets
        ], dtype=np.int64)
        if (len(self.offsets) != len(dataset.segments) + 1
                or not len(self.offsets) or self.offsets[0] != 0
                or np.any(np.diff(self.offsets) <= 0)):
            raise ValueError("dataset offsets must start at zero and bound non-empty segments")
        self.episode_by_type = {kind: [] for kind in self.weights}
        for index, segment in enumerate(dataset.segments):
            kind = segment["segment_type"]
            if kind not in self.episode_by_type:
                raise ValueError(f"Unknown Echo segment type: {kind}")
            self.episode_by_type[kind].append(index)
        missing = [kind for kind, rows in self.episode_by_type.items() if not rows]
        if missing and require_all_types:
            raise ValueError("Echo manifest lacks required data types: " + ", ".join(missing))
        self.kinds = [kind for kind in self.weights if self.episode_by_type[kind]]
        if not self.kinds:
            raise ValueError("Echo manifest has no sampleable segments")
        probabilities = np.asarray([self.weights[kind] for kind in self.kinds])
        # Scale before summing to keep large, individually finite weights valid.
        probabilities /= probabilities.max()
        self.probabilities = probabilities / probabilities.sum()
        self.candidates = {}
        self.segment_probabilities = {}
        for kind in self.kinds:
            candidates = np.asarray(self.episode_by_type[kind], dtype=np.int64)
            lengths = self.offsets[candidates + 1] - self.offsets[candidates]
            self.candidates[kind] = candidates
            self.segment_probabilities[kind] = lengths / lengths.sum()
        self._full_num_samples = math.ceil(self.samples_per_epoch / self.num_replicas)
        self.effective_samples = self._full_num_samples * self.num_replicas
        self.set_epoch(0)

    def set_epoch(self, epoch, batch_offset=0, batch_size=None):
        self.epoch = _integer(epoch, "epoch")
        self.batch_offset = _integer(batch_offset, "batch_offset")
        self.batch_size = None if batch_size is None else _integer(batch_size, "batch_size", 1)
        if self.batch_offset and self.batch_size is None:
            raise ValueError("batch_offset requires batch_size")
        if self.batch_size is not None:
            batch_count = math.ceil(self.full_num_samples / self.batch_size)
            if self.batch_offset > batch_count:
                raise ValueError("batch_offset exceeds the epoch batch count")
        # Each rank generates the same canonical epoch, then takes its contiguous
        # slice. This provides reproducible global counts without a collective.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 9127]))
        all_indices = np.empty(self.effective_samples, dtype=np.int64)
        all_types = np.empty(self.effective_samples, dtype=np.uint8)
        for begin in range(0, self.effective_samples, self.block_size):
            end = min(begin + self.block_size, self.effective_samples)
            type_index = int(rng.choice(len(self.kinds), p=self.probabilities))
            kind = self.kinds[type_index]
            segment_index = int(rng.choice(
                self.candidates[kind], p=self.segment_probabilities[kind],
            ))
            start, stop = self.offsets[segment_index:segment_index + 2]
            all_indices[begin:end] = start + rng.integers(stop - start, size=end - begin)
            all_types[begin:end] = type_index
        begin = self.rank * self.full_num_samples
        end = begin + self.full_num_samples
        self.indices = all_indices[begin:end].tolist()
        self.type_counts = {kind: 0 for kind in self.weights}
        self.global_type_counts = dict(self.type_counts)
        for index, kind in enumerate(self.kinds):
            self.type_counts[kind] = int(np.count_nonzero(all_types[begin:end] == index))
            self.global_type_counts[kind] = int(np.count_nonzero(all_types == index))
        self.start_offset = 0 if self.batch_size is None else min(
            self.batch_offset * self.batch_size, self.full_num_samples,
        )

    def __iter__(self):
        return iter(self.indices[self.start_offset:])

    def __len__(self):
        return len(self.indices) - self.start_offset

    @property
    def full_num_samples(self):
        return self._full_num_samples

    def audit(self):
        return {
            "sampler": type(self).__name__,
            "data_profile": self.data_profile,
            "seed": self.seed,
            "epoch": self.epoch,
            "rank": self.rank,
            "num_replicas": self.num_replicas,
            "samples_per_epoch": self.samples_per_epoch,
            "weights": dict(self.weights),
            "type_counts": dict(self.type_counts),
            "global_type_counts": dict(self.global_type_counts),
            "sampling_unit": "chunk_start_frame",
            "ratio_scope": "global_types",
            "sampling_with_replacement": True,
            "rank_partition": "canonical_epoch_contiguous_slices",
            "block_size": self.block_size,
            "effective_global_samples": self.effective_samples,
            "per_rank_epoch_samples": self.full_num_samples,
            "batch_offset": self.batch_offset,
            "batch_size": self.batch_size,
        }
