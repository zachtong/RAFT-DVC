"""On-the-fly (OTF) Phase-1 dataset.

Each worker holds its own ``Phase1SampleGenerator`` and synthesizes a brand
new (I1, I2, flow) sample on every ``__getitem__`` call.  No disk reads, no
caching.  This is the "infinite-data" paradigm: every gradient update sees
fresh data, and the only limit on sample diversity is the CPU generator
throughput.

Why it works
------------
* The Phase-1 generator is fast: ~10-50 ms per sample at size=32, scaling
  roughly as O(size^3) plus an O(N_beads^2) RSA placement cost.
* PyTorch ``DataLoader(num_workers=N, prefetch_factor=K)`` runs N parallel
  generator instances; the GPU pulls from a pre-filled queue.
* For input size >= 32, GPU step time exceeds CPU generation time, so
  workers stay ahead of the GPU even at modest N (4-8).

Determinism
-----------
Sample seeds combine ``seed_base`` + worker_id + idx so:
  * Two different workers never collide on the same seed within an epoch.
  * Two different epochs at the same idx produce different samples (because
    ``__getitem__`` is called many times with the same idx but the dataset
    object is freshly re-initialized per epoch in PyTorch when
    ``persistent_workers`` is False; with ``persistent_workers=True`` the
    same worker keeps incrementing samples through epochs naturally).

The ``length`` argument only controls how many samples constitute one
epoch (used by the LR scheduler).  It does not bound the number of unique
samples the generator can produce.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Lazy import inside class to avoid pulling the heavy generator module at
# library import time (matters for inference tooling).


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _import_generator():
    """Import ``Phase1SampleGenerator`` lazily so this module can be imported
    even if the scripts/ path is not yet on ``sys.path``."""
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    gen_dir = repo_root / "scripts" / "data_generation"
    if str(gen_dir) not in sys.path:
        sys.path.insert(0, str(gen_dir))
    from modules.sample_generator import Phase1SampleGenerator  # noqa: WPS433
    return Phase1SampleGenerator


# Mixed config alphabet (must match scripts/generate_phase1_dataset.py)
_MIXED_RADII: Tuple[int, ...] = (2, 4, 6)
_MIXED_DENSITIES: Tuple[Tuple[str, float], ...] = (
    ("sparse", 0.2),
    ("medium", 0.6),
    ("dense", 1.0),
)


# -----------------------------------------------------------------------------
# Base OTF dataset (single config)
# -----------------------------------------------------------------------------

class OnTheFlyPhase1Dataset(Dataset):
    """Generates Phase-1 samples per-call from a fixed (radius, density).

    Drop-in replacement for ``Phase1NPZDataset``: returns a dict with keys
    ``vol0`` (1, D, H, W), ``vol1``, ``flow`` (3, D, H, W), ``filename``,
    and optional ``meta`` (when ``include_metadata=True``).

    Augmentation is not applied here because the generator already
    randomises bead positions, displacement shape, and noise per sample --
    the dataset itself is the augmentation.  Use ``augment=True`` only if
    you want to add flips/rotations on top of generation diversity.
    """

    def __init__(
        self,
        size: int,
        radius: int,
        density_per_1000: float,
        length: int = 1000,
        feature_map_size: int = 16,
        seed_base: int = 42_000_000,
        include_metadata: bool = True,
        augment: bool = False,
    ):
        self.size = int(size)
        self.radius = int(radius)
        self.density_per_1000 = float(density_per_1000)
        self.length = int(length)
        self.feature_map_size = int(feature_map_size)
        self.seed_base = int(seed_base)
        self.include_metadata = bool(include_metadata)
        self.augment = bool(augment)

        # Lazy: per-worker generator instance + monotonic call counter.
        # CRITICAL: each ``__getitem__`` advances the counter so the seed
        # depends on call-history, NOT on the idx argument.  If we keyed
        # the seed off idx, each epoch would regenerate the exact same
        # 1000 samples (idx is reset every epoch by DataLoader), making
        # OTF mode equivalent to a fixed NPZ pool.
        self._gen = None
        self._gen_cls = None
        self._call_count = 0

    # -------------------------------------------------------------------------

    def _ensure_generator(self) -> None:
        if self._gen is not None:
            return
        if self._gen_cls is None:
            self._gen_cls = _import_generator()
        self._gen = self._gen_cls(
            size=self.size,
            radius=self.radius,
            density_per_1000=self.density_per_1000,
            feature_map_size=self.feature_map_size,
        )

    @staticmethod
    def _worker_seed_offset() -> int:
        info = torch.utils.data.get_worker_info()
        return int(info.id) if info is not None else 0

    @staticmethod
    def _augment_pair(
        I1: np.ndarray,
        I2: np.ndarray,
        flow: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Optional on-top augmentation: flips + 90-deg rotations."""
        for axis in range(3):
            if rng.random() > 0.5:
                I1 = np.flip(I1, axis=axis)
                I2 = np.flip(I2, axis=axis)
                flow = np.flip(flow, axis=axis + 1)
                flow[axis] = -flow[axis]
        k = int(rng.integers(0, 4))
        if k > 0:
            I1 = np.rot90(I1, k, axes=(1, 2))
            I2 = np.rot90(I2, k, axes=(1, 2))
            flow = np.rot90(flow, k, axes=(2, 3))
            for _ in range(k):
                fy, fx = flow[1].copy(), flow[2].copy()
                flow[1] = -fx
                flow[2] = fy
        return (
            np.ascontiguousarray(I1),
            np.ascontiguousarray(I2),
            np.ascontiguousarray(flow),
        )

    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        self._ensure_generator()
        wid = self._worker_seed_offset()
        # Monotonic per-worker counter (NOT idx) drives the seed so every
        # call within a worker gets a fresh sample.  100M offset between
        # workers makes collisions impossible until each worker has done
        # > 100M calls.
        self._call_count += 1
        seed = self.seed_base + wid * 100_000_007 + self._call_count
        sample = self._gen.generate(seed)
        I1 = sample["I1"]
        I2 = sample["I2"]
        flow = sample["flow"]

        if self.augment:
            aug_rng = np.random.default_rng(seed + 991)
            I1, I2, flow = self._augment_pair(I1, I2, flow, aug_rng)

        result: Dict[str, Any] = {
            "vol0": torch.from_numpy(I1.astype(np.float32)).unsqueeze(0),
            "vol1": torch.from_numpy(I2.astype(np.float32)).unsqueeze(0),
            "flow": torch.from_numpy(flow.astype(np.float32)),
            "filename": f"otf_w{wid}_c{self._call_count}_s{seed}",
        }
        if self.include_metadata:
            result["meta"] = dict(sample["metadata"])
        return result


# -----------------------------------------------------------------------------
# Mixed OTF dataset (radius × density randomly sampled per call)
# -----------------------------------------------------------------------------

class OnTheFlyMixedDataset(Dataset):
    """Mixed config: each ``__getitem__`` picks a random (radius, density)
    from the 9 single configs and generates one sample with that combo.

    Each worker pre-instantiates one generator per combo (9 generators per
    worker) to avoid repeated initialisation cost.
    """

    def __init__(
        self,
        size: int,
        length: int = 1000,
        feature_map_size: int = 16,
        seed_base: int = 43_000_000,
        include_metadata: bool = True,
        augment: bool = False,
    ):
        self.size = int(size)
        self.length = int(length)
        self.feature_map_size = int(feature_map_size)
        self.seed_base = int(seed_base)
        self.include_metadata = bool(include_metadata)
        self.augment = bool(augment)

        self._pool: Dict[Tuple[int, str], Any] = {}
        self._gen_cls = None
        self._call_count = 0

    def _ensure_pool(self) -> None:
        if self._pool:
            return
        if self._gen_cls is None:
            self._gen_cls = _import_generator()
        for radius in _MIXED_RADII:
            for dname, dval in _MIXED_DENSITIES:
                self._pool[(radius, dname)] = self._gen_cls(
                    size=self.size,
                    radius=radius,
                    density_per_1000=dval,
                    feature_map_size=self.feature_map_size,
                )

    @staticmethod
    def _worker_seed_offset() -> int:
        info = torch.utils.data.get_worker_info()
        return int(info.id) if info is not None else 0

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        self._ensure_pool()
        wid = self._worker_seed_offset()
        self._call_count += 1
        sel_seed = self.seed_base + wid * 100_000_007 + self._call_count
        sel_rng = np.random.default_rng(sel_seed)
        slot = int(sel_rng.integers(0, 9))
        radius = _MIXED_RADII[slot // 3]
        dname, _dval = _MIXED_DENSITIES[slot % 3]
        gen = self._pool[(radius, dname)]
        # Use a separate seed for generation so the mix-choice RNG doesn't
        # also drive bead placement (keeps the two factorised).
        gen_seed = sel_seed + 7_000_001
        sample = gen.generate(gen_seed)
        I1 = sample["I1"]
        I2 = sample["I2"]
        flow = sample["flow"]

        if self.augment:
            aug_rng = np.random.default_rng(gen_seed + 991)
            I1, I2, flow = OnTheFlyPhase1Dataset._augment_pair(
                I1, I2, flow, aug_rng,
            )

        meta = dict(sample["metadata"])
        meta["mix_choice"] = f"r{radius}_{dname}"

        result: Dict[str, Any] = {
            "vol0": torch.from_numpy(I1.astype(np.float32)).unsqueeze(0),
            "vol1": torch.from_numpy(I2.astype(np.float32)).unsqueeze(0),
            "flow": torch.from_numpy(flow.astype(np.float32)),
            "filename": f"otf_mixed_w{wid}_c{self._call_count}_r{radius}_{dname}",
        }
        if self.include_metadata:
            result["meta"] = meta
        return result


# -----------------------------------------------------------------------------
# Convenience: build the right OTF dataset from a data-config name
# -----------------------------------------------------------------------------

def parse_data_config_name(name: str) -> Dict[str, Any]:
    """Parse ``r{R}_{density}_size{S}`` or ``mixed_size{S}`` into params.

    Returns a dict with keys ``mixed`` (bool), ``size`` (int), and -- for
    single configs -- ``radius`` (int) and ``density`` (str).

    Raises ``ValueError`` on a malformed name.
    """
    if name.startswith("mixed_size"):
        try:
            size = int(name[len("mixed_size"):])
        except ValueError as exc:
            raise ValueError(f"Cannot parse size from {name!r}") from exc
        return {"mixed": True, "size": size}

    # r{R}_{density}_size{S}
    parts = name.split("_")
    if len(parts) != 3 or not parts[0].startswith("r") or not parts[2].startswith("size"):
        raise ValueError(
            f"Data-config name {name!r} does not match "
            f"'r{{R}}_{{density}}_size{{S}}' or 'mixed_size{{S}}'."
        )
    try:
        radius = int(parts[0][1:])
        size = int(parts[2][len("size"):])
    except ValueError as exc:
        raise ValueError(f"Cannot parse {name!r}: {exc}") from exc
    density = parts[1]
    if density not in {n for n, _ in _MIXED_DENSITIES}:
        raise ValueError(
            f"Unknown density {density!r} in {name!r}; expected one of "
            f"{[n for n, _ in _MIXED_DENSITIES]}."
        )
    return {"mixed": False, "size": size, "radius": radius, "density": density}


def density_value(name: str) -> float:
    """Map density bucket name to its per-1000-voxel value."""
    for n, v in _MIXED_DENSITIES:
        if n == name:
            return v
    raise ValueError(f"Unknown density bucket: {name!r}")


def build_otf_dataset(
    data_config_name: str,
    length: int,
    feature_map_size: int = 16,
    include_metadata: bool = True,
    augment: bool = False,
    seed_base: Optional[int] = None,
) -> Dataset:
    """Factory: returns a fully-configured OTF dataset for the given name."""
    parsed = parse_data_config_name(data_config_name)
    if parsed["mixed"]:
        return OnTheFlyMixedDataset(
            size=parsed["size"],
            length=length,
            feature_map_size=feature_map_size,
            include_metadata=include_metadata,
            augment=augment,
            seed_base=seed_base if seed_base is not None else 43_000_000,
        )
    return OnTheFlyPhase1Dataset(
        size=parsed["size"],
        radius=parsed["radius"],
        density_per_1000=density_value(parsed["density"]),
        length=length,
        feature_map_size=feature_map_size,
        include_metadata=include_metadata,
        augment=augment,
        seed_base=seed_base if seed_base is not None else 42_000_000,
    )


__all__ = [
    "OnTheFlyPhase1Dataset",
    "OnTheFlyMixedDataset",
    "parse_data_config_name",
    "density_value",
    "build_otf_dataset",
]
