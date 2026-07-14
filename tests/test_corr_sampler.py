"""Regression tests for the correlation-sampler axis convention.

Background (2026-07-12): bilinear_sampler_3d historically swapped the W and D
axes of the sampled volume (querying (h, w, d) returned vol[h, d, w]).  All
all-pairs correlation lookups therefore returned cost surfaces whose peak
drifted with the source voxel's position offset A = w - d, which made trained
models catastrophically size-dependent (native-size EPE 0.09 vs 3.8+ at 2x
size).  These tests pin the FIXED convention and document the legacy one.

Non-cubic volumes are essential here: with cubes an axis swap is invisible in
shapes and only corrupts positions.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.core.corr import CorrBlock, bilinear_sampler_3d, coords_grid_3d
from src.core.raft_dvc import RAFTDVC, RAFTDVCConfig
from src.core.utils import warp_volume_3d


def _query(vol: torch.Tensor, h: float, w: float, d: float, **kw) -> float:
    coords = torch.tensor([[[[[h, w, d]]]]], dtype=torch.float32)
    return bilinear_sampler_3d(vol, coords, **kw)[0, 0, 0, 0, 0].item()


class TestBilinearSampler3D:
    """Impulse tests on a NON-cubic volume."""

    H, W, D = 8, 12, 16
    SPIKE = (3, 7, 11)

    @pytest.fixture()
    def vol(self) -> torch.Tensor:
        v = torch.zeros(1, 1, self.H, self.W, self.D)
        v[(0, 0) + self.SPIKE] = 1.0
        return v

    def test_fixed_returns_value_at_queried_coord(self, vol):
        assert _query(vol, *self.SPIKE) == pytest.approx(1.0)

    def test_fixed_zero_away_from_spike(self, vol):
        h, w, d = self.SPIKE
        assert _query(vol, h, d, w) == pytest.approx(0.0)  # transposed query
        assert _query(vol, h, w + 2, d) == pytest.approx(0.0)

    def test_fixed_trilinear_weights(self, vol):
        h, w, d = self.SPIKE
        assert _query(vol, h, w + 0.25, d) == pytest.approx(0.75, abs=1e-6)
        assert _query(vol, h, w, d - 0.5) == pytest.approx(0.5, abs=1e-6)

    def test_legacy_swaps_w_and_d_on_cube(self):
        """Document the legacy behaviour (needs a cube so both axes exist)."""
        s = 16
        v = torch.zeros(1, 1, s, s, s)
        v[0, 0, 3, 7, 11] = 1.0
        assert _query(v, 3, 7, 11, legacy_wd_swap=True) == pytest.approx(0.0)
        assert _query(v, 3, 11, 7, legacy_wd_swap=True) == pytest.approx(1.0)

    def test_fixed_matches_direct_indexing_on_random_volume(self):
        torch.manual_seed(0)
        v = torch.randn(1, 1, self.H, self.W, self.D)
        for (h, w, d) in [(0, 0, 0), (7, 11, 15), (2, 9, 4), (5, 3, 13)]:
            assert _query(v, h, w, d) == pytest.approx(
                v[0, 0, h, w, d].item(), abs=1e-6)


class TestCorrBlockPeakLocation:
    """The lookup's cost-surface peak must sit at the true displacement,
    independent of the source voxel's position (the legacy bug made it drift
    by A = w - d)."""

    def _peak_offsets(self, legacy: bool) -> list:
        torch.manual_seed(1)
        G = 16
        C = 32
        shift = (1, 2, -1)  # integer feature-space displacement (h, w, d)
        fmap2 = torch.randn(1, C, G, G, G)
        fmap1 = torch.roll(fmap2, shifts=tuple(-s for s in shift), dims=(2, 3, 4))
        # fmap1[x] = fmap2[x + shift] -> best match of source x is x + shift
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=1, radius=4,
                            legacy_wd_swap=legacy)
        coords = coords_grid_3d(1, G, G, G, device=fmap1.device)
        out = corr_fn(coords)  # (1, 9^3, G, G, G)
        n = 9
        offsets = []
        # probe voxels far from borders, with varying A = w - d
        for (h, w, d) in [(8, 8, 8), (8, 10, 5), (8, 12, 6), (7, 5, 11)]:
            win = out[0, :, h, w, d].reshape(n, n, n)
            dy, dx, dz = np.unravel_index(int(win.argmax()), (n, n, n))
            offsets.append((dy - 4, dx - 4, dz - 4))
        return offsets

    def test_fixed_peak_at_true_shift_everywhere(self):
        assert self._peak_offsets(legacy=False) == [(1, 2, -1)] * 4

    def test_legacy_peak_drifts_with_position(self):
        offs = self._peak_offsets(legacy=True)
        assert offs != [(1, 2, -1)] * 4


class TestWarpVolume3D:
    def test_integer_translation_moves_spike_correctly(self):
        H, W, D = 8, 12, 16
        v = torch.zeros(1, 1, H, W, D)
        v[0, 0, 4, 6, 9] = 1.0
        # warp samples source at x + flow(x): spike lands where
        # (h + fh, w + fw, d + fd) == (4, 6, 9) -> at (3, 4, 10)
        flow = torch.zeros(1, 3, H, W, D)
        flow[0, 0], flow[0, 1], flow[0, 2] = 1.0, 2.0, -1.0
        warped = warp_volume_3d(v, flow)
        assert warped[0, 0, 3, 4, 10].item() == pytest.approx(1.0, abs=1e-5)
        assert warped[0, 0, 4, 6, 9].item() == pytest.approx(0.0, abs=1e-5)


class TestConfigVersioning:
    def test_new_config_defaults_to_fixed(self):
        assert RAFTDVCConfig().corr_sampler_version == 2

    def test_from_dict_without_field_defaults_to_fixed(self):
        cfg = RAFTDVCConfig.from_dict({'encoder_type': '1/2', 'corr_levels': 2})
        assert cfg.corr_sampler_version == 2

    def test_to_dict_round_trips_version(self):
        d = RAFTDVCConfig(corr_sampler_version=1).to_dict()
        assert d['corr_sampler_version'] == 1
        assert RAFTDVCConfig.from_dict(d).corr_sampler_version == 1

    def test_invalid_version_rejected(self):
        with pytest.raises(ValueError, match="corr_sampler_version"):
            RAFTDVCConfig(corr_sampler_version=3)

    def test_fixed_version_blocks_unported_corr_impls(self):
        with pytest.raises(ValueError, match="not been ported"):
            RAFTDVCConfig(corr_impl="on_the_fly", corr_sampler_version=2)

    def test_legacy_version_allows_otf(self):
        cfg = RAFTDVCConfig(corr_impl="on_the_fly", corr_sampler_version=1)
        assert cfg.corr_impl == "on_the_fly"

    def test_checkpoint_without_field_loads_as_legacy(self, tmp_path):
        cfg = RAFTDVCConfig(encoder_type='1/8', corr_levels=2, corr_radius=4)
        model = RAFTDVC(cfg)
        ckpt_path = tmp_path / "old_style.pth"
        cfg_dict = cfg.to_dict()
        del cfg_dict['corr_sampler_version']  # simulate pre-fix checkpoint
        torch.save({'model_config': cfg_dict,
                    'model_state_dict': model.state_dict()}, ckpt_path)
        loaded, _ = RAFTDVC.load_checkpoint(str(ckpt_path))
        assert loaded.config.corr_sampler_version == 1

    def test_checkpoint_with_field_keeps_version(self, tmp_path):
        cfg = RAFTDVCConfig(encoder_type='1/8', corr_levels=2, corr_radius=4)
        model = RAFTDVC(cfg)
        ckpt_path = tmp_path / "new_style.pth"
        torch.save({'model_config': cfg.to_dict(),
                    'model_state_dict': model.state_dict()}, ckpt_path)
        loaded, _ = RAFTDVC.load_checkpoint(str(ckpt_path))
        assert loaded.config.corr_sampler_version == 2
