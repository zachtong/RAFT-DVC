# Ported from VolRAFT (https://github.com/hereon-mbs/VolRAFT), MIT License,
# Copyright (c) 2024 Helmholtz-Zentrum Hereon.
# Modifications for RAFT-DVC by Zixiang (Zach) Tong <zachtong@utexas.edu>,
# University of Texas at Austin. Released under the MIT License; see LICENSE.
"""volraft_fixed.model -- VolRAFT architecture with the FIXED trilinear sampler.

``volraft.py``, ``volcorr.py``, ``extractor.py`` and ``update.py`` are
verbatim copies of the upstream VolRAFT ``models/VolRAFT`` package.
``utils/utils.py`` carries the two documented changes (fixed
``bilinear_sampler`` axis order; ``coords_grid`` kept unflipped by design).
"""

from .volraft import ModuleVolRAFT

__all__ = ['ModuleVolRAFT']
