# Third-party notices

This repository is released under the MIT License (see `LICENSE`). It also
contains, or derives from, the components listed below. Each is used under its
own license, reproduced or referenced here as those licenses require.

## VolRAFT

Directories: `volraft_asreleased/`, `volraft_fixed/`
Checkpoints: `checkpoints/volraft/`
Upstream: https://github.com/hereon-mbs/VolRAFT
License: MIT, Copyright (c) 2024 Helmholtz-Zentrum Hereon

The model code in both directories is derived from the upstream VolRAFT
release. `volraft_asreleased/` reproduces the upstream correlation-sampler
behaviour verbatim; `volraft_fixed/` is the same architecture with the
axis-order correction described in the accompanying paper. Both were retrained
on our synthetic particle volumes for the architecture comparison, so the
attached checkpoints are our weights over the upstream architecture, not
upstream's released weights. Upstream's own weights and results are not
redistributed here.

Reference: C. Y. Wong et al., "VolRAFT: Volumetric Optical Flow Network for
Digital Volume Correlation," CVPR Workshops, 2024.

MIT License text as published upstream:

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## RAFT

The architecture this work adapts to three dimensions.
Upstream: https://github.com/princeton-vl/RAFT
License: BSD 3-Clause, Copyright (c) 2020, Princeton Vision & Learning Lab

No RAFT source is vendored in this repository; `src/core/` is an independent
3D implementation. The reference is included because the design follows it.

Reference: Z. Teed and J. Deng, "RAFT: Recurrent All-Pairs Field Transforms for
Optical Flow," ECCV, 2020.

## ALDVC and FE-Global-DVC

Directory: `traditional_DIC_codes/benchmark/`
Upstream: https://github.com/YangMechanicsGroupUTAustin/ALDVC

Only the non-interactive wrappers and the patched helper routines needed to run
the baselines headlessly are included here; the full solver package is upstream.
This software is developed and maintained by the authors' group, which is
disclosed in the paper's competing-interests statement.

Reference: J. Yang et al., "Augmented Lagrangian Digital Volume Correlation
(ALDVC)," Experimental Mechanics, 2020, doi:10.1007/s11340-020-00607-3.
