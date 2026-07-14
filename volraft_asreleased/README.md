# VolRAFT as-released variant (SI size-fragility evidence ONLY)

Verbatim upstream VolRAFT model code (github.com/hereon-mbs/VolRAFT, MIT),
INCLUDING its correlation-sampler axis defect (H<->D swap; see
paper_1 Sect. sampler and tests/test_corr_sampler.py methodology).
Trained on our bead data purely to demonstrate, in an independent codebase,
that the defect (not the data domain) causes volume-size fragility:
compare against volraft_fixed (identical everything except the sampler fix).
Never used in the accuracy benchmark.
