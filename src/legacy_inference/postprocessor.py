"""
Postprocessor: map flow from processed resolution back to original resolution.

This module reverses what Preprocessor did:
    1. Upsample flow field to original shape.
    2. Scale flow displacement values by downsample_factor.

Design:
    - Stateless: uses PreprocessMeta from the Preprocessor.
    - No hidden coupling between Preprocessor and Postprocessor;
      all shared information travels through PreprocessMeta.
"""

import torch

from .preprocessor import PreprocessMeta


class Postprocessor:
    """
    Restore flow field to original volume resolution.

    Usage:
        post = Postprocessor()
        flow_original = post.restore(flow, meta)
    """

    def restore(
        self,
        flow: torch.Tensor,
        meta: PreprocessMeta,
    ) -> torch.Tensor:
        """
        Upsample and rescale flow to original volume resolution.

        Args:
            flow: (1, 3, D', H', W') at processed resolution.
            meta: PreprocessMeta from Preprocessor.

        Returns:
            flow_original: (1, 3, D_orig, H_orig, W_orig)
        """
        raise NotImplementedError
