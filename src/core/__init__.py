"""
RAFT-DVC: Deep Learning for Digital Volume Correlation
Core module package
"""

from .raft_dvc import RAFTDVC, RAFTDVCConfig
from .extractor import (
    BasicEncoder,
    MediumEncoder,
    ShallowEncoder,
    FullResEncoder,
    ContextEncoder
)
from .update import BasicUpdateBlock
from .corr import CorrBlock

# Optional / development-only correlation implementations.  Imported
# lazily so the package loads on deployment environments where these
# files are absent (e.g. TACC Vista, where only the standard CorrBlock
# is used).  Access via ``from src.core.corr_otf import CorrBlockOnTheFly``
# directly when needed.
try:
    from .corr_otf import CorrBlockOnTheFly  # noqa: F401
    _HAS_OTF = True
except ImportError:
    CorrBlockOnTheFly = None  # type: ignore[assignment]
    _HAS_OTF = False

__all__ = [
    'RAFTDVC',
    'RAFTDVCConfig',
    'BasicEncoder',
    'MediumEncoder',
    'ShallowEncoder',
    'FullResEncoder',
    'ContextEncoder',
    'BasicUpdateBlock',
    'CorrBlock',
    'CorrBlockOnTheFly',
]
