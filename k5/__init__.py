"""K5 differential-detail fusion research package.

K5 does not replace SHARP metric depth. It extracts context-consistent
TinyViM differential structure while keeping metric amplitude and integration
as later, separate research stages.
"""

from .types import ConsensusField, GradientField, NativeBox

__all__ = ["ConsensusField", "GradientField", "NativeBox"]
