"""Framework-independent MoSAR core."""

from .compatibility import gemma2_logit_softcap
from .config import MoSARConfig
from .hard_domain import HardDomainOutput, build_hard_domain
from .losses import CostLossOutput, normalized_reach_cost
from .outputs import RouterOutput, RoutingState, WorthFieldOutput
from .reference_attention import MoSARReferenceAttention, ReferenceAttentionOutput
from .regimes import RegimeKernel, pair_geometry
from .routers import PostPositionRouter, RoleSpecificPostPositionRouters
from .worth_field import WorthField

__all__ = [
    "CostLossOutput",
    "HardDomainOutput",
    "MoSARConfig",
    "MoSARReferenceAttention",
    "PostPositionRouter",
    "ReferenceAttentionOutput",
    "RegimeKernel",
    "RoleSpecificPostPositionRouters",
    "RouterOutput",
    "RoutingState",
    "WorthField",
    "WorthFieldOutput",
    "build_hard_domain",
    "gemma2_logit_softcap",
    "normalized_reach_cost",
    "pair_geometry",
]
