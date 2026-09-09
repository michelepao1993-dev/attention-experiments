from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import MoSARConfig
from .outputs import RouterOutput, RoutingState


class PostPositionRouter(nn.Module):
    """Route one attention role from post-positional head representations.

    The input must have shape `[batch, sequence, heads, head_dim]`. The module
    concatenates the head vectors, projects them to a compact role-specific
    representation, and predicts a distribution over regimes.

    No pairwise query-key interaction is computed here.
    """

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        config: MoSARConfig,
    ) -> None:
        super().__init__()
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError("num_heads and head_dim must be positive.")

        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.temperature = float(config.temperature)
        self.num_regimes = int(config.num_regimes)

        input_size = self.num_heads * self.head_dim
        self.input_projection = nn.Linear(input_size, config.router_hidden_size)
        self.output_projection = nn.Linear(config.router_hidden_size, config.num_regimes)
        self.reset_parameters(config)

    def reset_parameters(self, config: MoSARConfig) -> None:
        nn.init.normal_(
            self.input_projection.weight,
            mean=0.0,
            std=config.input_weight_std,
        )
        nn.init.zeros_(self.input_projection.bias)
        nn.init.normal_(
            self.output_projection.weight,
            mean=0.0,
            std=config.output_weight_std,
        )
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, post_position_heads: Tensor) -> RouterOutput:
        if post_position_heads.ndim != 4:
            raise ValueError(
                "router input must have shape [batch, sequence, heads, head_dim]; "
                f"received {tuple(post_position_heads.shape)}"
            )
        if post_position_heads.shape[-2:] != (self.num_heads, self.head_dim):
            raise ValueError(
                "router head shape mismatch: expected "
                f"({self.num_heads}, {self.head_dim}), received "
                f"{tuple(post_position_heads.shape[-2:])}"
            )

        flattened = post_position_heads.flatten(start_dim=-2)
        features = F.gelu(self.input_projection(flattened))
        logits = self.output_projection(features)
        probabilities = F.softmax(logits / self.temperature, dim=-1)
        return RouterOutput(features=features, logits=logits, probabilities=probabilities)


class RoleSpecificPostPositionRouters(nn.Module):
    """Separate query and key routers over role-specific representations."""

    def __init__(
        self,
        *,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        config: MoSARConfig,
    ) -> None:
        super().__init__()
        self.query_router = PostPositionRouter(
            num_heads=num_query_heads,
            head_dim=head_dim,
            config=config,
        )
        self.key_router = PostPositionRouter(
            num_heads=num_kv_heads,
            head_dim=head_dim,
            config=config,
        )

    def forward(self, query_post_position: Tensor, key_post_position: Tensor) -> RoutingState:
        return RoutingState(
            query=self.query_router(query_post_position),
            key=self.key_router(key_post_position),
        )
