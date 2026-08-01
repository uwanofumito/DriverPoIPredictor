"""Encodes control params (steering/throttle/brake/speed) + acceleration into tokens."""

import torch.nn as nn

from models.moroformer import BranchCompressor


class ControlEncoder(nn.Module):
    """Projects a [B, T, control_input_dim] time series into a fixed set of tokens.

    T can vary per sample (e.g. a short window of CAN/IMU snapshots); a single
    snapshot works too with T=1. Uses the same compress-via-learned-queries
    trick as MoRo-Former's BranchCompressor so token count stays fixed for
    downstream fusion regardless of T.
    """

    def __init__(self, control_input_dim, token_dim, num_control_tokens, num_heads=8):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(control_input_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.ReLU(),
        )
        self.compressor = BranchCompressor(token_dim, num_control_tokens, num_heads)

    def forward(self, control_series):
        """
        Args:
            control_series: [B, T, control_input_dim]
        Returns:
            tokens: [B, num_control_tokens, token_dim]
        """
        projected = self.project(control_series)  # [B, T, token_dim]
        return self.compressor(projected)
