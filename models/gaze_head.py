"""Attention-based gaze heatmap head.

Reuses the driver-branch embedding as a query attending over the road
image's ViT patch grid. With no gaze_target supplied it's trained only
indirectly through the classification loss (read the map as "what the model
weighted," not a physical point of regard). When the manifest provides a
gaze_target (see data/gaze_estimation.py — a Gaze360-pretrained estimator's
output, projected onto the road image via an uncalibrated heuristic FOV,
not measured eye-tracking on this dataset), train.py adds a soft-target loss
via gaze_target_heatmap() below so the map is actually supervised.
"""

import torch
import torch.nn as nn


class GazeHead(nn.Module):
    def __init__(self, token_dim, num_heads=8):
        super().__init__()
        self.query_proj = nn.Linear(token_dim, token_dim)
        self.attn = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)

    def forward(self, driver_summary, road_patch_tokens):
        """
        Args:
            driver_summary: [B, token_dim] pooled driver-branch embedding
            road_patch_tokens: [B, N, token_dim] road ViT patch tokens, N = H*W (square)
        Returns:
            heatmap: [B, H, W] attention weights over the road patch grid (sums to 1)
        """
        query = self.query_proj(driver_summary).unsqueeze(1)  # [B, 1, D]
        _, attn_weights = self.attn(
            query, road_patch_tokens, road_patch_tokens, need_weights=True
        )
        attn_weights = attn_weights.squeeze(1)  # [B, N]
        side = int(attn_weights.shape[-1] ** 0.5)
        return attn_weights.view(-1, side, side)


def gaze_target_heatmap(gaze_xy, side, sigma=1.5):
    """(gx, gy) in [0, 1], batched [B, 2] -> Gaussian-bump target [B, side, side]
    normalized to sum to 1, matching GazeHead's output so they're directly
    comparable (e.g. via soft cross-entropy)."""
    device = gaze_xy.device
    ys = torch.arange(side, device=device, dtype=torch.float32)
    xs = torch.arange(side, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [side, side]

    cx = gaze_xy[:, 0].view(-1, 1, 1) * (side - 1)
    cy = gaze_xy[:, 1].view(-1, 1, 1) * (side - 1)

    dist_sq = (grid_x.unsqueeze(0) - cx) ** 2 + (grid_y.unsqueeze(0) - cy) ** 2
    heat = torch.exp(-dist_sq / (2 * sigma ** 2))
    return heat / heat.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
