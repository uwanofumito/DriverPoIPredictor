"""Standalone pedestrian-state classifier (look-at-traffic / crossing /
walking-vs-standing), trained on JAAD's real per-frame behavior annotations.

Deliberately NOT wired into DriverStateModel — this is step one of the plan
in README's "Integrating pedestrian-state estimation later": get this
working and validated on its own first (its frozen ViT choice matches
DriverStateModel's on purpose, so its pooled features slot in later as an
additional token stream without retraining a different backbone).
"""

import torch.nn as nn
from transformers import AutoModel

LABEL_KEYS = ["look", "cross", "action"]


class PedestrianStateModel(nn.Module):
    def __init__(self, vision_model_path="facebook/dinov2-base", freeze_vision_model=True, dropout=0.1):
        super().__init__()
        self.vision_model = AutoModel.from_pretrained(vision_model_path)
        if freeze_vision_model:
            for p in self.vision_model.parameters():
                p.requires_grad = False

        hidden = self.vision_model.config.hidden_size
        self.heads = nn.ModuleDict({
            key: nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 2),
            )
            for key in LABEL_KEYS
        })

    def forward(self, pixel_values):
        patch_tokens = self.vision_model(pixel_values).last_hidden_state[:, 1:, :]
        pooled = patch_tokens.mean(dim=1)  # [B, hidden]
        return {key: head(pooled) for key, head in self.heads.items()}
