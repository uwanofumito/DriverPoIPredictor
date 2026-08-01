"""Driver-state classifier.

Tokenizer stage is lifted from Wild-Drive almost unmodified: a shared ViT
(DINOv2/DINOv3) extracts patch features from both camera streams, and
MoRo-Former fuses them into a fixed-size token set. The only structural
change is *what* fills the two MoRo-Former input slots — road-facing camera
takes the "camera" slot, driver-facing camera takes the slot Wild-Drive used
for LiDAR BEV features. MoRo-Former itself is modality-agnostic, so nothing
in models/moroformer.py needed to change.

Everything downstream of the tokenizer (LLM decoder, text-generation loss,
GRU trajectory planner) is gone, replaced by a small transformer classifier
head over [fusion tokens, control tokens, (optional) text tokens].
"""

import torch
import torch.nn as nn
from transformers import AutoModel

from models.control_encoder import ControlEncoder
from models.gaze_head import GazeHead
from models.moroformer import MoRoFormer
from models.text_encoder import TextEncoder


class DriverStateModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # ---------- ViT tokenizer (shared across road/driver cameras) ----------
        self.vision_model = AutoModel.from_pretrained(config.vision_model_path)
        if config.freeze_vision_model:
            for p in self.vision_model.parameters():
                p.requires_grad = False
        vision_hidden = self.vision_model.config.hidden_size

        def make_proj():
            return nn.Sequential(
                nn.Conv2d(vision_hidden, config.token_dim, kernel_size=1),
                nn.BatchNorm2d(config.token_dim),
                nn.ReLU(),
            )

        self.road_proj = make_proj()
        self.driver_proj = make_proj()

        # ---------- MoRo-Former fusion (driver -> "lidar" slot, road -> "camera" slot) ----------
        self.moroformer = MoRoFormer(
            hidden_dim=config.token_dim,
            num_tasks=config.moro_num_tasks,
            queries_per_task=config.moro_queries_per_task,
            num_heads=config.moro_num_heads,
            num_layers=config.moro_num_layers,
            tokens_per_branch=config.moro_tokens_per_branch,
        )

        # ---------- control (steering/throttle/brake/speed + accel) ----------
        self.control_encoder = ControlEncoder(
            config.control_input_dim, config.token_dim, config.num_control_tokens
        )

        # ---------- optional text ----------
        if config.use_text:
            self.text_encoder = TextEncoder(
                config.text_model_path,
                config.token_dim,
                config.num_text_tokens,
                freeze=config.freeze_text_model,
            )

        # ---------- gaze head (unsupervised attention visualization) ----------
        if config.use_gaze_head:
            self.gaze_head = GazeHead(config.token_dim)

        # ---------- classifier head ----------
        num_tokens = 1 + config.num_fusion_tokens + config.num_control_tokens  # +1 for CLS
        if config.use_text:
            num_tokens += config.num_text_tokens

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.token_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, config.token_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.token_dim,
            nhead=config.classifier_num_heads,
            dim_feedforward=config.token_dim * 4,
            dropout=config.classifier_dropout,
            batch_first=True,
        )
        self.classifier_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.classifier_num_layers
        )
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(config.token_dim),
            nn.Linear(config.token_dim, config.token_dim),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.token_dim, config.num_classes),
        )

    def _patches_to_grid(self, patch_embeds):
        """[B, T, C] ViT patch tokens (CLS already dropped) -> [B, C, H, W]."""
        b, t, c = patch_embeds.shape
        side = int(round(t**0.5))
        if side * side != t:
            raise ValueError(
                f"Expected a square ViT patch grid, got {t} patches. "
                f"Use an input resolution that's a multiple of the ViT patch size."
            )
        return patch_embeds.view(b, side, side, c).permute(0, 3, 1, 2)

    def forward(self, road_pixel_values, driver_pixel_values, control_series, texts=None):
        """
        Args:
            road_pixel_values:   [B, 3, H, W] road-facing camera frame(s)
            driver_pixel_values: [B, 3, H, W] driver-facing camera frame(s)
            control_series:      [B, T, control_input_dim] steering/throttle/brake/speed + accel
            texts:                list[str] of length B, optional free-text context
        Returns:
            dict with "logits" [B, num_classes] and "gaze_heatmap" [B, H, W] or None
        """
        b = road_pixel_values.shape[0]

        road_patches = self.vision_model(road_pixel_values).last_hidden_state[:, 1:, :]
        driver_patches = self.vision_model(driver_pixel_values).last_hidden_state[:, 1:, :]

        road_grid = self._patches_to_grid(road_patches)
        driver_grid = self._patches_to_grid(driver_patches)

        road_proj = self.road_proj(road_grid)      # [B, token_dim, H, W]
        driver_proj = self.driver_proj(driver_grid)

        fusion_tokens = self.moroformer(driver_proj, road_proj)  # [B, num_fusion_tokens, token_dim]
        control_tokens = self.control_encoder(control_series)     # [B, num_control_tokens, token_dim]

        token_streams = [fusion_tokens, control_tokens]
        if self.config.use_text:
            texts = texts if texts is not None else [""] * b
            token_streams.append(self.text_encoder(texts))

        cls = self.cls_token.expand(b, -1, -1)
        seq = torch.cat([cls] + token_streams, dim=1)
        seq = seq + self.pos_embed[:, : seq.shape[1], :]

        encoded = self.classifier_encoder(seq)
        logits = self.classifier_head(encoded[:, 0])

        gaze_heatmap = None
        if self.config.use_gaze_head:
            driver_summary = driver_proj.mean(dim=(2, 3))                    # [B, token_dim]
            road_patch_tokens = road_proj.flatten(2).permute(0, 2, 1)        # [B, H*W, token_dim]
            gaze_heatmap = self.gaze_head(driver_summary, road_patch_tokens)

        return {"logits": logits, "gaze_heatmap": gaze_heatmap}
