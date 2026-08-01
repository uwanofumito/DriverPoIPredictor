"""Optional text branch: leaves room for free-text context (weather, road type,
notes) without requiring it. Frozen pretrained encoder + learned queries
compress variable-length text into a fixed token count; samples with no text
fall back to a learned "no-text" placeholder so the token count/shape never
changes based on whether text was supplied.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from models.moroformer import BranchCompressor


class TextEncoder(nn.Module):
    def __init__(self, text_model_path, token_dim, num_text_tokens, freeze=True, num_heads=8):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_path)
        self.text_model = AutoModel.from_pretrained(text_model_path)
        self.freeze = freeze
        if freeze:
            for p in self.text_model.parameters():
                p.requires_grad = False

        text_hidden = self.text_model.config.hidden_size
        self.project = nn.Sequential(
            nn.Linear(text_hidden, token_dim),
            nn.LayerNorm(token_dim),
            nn.ReLU(),
        )
        self.compressor = BranchCompressor(token_dim, num_text_tokens, num_heads)
        self.no_text_tokens = nn.Parameter(torch.randn(num_text_tokens, token_dim) * 0.02)

    def forward(self, texts):
        """
        Args:
            texts: list[str] of length B. "" (or whitespace-only) means "no text".
        Returns:
            tokens: [B, num_text_tokens, token_dim]
        """
        device = self.no_text_tokens.device
        b = len(texts)

        enc = self.tokenizer(
            list(texts), padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.set_grad_enabled(not self.freeze):
            hidden = self.text_model(**enc).last_hidden_state  # [B, L, D_text]
        projected = self.project(hidden)
        compressed = self.compressor(projected)  # [B, num_text_tokens, token_dim]

        has_text = torch.tensor(
            [len(t.strip()) > 0 for t in texts], device=device
        ).view(b, 1, 1)
        no_text = self.no_text_tokens.unsqueeze(0).expand(b, -1, -1)
        return torch.where(has_text, compressed, no_text)
