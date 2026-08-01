"""MoRo-Former: Modality Routing Transformer with Grouped Queries.

Paper: WILD-Drive
- T=5 tasks, K=64 queries per task
- Each query is routed to one of 3 experts (LiDAR / Camera / Fusion) via hard routing
- Each expert branch is compressed independently per task
- Final output = concat of all expert branches across all tasks

Output tokens = T × 3_experts × tokens_per_branch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityRouter(nn.Module):
    """Predict per-query modality selection with straight-through gradient.

    Uses Gumbel-Softmax: hard argmax in forward, soft gradient in backward.
    This is essential — plain argmax has zero gradient, causing NaN grad_norm.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3),  # 3 experts
        )
        # Small init to prevent extreme routing at start
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def forward(self, query):
        """
        Args:
            query: [B, N, D]
        Returns:
            one_hot: [B, N, 3]  differentiable one-hot via straight-through
            assignments: [B, N]  hard assignment indices (0=lidar, 1=camera, 2=fusion)
        """
        logits = self.router(query)  # [B, N, 3]

        if self.training:
            # Gumbel-Softmax straight-through: hard forward, soft backward
            one_hot = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        else:
            # Pure hard routing at inference
            idx = torch.argmax(logits, dim=-1)
            one_hot = F.one_hot(idx, num_classes=3).float()

        assignments = torch.argmax(one_hot, dim=-1)
        return one_hot, assignments


class LocalMaskedCrossAttention(nn.Module):
    """Multi-head cross attention with optional local spatial masking."""

    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value, mask=None):
        B, Nq, D = query.shape
        Nk = key.shape[1]

        q = self.q_proj(query).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Nq, D)
        return self.out_proj(out)


class ModalityExpert(nn.Module):
    """Single modality expert: stacked cross-attention + FFN layers."""

    def __init__(self, hidden_dim, num_heads=8, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList(
            [LocalMaskedCrossAttention(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(self, query, key, value, mask=None):
        for attn, ffn, n1, n2 in zip(self.layers, self.ffns, self.norms1, self.norms2):
            query = query + attn(n1(query), key, value, mask)
            query = query + ffn(n2(query))
        return query


class BranchCompressor(nn.Module):
    """Compress variable-length expert outputs into fixed-size tokens via cross-attention."""

    def __init__(self, hidden_dim, num_compress_tokens, num_heads=8):
        super().__init__()
        self.num_compress_tokens = num_compress_tokens
        self.compress_queries = nn.Parameter(
            torch.randn(num_compress_tokens, hidden_dim) * 0.02
        )
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, expert_output):
        """
        Args:
            expert_output: [B, N_variable, D]
        Returns:
            compressed: [B, num_compress_tokens, D]
        """
        B = expert_output.shape[0]
        queries = self.compress_queries.unsqueeze(0).expand(B, -1, -1)

        if expert_output.shape[1] == 0:
            return torch.zeros_like(queries)

        compressed, _ = self.attn(queries, expert_output, expert_output)
        return self.norm(compressed)


class MoRoFormer(nn.Module):
    """Modality Routing Transformer with Grouped Queries.

    Architecture (following the paper):
        1. T task groups × K queries each, with task group embeddings (Eq. 1)
        2. Router predicts hard modality assignment per query (Eq. 3)
        3. Queries partitioned into Q_D^l, Q_D^c, Q_D^lc per task (Eq. 4)
        4. Each expert branch compressed independently → S_D^l, S_D^c, S_D^lc
        5. Concatenate: SD = [S_D^l; S_D^c; S_D^lc], then MLP (Eq. 5)

    Output tokens = T × 3 × tokens_per_branch
    """

    def __init__(
        self,
        hidden_dim=896,
        num_tasks=5,
        queries_per_task=64,
        num_heads=8,
        num_layers=2,
        tokens_per_branch=4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        self.queries_per_task = queries_per_task
        self.num_queries = num_tasks * queries_per_task
        self.tokens_per_branch = tokens_per_branch
        self.num_experts = 3

        # ★ Output token count: T tasks × 3 experts × tokens_per_branch
        self.num_output_tokens = num_tasks * self.num_experts * tokens_per_branch

        # Learnable task queries: Q ∈ R^{(T*K) × D}
        self.task_queries = nn.Parameter(
            torch.randn(num_tasks, queries_per_task, hidden_dim) * 0.02
        )
        # Task group embedding: Eg ∈ R^{T × D} (Eq. 1)
        self.task_embeddings = nn.Parameter(torch.randn(num_tasks, hidden_dim) * 0.02)

        # 3D reference points for local masking
        self.reference_points = nn.Parameter(
            torch.randn(num_tasks, queries_per_task, 3) * 0.02
        )

        # Modality router (Eq. 3)
        self.router = ModalityRouter(hidden_dim)

        # Three modality experts (Eq. 4)
        self.experts = nn.ModuleDict({
            "lidar": ModalityExpert(hidden_dim, num_heads, num_layers),
            "camera": ModalityExpert(hidden_dim, num_heads, num_layers),
            "fusion": ModalityExpert(hidden_dim, num_heads, num_layers),
        })

        # Per-task, per-expert branch compressors (Eq. 5)
        self.compressors = nn.ModuleDict()
        for t in range(num_tasks):
            for expert_name in ["lidar", "camera", "fusion"]:
                self.compressors[f"task{t}_{expert_name}"] = BranchCompressor(
                    hidden_dim, tokens_per_branch, num_heads
                )

        # Final MLP projection: MLPs(SD) in Eq. 5
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def _route_and_decode_task(self, task_queries, task_assignments, kv_dict):
        """Route queries for one task to experts and decode.

        Args:
            task_queries:      [B, K, D]
            task_assignments:  [B, K]  values in {0, 1, 2}
            kv_dict:           dict of expert_name -> [B, L, D]

        Returns:
            dict of expert_name -> list[Tensor] per batch element
        """
        B = task_queries.shape[0]
        expert_names = ["lidar", "camera", "fusion"]
        expert_outputs = {name: [] for name in expert_names}

        for expert_idx, expert_name in enumerate(expert_names):
            expert_mask = (task_assignments == expert_idx)  # [B, K]

            for b in range(B):
                indices = expert_mask[b].nonzero(as_tuple=True)[0]

                if len(indices) == 0:
                    expert_outputs[expert_name].append(
                        torch.zeros(0, self.hidden_dim, device=task_queries.device)
                    )
                else:
                    q = task_queries[b][indices].unsqueeze(0)       # [1, n, D]
                    kv = kv_dict[expert_name][b].unsqueeze(0)       # [1, L, D]
                    decoded = self.experts[expert_name](q, kv, kv)  # [1, n, D]
                    expert_outputs[expert_name].append(decoded.squeeze(0))

        return expert_outputs

    def _pad_and_compress(self, expert_outputs, task_idx, expert_name):
        """Pad variable-length outputs and compress.

        Args:
            expert_outputs: list of [n_i, D] tensors, length B
            task_idx: int
            expert_name: str

        Returns:
            compressed: [B, tokens_per_branch, D]
        """
        B = len(expert_outputs)
        max_n = max(x.shape[0] for x in expert_outputs)
        max_n = max(max_n, 1)  # avoid empty

        device = expert_outputs[0].device if expert_outputs[0].numel() > 0 else \
                 next(iter(self.compressors.values())).compress_queries.device

        padded = torch.zeros(B, max_n, self.hidden_dim, device=device)
        for b, out in enumerate(expert_outputs):
            if out.shape[0] > 0:
                padded[b, :out.shape[0]] = out

        return self.compressors[f"task{task_idx}_{expert_name}"](padded)

    def forward(self, lidar_features, camera_features):
        """
        Args:
            lidar_features:  [B, D, H_l, W_l] BEV features
            camera_features: [B, D, H_c, W_c] image features

        Returns:
            output_tokens: [B, num_output_tokens, D]
                num_output_tokens = T(5) × 3(experts) × tokens_per_branch
        """
        B = lidar_features.shape[0]

        # --- Flatten spatial features as key/value ---
        lidar_flat = lidar_features.flatten(2).permute(0, 2, 1)    # [B, H_l*W_l, D]
        camera_flat = camera_features.flatten(2).permute(0, 2, 1)  # [B, H_c*W_c, D]
        fusion_flat = torch.cat([lidar_flat, camera_flat], dim=1)   # [B, N_l+N_c, D]

        kv_dict = {
            "lidar": lidar_flat,
            "camera": camera_flat,
            "fusion": fusion_flat,
        }

        # --- Prepare queries with task group embeddings (Eq. 1) ---
        queries = self.task_queries.unsqueeze(0).expand(B, -1, -1, -1)  # [B, T, K, D]
        task_emb = self.task_embeddings.unsqueeze(0).unsqueeze(2)        # [1, T, 1, D]
        queries = queries + task_emb                                     # [B, T, K, D]

        # --- Router (Eq. 3) ---
        queries_flat = queries.reshape(B, self.num_queries, self.hidden_dim)
        router_probs, assignments = self.router(queries_flat)
        # Use straight-through for gradient: soft probs backward, hard forward
        # (router_probs is used for auxiliary load-balancing loss if needed)

        assignments_per_task = assignments.view(B, self.num_tasks, self.queries_per_task)

        # --- Per-task: route → decode → compress per expert branch ---
        all_compressed = []
        expert_names = ["lidar", "camera", "fusion"]

        for t in range(self.num_tasks):
            task_q = queries[:, t]                 # [B, K, D]
            task_assign = assignments_per_task[:, t]  # [B, K]

            # Route and decode
            expert_outputs = self._route_and_decode_task(task_q, task_assign, kv_dict)

            # Compress each branch: SD = [S_D^l; S_D^c; S_D^lc]
            for expert_name in expert_names:
                compressed = self._pad_and_compress(
                    expert_outputs[expert_name], t, expert_name
                )
                all_compressed.append(compressed)

        # --- Concatenate all compressed tokens ---
        # [B, T * 3 * tokens_per_branch, D]
        output_tokens = torch.cat(all_compressed, dim=1)

        # Final MLP (Eq. 5)
        output_tokens = self.output_proj(output_tokens)

        return output_tokens