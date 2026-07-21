"""
nanoGPT-scale decoder transformer with an optional ANCHORED GOAL CHANNEL
(primitive A). v1 and v2 are the SAME class; the only structural difference
is the `anchor` flag -> that is the ablation.

v1 (anchor=False): the goal enters only as a token at position 1. Its
    influence on the query position must survive L filler tokens of causal
    self-attention. As L grows this influence disperses.

v2 (anchor=True): a persistent goal register g_vec = GoalReg[goal_id] is
    added into the residual stream at the input of EVERY block, at every
    position. This contribution does not decay with context length because
    it is re-injected outside the softmax-attention competition. The forward
    pass reads it as a standing constraint but cannot rewrite it in-context.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    n_ans: int
    n_goals: int
    block_size: int = 512
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    anchor: bool = False          # <-- the one structural change


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
            .view(1, 1, cfg.block_size, cfg.block_size),
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        h = self.n_head
        q = q.view(B, T, h, C // h).transpose(1, 2)
        k = k.view(B, T, h, C // h).transpose(1, 2)
        v = v.view(B, T, h, C // h).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // h)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.n_ans)
        if cfg.anchor:
            # persistent per-goal register (primitive A)
            self.goal_reg = nn.Embedding(cfg.n_goals, cfg.n_embd)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def backbone(self, idx, goal=None):
        """Return final hidden states [B,T,C]. goal used only if cfg.anchor."""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        g_vec = None
        if self.cfg.anchor:
            assert goal is not None, "anchor model needs goal ids"
            g_vec = self.goal_reg(goal).unsqueeze(1)          # [B,1,C]
        for blk in self.blocks:
            if g_vec is not None:
                x = x + g_vec                                 # re-inject every layer
            x = blk(x)
        return self.ln_f(x)

    def forward(self, idx, qpos, goal=None):
        """Answer logits [B, n_ans] read at the query position."""
        h = self.backbone(idx, goal=goal)
        hq = h[torch.arange(h.size(0), device=h.device), qpos]   # [B,C]
        return self.head(hq), hq


class ExternalMonitor(nn.Module):
    """
    Wrapper baseline: a frozen v1 body + a bolt-on head that ALSO receives the
    true goal id from an external store (oracle). It only injects the goal at
    the OUTPUT. Tests whether an external wrapper matches the internal anchored
    channel. Expectation: competitive on end-only tasks, weaker on 'hard'.
    """
    def __init__(self, n_embd, n_goals, n_ans):
        super().__init__()
        self.goal_emb = nn.Embedding(n_goals, n_embd)
        self.head = nn.Linear(2 * n_embd, n_ans)

    def forward(self, hq_frozen, goal):
        z = torch.cat([hq_frozen, self.goal_emb(goal)], dim=-1)
        return self.head(z)
