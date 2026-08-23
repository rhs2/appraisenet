"""Deep tabular models, implemented in-repo (no fragile third-party wrappers):

  - `EmbedMLP`: entity embeddings for every categorical + an MLP trunk. The classic
    strong deep baseline for tabular data (Guo & Berkhahn, 2016).
  - `FTTransformer`: a compact Feature-Tokenizer Transformer (Gorishniy et al., 2021):
    every feature (categorical embedding or linearly projected numeric) becomes a token,
    a [CLS] token attends over them, a head reads the price.

Both are wrapped in a scikit-style `fit(X) / predict(X)` interface operating on the
`FeatureSpace.embed_tensors` view, train with AdamW + early stopping on a validation
slice, and run on Apple MPS / CUDA when available.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _Base:
    def __init__(self, cardinalities: list[int], n_num: int, epochs: int = 40, batch: int = 1024,
                 lr: float = 1e-3, weight_decay: float = 1e-4, val_frac: float = 0.1,
                 patience: int = 6, seed: int = 42):
        self.cards, self.n_num = cardinalities, n_num
        self.epochs, self.batch, self.lr, self.wd = epochs, batch, lr, weight_decay
        self.val_frac, self.patience, self.seed = val_frac, patience, seed
        self.dev = _device()
        self.net: nn.Module | None = None

    def _build(self) -> nn.Module:  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, cats: np.ndarray, nums: np.ndarray, y: np.ndarray) -> "_Base":
        torch.manual_seed(self.seed)
        # standardise the target: log-price sits near 9-10, far from the network's
        # initialisation scale; training in z-space converges in a handful of epochs
        self.y_mean, self.y_std = float(np.mean(y)), float(np.std(y) or 1.0)
        y = (y - self.y_mean) / self.y_std
        rng = np.random.RandomState(self.seed)
        val = rng.rand(len(y)) < self.val_frac
        self.net = self._build().to(self.dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = nn.HuberLoss(delta=1.0)

        def tensors(mask):
            return (torch.as_tensor(cats[mask]).to(self.dev), torch.as_tensor(nums[mask]).to(self.dev),
                    torch.as_tensor(y[mask], dtype=torch.float32).to(self.dev))

        ct, nt, yt = tensors(~val)
        cv, nv, yv = tensors(val)
        best, best_state, bad = math.inf, None, 0
        n = len(yt)
        for _ in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, device=self.dev)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad()
                loss = loss_fn(self.net(ct[idx], nt[idx]).squeeze(-1), yt[idx])
                loss.backward()
                opt.step()
            sched.step()
            self.net.eval()
            with torch.no_grad():
                vloss = float(loss_fn(self.net(cv, nv).squeeze(-1), yv))
            if vloss < best - 1e-4:
                best, bad = vloss, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, cats: np.ndarray, nums: np.ndarray) -> np.ndarray:
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(cats), 4096):
                c = torch.as_tensor(cats[i:i + 4096]).to(self.dev)
                x = torch.as_tensor(nums[i:i + 4096]).to(self.dev)
                out.append(self.net(c, x).squeeze(-1).cpu().numpy())
        return np.concatenate(out) * self.y_std + self.y_mean


class EmbedMLP(_Base):
    def _build(self) -> nn.Module:
        cards, n_num = self.cards, self.n_num

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.embs = nn.ModuleList([nn.Embedding(c, min(32, max(4, round(1.6 * c ** 0.56)))) for c in cards])
                width = sum(e.embedding_dim for e in self.embs) + n_num
                self.trunk = nn.Sequential(
                    nn.BatchNorm1d(width), nn.Linear(width, 256), nn.GELU(), nn.Dropout(0.15),
                    nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.10), nn.Linear(128, 1))

            def forward(self, cats, nums):
                e = torch.cat([emb(cats[:, i]) for i, emb in enumerate(self.embs)], dim=1)
                return self.trunk(torch.cat([e, nums], dim=1))

        return Net()


class FTTransformer(_Base):
    def __init__(self, *a, d_token: int = 48, n_layers: int = 3, n_heads: int = 6, **kw):
        super().__init__(*a, **kw)
        self.d_token, self.n_layers, self.n_heads = d_token, n_layers, n_heads

    def _build(self) -> nn.Module:
        cards, n_num, d = self.cards, self.n_num, self.d_token
        n_layers, n_heads = self.n_layers, self.n_heads

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.cat_tok = nn.ModuleList([nn.Embedding(c, d) for c in cards])
                self.num_w = nn.Parameter(torch.randn(n_num, d) * 0.02)
                self.num_b = nn.Parameter(torch.zeros(n_num, d))
                self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
                layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=4 * d,
                                                   dropout=0.1, activation="gelu", batch_first=True,
                                                   norm_first=True)
                self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
                self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

            def forward(self, cats, nums):
                toks = [emb(cats[:, i]).unsqueeze(1) for i, emb in enumerate(self.cat_tok)]
                toks.append(nums.unsqueeze(-1) * self.num_w + self.num_b)
                x = torch.cat([self.cls.expand(cats.size(0), -1, -1)] + toks, dim=1)
                return self.head(self.encoder(x)[:, 0])

        return Net()
