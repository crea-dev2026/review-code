import torch
from torch import nn
from torch.nn import functional as F


class CREAExplanation(nn.Module):
    """Window-level shift testing and soft-mask estimation."""

    def __init__(self, bins=100, permutations=1000, steps=100,
                 alpha=0.05, learning_rate=0.05, epsilon=1e-6):
        super().__init__()
        self.bins = bins
        self.permutations = permutations
        self.steps = steps
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def _bin_ids(self, scores):
        if scores.ndim != 1 or scores.numel() == 0:
            raise ValueError("Expected a nonempty vector of scores.")
        if not torch.isfinite(scores).all():
            raise ValueError("Scores must be finite.")
        if ((scores < 0) | (scores > 1)).any():
            raise ValueError("Expected calibrated scores in [0, 1].")

        edges = torch.arange(
            1, self.bins, device=scores.device, dtype=scores.dtype
        ) / self.bins
        return torch.bucketize(scores.detach().contiguous(), edges)

    def _histogram(self, ids, weights=None):
        if weights is None:
            weights = torch.ones_like(ids, dtype=torch.float32)

        counts = weights.new_zeros(self.bins)
        counts = counts.scatter_add(0, ids, weights) + self.epsilon
        return counts / counts.sum()

    def _symmetric_kl(self, first, second):
        p = self._histogram(first)
        q = self._histogram(second)
        return 0.5 * ((p - q) * (p.log() - q.log())).sum()

    @torch.no_grad()
    def _permutation_test(self, control_ids, current_ids, block_size, generator):
        if block_size < 1:
            raise ValueError("block_size must be positive.")
        if len(control_ids) % block_size or len(current_ids) % block_size:
            raise ValueError("Both groups must contain complete blocks.")

        observed = self._symmetric_kl(control_ids, current_ids)
        blocks = torch.cat((control_ids, current_ids)).reshape(-1, block_size)
        control_blocks = len(control_ids) // block_size
        extreme = 0

        for _ in range(self.permutations):
            order = torch.randperm(
                len(blocks), device=blocks.device, generator=generator
            )
            first = blocks[order[:control_blocks]].reshape(-1)
            second = blocks[order[control_blocks:]].reshape(-1)
            extreme += int(self._symmetric_kl(first, second) >= observed)

        return (extreme + 1) / (self.permutations + 1)

    @torch.enable_grad()
    def _fit_masks(self, control_ids, current_ids):
        ids = torch.cat((control_ids, current_ids))
        logits = nn.Parameter(torch.zeros(len(ids), device=ids.device))
        optimizer = torch.optim.Adam([logits], lr=self.learning_rate)
        target = self._histogram(current_ids)

        for _ in range(self.steps):
            weights = logits.sigmoid()
            retained, selected = weights.split(
                (len(control_ids), len(current_ids))
            )
            fitted = self._histogram(ids, weights)

            fit_loss = (target * (target.log() - fitted.log())).sum()
            label_cost = ((1 - retained).sum() + selected.sum()) / len(ids)
            entropy = (F.softplus(logits) - weights * logits).mean()

            optimizer.zero_grad()
            (fit_loss + label_cost + entropy).backward()
            optimizer.step()

        return logits.detach().sigmoid().split(
            (len(control_ids), len(current_ids))
        )

    def forward(self, control_scores, current_scores, *, block_size,
                generator=None):
        control_ids = self._bin_ids(control_scores)
        current_ids = self._bin_ids(current_scores)
        p_value = self._permutation_test(
            control_ids, current_ids, block_size, generator
        )

        if p_value >= self.alpha:
            return p_value, None, None

        control_mask, current_mask = self._fit_masks(control_ids, current_ids)
        return p_value, control_mask, current_mask