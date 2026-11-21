import torch
from torch import nn


class LoFiTCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2),
            nn.ReLU(),
        )
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, history, current):
        # history: [batch, time, features], excluding the current record.
        if history.size(1) < 7:
            raise ValueError("Lo-Fi requires at least 7 history steps.")

        hidden = self.network(history.transpose(1, 2))
        prediction = self.output(hidden[:, :, -1])
        return (prediction - current).square().mean(dim=1)


class HiFiTransformer(nn.Module):
    def __init__(self, input_dim, width=32, heads=4,
                 max_length=256, dropout=0.3):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, width)
        self.position = nn.Parameter(torch.empty(max_length, width))
        nn.init.normal_(self.position, std=0.02)

        self.encoder = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 2,
            dropout=dropout,
        )
        self.output = nn.Linear(width, input_dim)

    def forward(self, history, current):
        length = history.size(1)
        if not 1 <= length <= self.position.size(0):
            raise ValueError("History length exceeds the supported range.")

        hidden = self.input_projection(history) + self.position[:length]
        encoded = self.encoder(hidden.transpose(0, 1))
        prediction = self.output(encoded[-1])
        return (prediction - current).square().mean(dim=1)


class CREARouting(nn.Module):
    def __init__(self, input_dim, q=0.9, alpha_s=0.6,
                 alpha_delta=0.4, epsilon=1e-6):
        super().__init__()
        self.q = q
        self.alpha_s = alpha_s
        self.alpha_delta = alpha_delta
        self.epsilon = epsilon

        self.lo_fi = LoFiTCN(input_dim)
        self.hi_fi = HiFiTransformer(input_dim)

        self.register_buffer(
            "bin_edges", torch.tensor([0.2, 0.4, 0.6, 0.8])
        )
        self.register_buffer("control_histogram", torch.zeros(5))
        self.register_buffer("threshold", torch.tensor(0.0))
        self.register_buffer("score_bounds", torch.zeros(2, 2))
        self.register_buffer("reference_ready", torch.tensor(False))

    def _histogram(self, risk):
        if risk.numel() == 0:
            raise ValueError("Risk scores cannot be empty.")

        bin_ids = torch.bucketize(
            risk.reshape(-1).contiguous(), self.bin_edges
        )
        counts = torch.bincount(bin_ids, minlength=5).to(risk.dtype)
        counts = counts + self.epsilon
        return counts / counts.sum()

    @staticmethod
    def _scale(raw_score, bounds):
        low, high = bounds
        span = (high - low).clamp_min(1e-8)
        return ((raw_score - low) / span).clamp(0, 1)

    @torch.no_grad()
    def fit_reference(self, control_risk, control_history, control_current):
        if self.training:
            raise RuntimeError("Call eval() before reference calibration.")
        if control_risk.ndim != 1 or not (
            len(control_risk) == len(control_history) == len(control_current)
        ):
            raise ValueError("Control inputs must contain one entry per record.")

        self.control_histogram.copy_(self._histogram(control_risk))
        self.threshold.copy_(
            self.alpha_s * torch.quantile(control_risk, self.q)
        )

        lo_raw = self.lo_fi(control_history, control_current)
        hi_raw = self.hi_fi(control_history, control_current)
        quantiles = lo_raw.new_tensor([0.05, 0.95])

        self.score_bounds[0].copy_(torch.quantile(lo_raw, quantiles))
        self.score_bounds[1].copy_(torch.quantile(hi_raw, quantiles))
        self.reference_ready.fill_(True)

    @torch.no_grad()
    def forward(self, base_risk, history, current):
        if self.training or not self.reference_ready.item():
            raise RuntimeError("Call eval() and fit_reference() before routing.")
        if base_risk.ndim != 1 or not (
            len(base_risk) == len(history) == len(current)
        ):
            raise ValueError("Window inputs must contain one entry per record.")

        reference = self.control_histogram
        observed = self._histogram(base_risk)
        drift = 0.5 * (
            (reference - observed) * (reference.log() - observed.log())
        ).sum()

        routing_score = self.alpha_s * base_risk + self.alpha_delta * drift
        hi_fi_gate = routing_score > self.threshold

        lo_raw = self.lo_fi(history, current)
        scores = self._scale(lo_raw, self.score_bounds[0])

        if hi_fi_gate.any():
            hi_raw = self.hi_fi(history[hi_fi_gate], current[hi_fi_gate])
            scores[hi_fi_gate] = self._scale(hi_raw, self.score_bounds[1])

        return scores, hi_fi_gate, drift