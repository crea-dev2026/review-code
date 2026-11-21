import torch
from torch import nn
from sklearn.cluster import KMeans


class CREAICA(nn.Module):


    def __init__(self, input_dim, latent_dim=32, hidden_dim=64,
                 n_anchors=10, beta=1e-3, gamma=1.2,
                 uncertainty_weight=1.0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.uncertainty_weight = uncertainty_weight
        self.n_anchors = n_anchors

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # 随模型一起保存，供后续窗口使用。
        self.register_buffer("anchors", torch.zeros(n_anchors, latent_dim))
        self.register_buffer("risk_low", torch.zeros(3))
        self.register_buffer("risk_high", torch.ones(3))
        self.register_buffer("reference_ready", torch.tensor(False))

    def encode(self, x):
        hidden = self.encoder(x)
        mean = self.mean_head(hidden)
        logvar = self.logvar_head(hidden).clamp(-16, 12)
        return mean, logvar

    def training_loss(self, x):
        mean, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)
        reconstructed = self.decoder(z)

        reconstruction_loss = (x - reconstructed).square().sum(dim=1)
        kl_loss = 0.5 * (
            mean.square() + logvar.exp() - 1 - logvar
        ).sum(dim=1)
        return (reconstruction_loss + self.beta * kl_loss).mean()

    def _signals(self, x):
        mean, logvar = self.encode(x)
        distance = torch.cdist(mean, self.anchors).amin(dim=1)
        uncertainty = logvar.exp().mean(dim=1)

        # 推理时用潜在均值解码，同一输入可以得到稳定的风险响应。
        reconstruction_error = (
            x - self.decoder(mean)
        ).square().sum(dim=1)
        return torch.stack(
            [distance, uncertainty, reconstruction_error], dim=1
        )

    @torch.no_grad()
    def fit_reference(self, normal_control):
        """用历史正常控制集初始化锚点和风险归一化参数。"""
        if len(normal_control) < self.n_anchors:
            raise ValueError("正常控制样本数必须至少等于锚点数。")

        mean, _ = self.encode(normal_control)
        clustering = KMeans(
            n_clusters=self.n_anchors,
            init="k-means++",
            n_init=10,
            random_state=42,
        )
        clustering.fit(mean.detach().cpu().numpy())
        self.anchors.copy_(mean.new_tensor(clustering.cluster_centers_))

        signals = self._signals(normal_control)
        self.risk_low.copy_(torch.quantile(signals, 0.05, dim=0))
        self.risk_high.copy_(torch.quantile(signals, 0.95, dim=0))
        self.reference_ready.fill_(True)

    def forward(self, x, risk_weights=(1/3, 1/3, 1/3)):
        """返回基础风险和内在置信度，每个输入样本各有一个值。"""
        if not self.reference_ready.item():
            raise RuntimeError("请先调用 fit_reference() 建立正常参考。")

        signals = self._signals(x)
        scale = (self.risk_high - self.risk_low).clamp_min(1e-8)
        normalized = ((signals - self.risk_low) / scale).clamp(0, 1)

        weights = x.new_tensor(risk_weights)
        if (weights.shape != (3,) or (weights < 0).any()
                or not torch.isclose(weights.sum(), x.new_tensor(1.0))):
            raise ValueError("三个风险权重必须非负且总和为 1。")

        base_risk = normalized @ weights
        distance, uncertainty = signals[:, 0], signals[:, 1]
        confidence = torch.exp(
            -distance.square() / (2 * self.gamma ** 2)
            - self.uncertainty_weight * uncertainty
        )
        return base_risk, confidence


if __name__ == "__main__":
    # 合成数据仅用于检查代码连接和输出形状。
    torch.manual_seed(42)
    train = torch.randn(96, 8)
    control = torch.randn(48, 8)
    current = torch.randn(12, 8)

    model = CREAICA(input_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for _ in range(3):
        optimizer.zero_grad()
        loss = model.training_loss(train)
        loss.backward()
        optimizer.step()

    model.eval()
    model.fit_reference(control)
    with torch.no_grad():
        risk, confidence = model(current)

    assert risk.shape == confidence.shape == (12,)
    assert torch.isfinite(risk).all() and torch.isfinite(confidence).all()
    print("CREA ICA check passed.")
    print("Risk scores:", risk.shape[0])
    print("Confidence scores:", confidence.shape[0])
    



