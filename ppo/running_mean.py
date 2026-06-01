import torch


class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4, device="cpu"):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon  # avoid division by zero at start

    def update(self, x: torch.Tensor):
        # x: (batch, *shape) -> reduce over the batch axis
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        # Welford / parallel variance update (broadcasts over *shape)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / total)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.count = total

    @property
    def std(self):
        return torch.sqrt(self.var + 1e-8)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean