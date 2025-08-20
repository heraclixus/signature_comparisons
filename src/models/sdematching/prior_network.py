import torch
import torch.nn as nn
import torch.distributions as D
from models.sdematching.abstract_sde import SDE
from torch import Tensor

class PriorInitDistribution(nn.Module):
    def __init__(self, latent_size: int):
        super().__init__()

        self.m = nn.Parameter(torch.zeros(1, latent_size))
        self.log_s = nn.Parameter(torch.zeros(1, latent_size))

    def forward(self) -> D.Distribution:
        m = self.m
        s = torch.exp(self.log_s)
        return D.Independent(D.Normal(m, s), 1)

class PriorSDE(SDE):
    def __init__(self, latent_size: int, hidden_size: int):
        super().__init__()

        self.drift_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )

        self.vol_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.Softplus(),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid()
                )
                for _ in range(latent_size)
            ]
        )

    def drift(self, z: Tensor, t: Tensor, *args) -> Tensor:
        return self.drift_net(z)

    def vol(self, z: Tensor, t: Tensor, *args) -> Tensor:
        z = torch.split(z, 1, dim=1)
        g = [net_i(z_i) for net_i, z_i in zip(self.vol_nets, z)]
        return torch.cat(g, dim=1)


class PriorObservation(nn.Module):
    def __init__(self, latent_size: int, data_size: int, noise_std: float):
        super().__init__()

        self.net = nn.Linear(latent_size, data_size)
        self.noise_std = noise_std

    def get_coeffs(self, z: Tensor) -> tuple[Tensor, Tensor]:
        m = self.net(z)
        s = torch.ones_like(m) * self.noise_std
        return m, s

    def forward(self, z: Tensor) -> D.Distribution:
        m, s = self.get_coeffs(z)
        return D.Independent(D.Normal(m, s), 1)