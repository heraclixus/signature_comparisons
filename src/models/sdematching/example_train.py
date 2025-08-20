from typing import Any, Sequence, Callable
from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor
from torch import distributions as D
from models.sdematching.matching_network import MatchingSDE
from models.sdematching.prior_network import PriorInitDistribution, PriorSDE, PriorObservation
from models.sdematching.posterior_network import PosteriorEncoder, PosteriorAffine

import matplotlib.pyplot as plt
from tqdm import trange

def solve_sde(
        sde: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
        z: Tensor,
        ts: float,
        tf: float,
        n_steps: int
) -> Tensor:
    tt = torch.linspace(ts, tf, n_steps + 1)[:-1]
    dt = (tf - ts) / n_steps
    dt_2 = abs(dt) ** 0.5

    path = [z]
    for t in tt:
        f, g = sde(z, t)
        w = torch.randn_like(z)
        z = z + f * dt + g * w * dt_2

        path.append(z)

    return torch.stack(path)



def grad(f: Callable[[Tensor], ...], x: Tensor) -> tuple[Tensor, Tensor]:
    create_graph = torch.is_grad_enabled()

    with torch.enable_grad():
        x = x.clone()

        if not x.requires_grad:
            x.requires_grad = True

        y = f(x)

        (gradient, ) = torch.autograd.grad(y.sum(), x, create_graph=create_graph)

    return y, gradient

def visualise_data(xs: Tensor):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for xs_i in xs:
        ax.plot(xs_i[:, 0], xs_i[:, 1], xs_i[:, 2])

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel('$z_1$', labelpad=0., fontsize=16)
    ax.set_ylabel('$z_2$', labelpad=.5, fontsize=16)
    ax.set_zlabel('$z_3$', labelpad=0., horizontalalignment='center', fontsize=16)

class SDE(nn.Module, ABC):
    @abstractmethod
    def drift(self, z: Tensor, t: Tensor, *args: Any) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def vol(self, z: Tensor, t: Tensor, *args: Any) -> Tensor:
        raise NotImplementedError

    def forward(self, z: Tensor, t: Tensor, *args: Any) -> tuple[Tensor, Tensor]:
        drift = self.drift(z, t, *args)
        vol = self.vol(z, t, *args)
        return drift, vol


class StochasticLorenzSDE(SDE):
    def __init__(self, a: Sequence = (10., 28., 8 / 3), b: Sequence = (.15, .15, .15)):
        super().__init__()
        self.a = a
        self.b = b

    def drift(self, x: Tensor, t: Tensor, *args) -> Tensor:
        x1, x2, x3 = torch.split(x, [1, 1, 1], dim=1)
        a1, a2, a3 = self.a

        f1 = a1 * (x2 - x1)
        f2 = a2 * x1 - x2 - x1 * x3
        f3 = x1 * x2 - a3 * x3

        return torch.cat([f1, f2, f3], dim=1)

    def vol(self, x: Tensor, t: Tensor, *args) -> Tensor:
        x1, x2, x3 = torch.split(x, [1, 1, 1], dim=1)
        b1, b2, b3 = self.b

        g1 = x1 * b1
        g2 = x2 * b2
        g3 = x3 * b3

        return torch.cat([g1, g2, g3], dim=1)
    

def gen_data(
        batch_size: int,
        ts: float,
        tf: float,
        n_steps: int,
        noise_std: float,
        n_inner_steps: int=100
) -> tuple[Tensor, Tensor]:
    sde = StochasticLorenzSDE()

    z0 = torch.randn(batch_size, 3)
    zs = solve_sde(sde, z0, ts, tf, n_steps=n_steps * n_inner_steps)
    zs = zs[::n_inner_steps]
    zs = zs.permute(1, 0, 2)

    mean = torch.mean(zs, dim=(0, 1))
    std = torch.std(zs, dim=(0, 1))

    eps = torch.randn_like(zs)
    xs = (zs - mean) / std + noise_std * eps

    ts = torch.linspace(ts, tf, n_steps + 1)
    ts = ts[None, :, None].repeat(batch_size, 1, 1)

    return xs, ts





batch_size = 2 ** 10
ts = 0.
tf = 1.
n_steps = 40
noise_std = .01

xs, ts = gen_data(batch_size, ts, tf, n_steps, noise_std)

visualise_data(xs[:6])


def train(sde_matching: MatchingSDE, xs: Tensor, ts: Tensor):
    iter = 4000

    optim = torch.optim.Adam(sde_matching.parameters(), lr=0.001)

    pbar = trange(iter)
    for _ in pbar:
        loss = sde_matching(xs, ts)

        loss = loss.mean()

        pbar.set_description(f"{loss.item():.4f}")

        optim.zero_grad()
        loss.backward()
        optim.step()

data_size = 3
latent_size = 4
hidden_size = 100

p_init_distr = PriorInitDistribution(latent_size)
p_sde = PriorSDE(latent_size, hidden_size)
p_observe = PriorObservation(latent_size, data_size, noise_std)

q_enc = PosteriorEncoder(data_size, hidden_size)
q_affine = PosteriorAffine(latent_size, hidden_size)

sde_matching = MatchingSDE(p_init_distr, p_sde, p_observe, q_enc, q_affine)

train(sde_matching, xs, ts)

bs = 6

z0 = p_init_distr().rsample([bs])[:, 0]

zs = solve_sde(p_sde, z0, 0., 3, n_steps=1000)
zs = zs.permute(1, 0, 2)

zs = zs.reshape(-1, latent_size)
zs, _ = p_observe.get_coeffs(zs)
zs = zs.reshape(bs, -1, data_size)