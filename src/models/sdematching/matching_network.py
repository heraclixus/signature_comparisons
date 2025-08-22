import torch
import torch.nn as nn
from torch import Tensor
import torch.distributions as D
from models.sdematching.abstract_sde import grad
from models.sdematching.abstract_sde import SDE

class MatchingSDE(nn.Module):
    def __init__(
            self,
            p_init_distr: nn.Module,
            p_sde: SDE,
            p_observe: nn.Module,
            q_enc: nn.Module,
            q_affine: nn.Module
    ):
        super().__init__()

        self.p_init_distr = p_init_distr
        self.p_sde = p_sde
        self.p_observe = p_observe
        self.q_enc = q_enc
        self.q_affine = q_affine

    def loss_prior(self, ctx: Tensor) -> Tensor:
        bs = ctx.shape[0]

        t0 = torch.zeros(bs, 1, device=ctx.device)

        m0, s0 = self.q_affine(ctx, t0)
        q_z0 = D.Independent(D.Normal(m0, s0), 1)

        p_z0 = self.p_init_distr()

        loss_prior = D.kl_divergence(q_z0, p_z0)

        return loss_prior

    def loss_diff(self, ctx: Tensor, t: Tensor) -> Tensor:
        (m, s), (dm, ds) = self.q_affine(ctx, t, return_t_dir=True)

        eps = torch.randn_like(m)
        z = m + s * eps

        def g2_in(z_in):
            return self.p_sde.vol(z_in, t) ** 2

        g2, d_g2 = grad(g2_in, z)

        q_dz = dm + ds * eps
        q_score = - eps / s
        q_drift = q_dz + 0.5 * g2 * q_score + 0.5 * d_g2

        p_drift = self.p_sde.drift(z, t)

        loss_diff = 0.5 * (q_drift - p_drift) ** 2 / g2
        loss_diff = loss_diff.sum(dim=1)

        return loss_diff

    def loss_recon(self, ctx: Tensor, x: Tensor, t: Tensor) -> Tensor:
        m, s = self.q_affine(ctx, t)

        eps = torch.randn_like(m)
        z = m + s * eps

        p_x = self.p_observe(z)

        loss_recon = -p_x.log_prob(x)

        return loss_recon

    def forward(self, xs: Tensor, ts: Tensor) -> Tensor:
        bs = xs.shape[0]
        n = xs.shape[1]

        ctx = self.q_enc(xs)

        # prior loss
        loss_prior = self.loss_prior(ctx)

        # diffusion loss
        t = torch.rand(bs, 1, device=ts.device) * (ts[:, -1] - ts[:, 0]) + ts[:, 0]

        loss_diff = self.loss_diff(ctx, t)

        # reconstruction loss
        rng = torch.arange(bs, device=ts.device)
        u = torch.randint(n, [bs], device=ts.device)
        t_u = ts[rng, u]
        x_u = xs[rng, u]

        loss_recon = self.loss_recon(ctx, x_u, t_u)

        # full loss
        loss = loss_prior + loss_diff + loss_recon

        return loss