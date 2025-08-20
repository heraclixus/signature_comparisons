import torch.nn as nn
from torch import Tensor
import torch
from typing import Union
from models.sdematching.abstract_sde import t_dir

class PosteriorEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        out, h = self.gru(x)
        return torch.cat([h[0, :, None], out], dim=1)


class PosteriorAffine(nn.Module):
    def __init__(self, latent_size: int, hidden_size: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * latent_size),
        )
        self.sm = nn.Softmax(dim=-1)

    def get_coeffs(self, ctx: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        l = ctx.shape[1] - 1

        h, out = ctx[:, 0], ctx[:, 1:]
        ts = torch.linspace(0, 1, l)[None, :]
        c = self.sm(-(l * (ts - t)) ** 2)
        out = (out * c[:, :, None]).sum(dim=1)
        ctx_t = torch.cat([h + out, t], dim=1)

        m, log_s = self.net(ctx_t).chunk(chunks=2, dim=1)
        s = torch.exp(log_s)

        return m, s

    def forward(
            self,
            ctx: Tensor,
            t: Tensor,
            return_t_dir: bool = False
    ) -> Union[tuple[Tensor, Tensor], tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]]:
        if return_t_dir:
            def f(t_in: Tensor) -> Tensor:
                return self.get_coeffs(ctx, t_in)

            return t_dir(f, t)
        else:
            return self.get_coeffs(ctx, t)