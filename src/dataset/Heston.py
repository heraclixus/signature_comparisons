import torch


class HestonModel(torch.nn.Module):
    def __init__(self, mu, kappa, theta, sigma, rho):
        super(HestonModel, self).__init__()
        # Parameters as tensors
        self.mu = torch.tensor(mu, dtype=torch.float32)
        self.kappa = torch.tensor(kappa, dtype=torch.float32)
        self.theta = torch.tensor(theta, dtype=torch.float32)
        self.sigma = torch.tensor(sigma, dtype=torch.float32)
        self.rho = torch.tensor(rho, dtype=torch.float32)

        # Specify the noise type as 'general'
        self.noise_type = 'general'
        self.sde_type = 'ito'

    def f(self, t, y):
        # Drift part
        S, V = y[..., 0], y[..., 1]
        dS = self.mu * S  # Change this
        dV = self.kappa * (self.theta - V)  # Change this
        return torch.stack([dS, dV], dim=-1)

    def g(self, t, y):
        # Diffusion part corrected to account for noise dimensionality
        S, V = y[..., 0], y[..., 1]
        vol_S = torch.sqrt(V)  # Change this
        vol_v = self.sigma * torch.sqrt(V)  # Change this

        # Constructing a tensor of shape (batch_size, state_dim, noise_dim)
        dW1_dS = vol_S * S  # dW1 effect on S  # Change this
        dW1_dV = torch.zeros_like(S)  # dW1 has no direct effect on V  # Change this

        dW2_dS = torch.zeros_like(S)  # dW2 has no direct effect on S, Change this
        dW2_dV = self.rho * vol_S + torch.sqrt(1 - self.rho ** 2) * vol_v  # dW2 effect on V, Change this

        # Stacking to get the correct shape: (batch, state_channels, noise_channels)
        return torch.stack([torch.stack([dW1_dS, dW1_dV], dim=-1),
                            torch.stack([dW2_dS, dW2_dV], dim=-1)], dim=-1)
