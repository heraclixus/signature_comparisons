from abc import ABC, abstractmethod
from typing import Any, Callable

import torch.nn as nn
from torch import Tensor
import torch


def grad(f: Callable[[Tensor], Any], x: Tensor) -> tuple:
    """Compute gradient of f with respect to x."""
    create_graph = torch.is_grad_enabled()
    
    with torch.enable_grad():
        x = x.clone()
        
        if not x.requires_grad:
            x.requires_grad = True
        
        y = f(x)
        
        (gradient,) = torch.autograd.grad(y.sum(), x, create_graph=create_graph)
    
    return y, gradient

def jvp(f: Callable[[Tensor], Any], x: Tensor, v: Tensor) -> tuple:
    return torch.autograd.functional.jvp(
        f, x, v,
        create_graph=torch.is_grad_enabled()
    )

def t_dir(f: Callable[[Tensor], Any], t: Tensor) -> tuple:
    return jvp(f, t, torch.ones_like(t))

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
