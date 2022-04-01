import torch
from torch.nn import Sequential, Module, Linear, Tanh
from typing import Tuple
from lisa.lie_field_torch import LieField

class FullyConnected(Module):

    def __init__(self, hdim: int = 10):
        super(FullyConnected, self).__init__()
        # define mlp
        self.mlp = Sequential(Linear(1, 32, bias = False), Tanh(),
                              Linear(32, hdim, bias = False), Tanh(),
                              Linear(hdim, hdim, bias = False), Tanh(),
                              Linear(hdim, 1, bias = False)
                             )
        # Init weights
        self.apply(self._init_weights)
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    def forward(self, x: torch.Tensor,
                      symms: Tuple[LieField] = None,
                      eps: float = 1e-3) -> torch.Tensor:

        u = self.mlp(x)

        if symms:
            x_tup = (x,)
            u_tup = (u,)
            for lf in symms:
                xix, etau = lf.xix, lf.etau

                inf_x = xix(x, u)
                xstar = x + eps*inf_x
                Tstar = self.mlp(xstar)

                inf_u = etau(xstar, Tstar)
                ustar = Tstar - eps*inf_u
                x_tup += (xstar,)
                u_tup += (ustar,)
            x = torch.cat(x_tup, dim = 0)
            u = torch.cat(u_tup, dim = 0)

        return x,u

    def _init_weights(self, m: Module):
        if isinstance(m, Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def num_parameters(self):
        count = 0
        for _, param in self.named_parameters():
            count += param.numel()
        return count

    @property
    def device(self):
        return self.dummy_param.device
