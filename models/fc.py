import torch
from torch.nn import Sequential, Module, Linear, Tanh
from lisa.lie_field_torch import LieField, Identity

class FullyConnected(Module):

    def __init__(self, hdim: int = 10):
        super(FullyConnected, self).__init__()
        # define mlp
        self.mlp = Sequential(Linear(1, 32, bias = True), Tanh(),
                              Linear(32, hdim, bias = True), Tanh(),
                              Linear(hdim, hdim, bias = True), Tanh(),
                              Linear(hdim, 1, bias = True)
                             )
        # Init weights
        self.apply(self._init_weights)
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    def forward(self, x: torch.Tensor,
                      symm: LieField = Identity(),
                      symm_method: str = 'approx',
                      eps: float = 1e-3) -> torch.Tensor:

        u = self.mlp(x)

        if symm_method == 'approx':
            xix, etau = symm.xix, symm.etau

            inf_x = xix(x,u)
            xstar = x + eps*inf_x

            T = self.mlp(xstar)
            inf_u = etau(xstar, T)

            u = T - eps*inf_u

        elif symm_method == 'full':
            X, U = symm.X, symm.U
            xstar = X(x,u, eps)
            T = self.mlp(xstar)
            u = U(xstar, T, -eps)

        return x, u


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
