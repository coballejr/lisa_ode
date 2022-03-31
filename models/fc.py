import torch
from torch.nn import Sequential, Module, Linear, Tanh

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.mlp(x)
        return u

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
