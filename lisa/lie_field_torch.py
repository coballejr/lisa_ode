import torch
from math import exp
from typing import Callable

class LieField:

    def __init__(self, xix: Callable[[torch.Tensor, torch.Tensor],
                                     torch.Tensor],
                       etau: Callable[[torch.Tensor, torch.Tensor],
                                     torch.Tensor]
                ):

        self.xix = xix
        self.etau = etau

# specific symmetry fields
class Identity(LieField):

    def __init__(self):
        xix = lambda x,u: 0
        etau = lambda x,u: 0
        self.X = lambda x, u, eps: x
        self.U = lambda x, u, eps: u
        super(Identity, self).__init__(xix = xix, etau = etau)

class Scaling(LieField):

    def __init__(self):
        self.X = lambda x, u, eps: exp(eps)*x
        self.U = lambda x, u, eps: exp(-eps)*u
        xix = lambda x,u: x
        etau = lambda x,u: -u
        super(Scaling, self).__init__(xix = xix, etau = etau)

class Translation(LieField):

    def __init__(self):
        self.X = lambda x, u, eps: x + eps
        self.U = lambda x, u, eps: u
        xix = lambda x,u: 1
        etau = lambda x,u: 0
        super(Translation, self).__init__(xix = xix, etau = etau)
