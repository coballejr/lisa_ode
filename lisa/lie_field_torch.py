import torch
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

class Scaling(LieField):

    def __init__(self):
        xix = lambda x,u: x
        etau = lambda x,u: -u
        super(Scaling, self).__init__(xix = xix, etau = etau)
