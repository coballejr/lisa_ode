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

class uTranslation(LieField):

    def __init__(self):
        xix = lambda x,u: torch.zeros_like(x)
        etau = lambda x,u: torch.ones_like(x)
        super(uTranslation, self).__init__(xix = xix, etau = etau)
