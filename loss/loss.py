import torch
import torch.nn as nn
from typing import Tuple
from .grad_graph import GradGraph
from .seperable import SeperableODE
from lisa.lie_field_torch import LieField, Identity, Translation, Scaling
import math

Tensor = torch.Tensor

def lossL2(*args):
    out = 0
    for arg in args:
        if isinstance(arg, torch.Tensor):
            out = out + torch.mean(torch.pow(arg, 2))
    return out

class PINNTrainingLoss:

    def __init__(self,
        model: nn.Module,
        lambda_pde: float = 1.0,
        lambda_init: float = 1.0,
    ):

        self.model = model
        self.grad_graph = GradGraph(['x'], ['u'])

        self.pde = SeperableODE()
        self.grad_graph.add_pde(self.pde)

        self.lambda_pde = lambda_pde
        self.lambda_init = lambda_init

    def prep_inputs(self, inputs:Tensor, grad: bool = False):
        inputs = inputs.to(self.model.device)
        if grad:
            inputs.requires_grad=True
        return inputs

    def pde_loss(self, inputs: Tensor,
                       symm: LieField = None,
                       symm_method: str = 'approx',
                       eps: float = 1e-3):

        x = self.prep_inputs(inputs, grad=True)
        x,u = self.model(x, symm = symm, symm_method = symm_method, eps = eps)
        grads = self.grad_graph.calc_grad(x=x, u=u)
        resid_u = self.pde(grads)
        loss = lossL2(resid_u)
        return self.lambda_pde*loss

    def init_loss(self, inputs:Tensor, targets:Tensor):
        inputs  = self.prep_inputs(inputs)
        _, outputs = self.model(inputs)
        targets = targets.to(self.model.device)
        mse_fnc = nn.MSELoss()

        return self.lambda_init*mse_fnc(outputs, targets)

    def __call__(self,
        field_points: Tensor,
        init_points: Tuple[Tensor],
        symm: LieField,
        symm_method: str = 'approx',
        eps: float = 1e-3
    ):
        # Calculate loss components
        e_pde = self.pde_loss(inputs = field_points, symm = symm, symm_method =
                              symm_method, eps = eps)
        e_init = self.init_loss(init_points[0], init_points[1])

        return e_pde + e_init

class EvalLoss:

    def __init__(self,
        model: nn.Module,
        error_type: str = "mse"
    ):
        self.model = model
        if error_type == "mse":
            self.error_fnc = nn.MSELoss()
        else:
            self.error_fnc = nn.L1Loss()

    @torch.no_grad()
    def __call__(self,
        input: Tensor,
        target: Tensor
    ):
        input = input.to(self.model.device)
        target = target.to(self.model.device)
        _, u = self.model(input)
        u_error = self.error_fnc(u, target)

        return u_error

# equivariance loss

class EquLoss:

    def __init__(self,
        model: nn.Module,
        error_type: str = "mse",
        symm: LieField = Identity(),
        symm_method: str = 'full'
    ):
        self.model = model
        self.symm = symm
        self.symm_method = symm_method

        if error_type == "mse":
            self.error_fnc = nn.MSELoss()
        else:
            self.error_fnc = nn.L1Loss()

        if isinstance(symm, Identity):
            self.soln = self._soln_id
        elif isinstance(symm, Translation):
            self.soln = self._soln_trans
        elif isinstance(symm, Scaling):
            self.soln = self._soln_scale
        else:
            raise NotImplementedError('Unknown symm.')

    @torch.no_grad()
    def __call__(self,
        input: Tensor,
        u0: float,
        eps: float
    ):

        input = input.to(self.model.device)
        _, u = self.model(input, symm = self.symm, symm_method = self.symm_method, eps =
                         eps)
        target = self.soln(input, u0, eps)
        u_error = self.error_fnc(u, target)

        return u_error

    def _soln_id(self, x: Tensor,
                       u0: float,
                       eps:float) -> Tensor:

        c = -1/u0
        u = -(1 / (x+c))
        return u

    def _soln_trans(self, x: Tensor,
                          u0: float,
                          eps: float) -> Tensor:

        c = -1/u0
        x = x + eps
        u = -(1 / (x+c))
        return u


    def _soln_scale(self, x: Tensor,
                          u0: float,
                          eps: float) -> Tensor:
        alpha = math.exp(eps)
        c = -1/u0
        x = alpha*x
        u = -(alpha / (x+c))
        return u


