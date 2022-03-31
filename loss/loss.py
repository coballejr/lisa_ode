import torch
import torch.nn as nn
from typing import Tuple
from .grad_graph import GradGraph
from .convection_diffusion import ConvDiff

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
        lambda_bound: float = 1.0,
        boundary_type: str = "dirichlet",
        v: float = 1,
        k: float = 1
    ):

        self.model = model
        self.grad_graph = GradGraph(['x'], ['u'])

        self.pde = ConvDiff(v, k)
        self.grad_graph.add_pde(self.pde)

        if boundary_type == "periodic":
            self.boundary_loss = self.boundary_loss_periodic
        else:
            self.boundary_loss = self.boundary_loss_dirichlet

        self.lambda_pde = lambda_pde
        self.lambda_bound = lambda_bound

    def prep_inputs(self, inputs:Tensor, grad: bool = False):
        inputs = inputs.to(self.model.device)
        if grad:
            inputs.requires_grad=True
        return inputs

    def pde_loss(self, inputs: Tensor):
        x = self.prep_inputs(inputs, grad=True)
        u = self.model(x)
        grads = self.grad_graph.calc_grad(x=x, u=u)
        resid_u = self.pde(grads)
        loss = lossL2(resid_u)
        return self.lambda_pde*loss

    def boundary_loss_periodic(self, inputs1:Tensor, inputs2:Tensor):
        mse_fnc = nn.MSELoss()
        inputs1 = self.prep_inputs(inputs1)
        output1 = self.model(inputs1)
        inputs2 = self.prep_inputs(inputs2)
        output2 = self.model(inputs2)

        return self.lambda_bound*mse_fnc(output1, output2)

    def boundary_loss_dirichlet(self, inputs:Tensor, targets:Tensor):
        inputs  = self.prep_inputs(inputs)
        outputs = self.model(inputs)
        targets = targets.to(self.model.device)
        mse_fnc = nn.MSELoss()

        return self.lambda_bound*mse_fnc(outputs, targets)

    def __call__(self,
        field_points: Tensor,
        boundary_points: Tuple[Tensor],
    ):
        # Calculate loss components

        e_pde = self.pde_loss(field_points)
        e_boundary = self.boundary_loss(boundary_points[0], boundary_points[1])

        return e_pde + e_boundary

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



        u = self.model(input)
        u_error = self.error_fnc(u, target)

        return u_error
