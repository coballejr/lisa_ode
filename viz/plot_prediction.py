import matplotlib.pyplot as plt
import numpy as np
import torch
from lisa.lie_field_torch import LieField, Identity
from typing import Tuple
from pathlib import Path
from data.loaders import soln

@torch.no_grad()
def plot_prediction(
    model: torch.nn.Module,
    nsamps: int = 1000,
    u0: float = 1.0,
    symms: Tuple[LieField] = (Identity(),),
    symm_method: str = 'approx',
    eps_list: np.array = np.array([0]),
    plot_dir: Path = Path('.'),
    epoch: int = 0
):

    x = np.linspace(0, 0.75, nsamps).reshape(-1, 1)
    x_in = torch.Tensor(x)
    ut = soln(x, u0)

    for i, symm in enumerate(symms):
        for eps in eps_list:
            symm_label = 'Prediciton' + str(i) + 'eps' + str(np.round_(eps, 3))
            _, u = model(x_in, symm, symm_method, eps)
            u = u.cpu().numpy()
            plt.plot(x, u, linewidth = 2, label = symm_label)

    plt.plot(x, ut, c = 'black', linewidth = 4, label = 'Ground Truth')
    plt.xlim([0, 0.8])
    plt.ylim([0, 5])
    plt.legend()
    plt.title('Epoch ' + str(epoch))
    plt.savefig(plot_dir / f"upred_{epoch}.png")
    plt.close()
