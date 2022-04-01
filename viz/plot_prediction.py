import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from data.loaders import soln

@torch.no_grad()
def plot_prediction(
    model: torch.nn.Module,
    nsamps: int = 1000,
    u0: float = 1.0,
    plot_dir: Path = Path('.'),
    epoch: int = 0
):

    x = np.linspace(0, 0.95, nsamps).reshape(-1, 1)
    x_in = torch.Tensor(x)
    _, u = model(x_in, symms = None)
    u = u.cpu().numpy()
    ut = soln(x, u0)

    plt.plot(x, u,'r--', linewidth = 2, label = 'Prediciton')
    plt.plot(x, ut, c = 'black', linewidth = 4, label = 'Ground Truth')
    plt.xlim([0, 1.2])
    plt.ylim([0, 20])
    plt.legend()
    plt.title('Epoch ' + str(epoch))
    plt.savefig(plot_dir / f"upred_{epoch}.png")
    plt.close()
