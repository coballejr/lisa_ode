import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from data.loaders import soln

@torch.no_grad()
def plot_prediction(
    model: torch.nn.Module,
    nsamps: int = 1000,
    v: float = 1.0,
    k: float = 1.0,
    plot_dir: Path = Path('.'),
    epoch: int = 0
):

    x = np.linspace(0, 1, nsamps).reshape(-1, 1)
    x_in = torch.Tensor(x)
    u = model(x_in).cpu().numpy()
    ut = soln(x, v, k)

    plt.plot(x, u,'r--', linewidth = 2, label = 'Prediciton')
    plt.plot(x, ut, c = 'black', linewidth = 4, label = 'Ground Truth')
    plt.xlim([0, 1.2])
    plt.ylim([0, 1.2])
    plt.legend()
    plt.title('Epoch ' + str(epoch))
    plt.savefig(plot_dir / f"upred_{epoch}.png")
    plt.close()
