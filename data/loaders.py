import torch
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from .pinns_loader import PINNS_Dataloader

Tensor = torch.Tensor

def soln(x: np.array,
         u0: float) -> np.array:

    c = -1/u0
    u = -(1 / (x+c))
    return u

def create_training_loader(
    n_field: int,
    n_init: int,
    u0: float,
    field_batch_size: int,
    init_batch_size: int,
    num_workers: int = 1
):
    # Create datasets
    field_dataset = Field1D(n_field)
    init_dataset = Initial1D(u0,n_init)

    data_loader = PINNS_Dataloader(
        field_dataset,
        init_dataset,
        field_batch_size,
        init_batch_size,
        num_workers = num_workers
    )
    return data_loader

def create_eval_loader(
    n_field: int,
    n_init: int,
    u0: float,
    batch_size: int,
    num_workers: int = 1
):
    # Create dataset (combined field and boundary points)
    dataset = Eval1D(n_field, n_init, u0)

    data_loader = DataLoader(
        dataset,
        batch_size = batch_size,
        drop_last = False,
        shuffle = False,
        num_workers = num_workers
    )
    return data_loader


class Field1D(Dataset):

    def __init__(self,
        n_examples: int,
    ):
        x = np.random.uniform(0,0.75, (n_examples, 1))
        self.inputs = torch.Tensor(x)

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, i: int) -> Tuple[Tensor]:
        return self.inputs[i]


class Initial1D(Dataset):

    def __init__(self,
        u0: float,
        n_examples: int
    ):
        x = np.zeros((n_examples,1))
        u0 = u0*np.ones((n_examples,1))
        self.input = torch.Tensor(x)
        self.target = torch.Tensor(u0)

    def __len__(self):
        return self.input.size(0)

    def __getitem__(self, i: int) -> Tuple[Tensor]:
        return self.input[i], self.target[i]

class Eval1D(Dataset):

    def __init__(self,
        n_field: int,
        n_init: int,
        u0: float
    ):
        super().__init__()

        # Sample domain points
        xf = np.random.uniform(0,0.75, (n_field, 1))
        uf= soln(xf, u0)
        # Sampling boundary points
        xb = np.zeros((n_init,1))
        ub = u0*np.ones((n_init,1))
        x = np.concatenate((xf, xb), axis=0)
        u = np.concatenate((uf, ub), axis=0)

        self.input = torch.Tensor(x)
        self.label = torch.Tensor(u)

    def __len__(self):
        return self.input.size(0)

    def __getitem__(self, i: int) -> Tuple[Tensor]:
        return self.input[i], self.label[i]
