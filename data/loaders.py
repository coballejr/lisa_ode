import torch
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from .pinns_loader import PINNS_Dataloader

Tensor = torch.Tensor

def soln(x: np.array,
         v: float,
         k: float) -> np.array:

    P = v/k
    u = (1 /(np.exp(P)-1))*(np.exp(P*x)-1)

    return u

def create_training_loader(
    n_field: int,
    n_boundary: int,
    field_batch_size: int,
    boundary_batch_size: int,
    num_workers: int = 1
):
    # Create datasets
    field_dataset = Field1D(n_field)
    boundary_dataset = Boundary1D(n_boundary)

    data_loader = PINNS_Dataloader(
        field_dataset,
        boundary_dataset,
        field_batch_size,
        boundary_batch_size,
        num_workers = num_workers
    )
    return data_loader

def create_eval_loader(
    n_field: int,
    n_boundary: int,
    batch_size: int,
    v: float,
    k: float,
    num_workers: int = 1
):
    # Create dataset (combined field and boundary points)
    dataset = Eval1D(n_field, n_boundary, v, k)

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
        x = np.random.rand(n_examples,1)
        self.inputs = torch.Tensor(x)

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, i: int) -> Tuple[Tensor]:
        return self.inputs[i]


class Boundary1D(Dataset):

    def __init__(self,
        n_examples: int
    ):
        x = np.random.choice([0, 1], n_examples).reshape(-1, 1)
        self.input = torch.Tensor(x)
        self.target = torch.Tensor(x)

    def __len__(self):
        return self.input.size(0)

    def __getitem__(self, i: int) -> Tuple[Tensor]:
        return self.input[i], self.target[i]

class Eval1D(Dataset):

    def __init__(self,
        n_field: int,
        n_boundary: int,
        v: float,
        k: float
    ):
        super().__init__()

        # Sample domain points
        xf= np.random.rand(n_field,1)
        uf= soln(xf, v, k)
        # Sampling boundary points
        xb = np.random.choice([0, 1], n_boundary).reshape(-1, 1)
        ub = xb
        x = np.concatenate((xf, xb), axis=0)
        u = np.concatenate((uf, ub), axis=0)

        self.input = torch.Tensor(x)
        self.label = torch.Tensor(u)

    def __len__(self):
        return self.input.size(0)

    def __getitem__(self, i: int) -> Tuple[Tensor]:
        return self.input[i], self.label[i]
