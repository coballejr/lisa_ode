'''https://github.com/NickGeneva/taylor_green_pinns/blob/main/data/pinns_loader.py
'''

from torch.utils.data import Dataset, DataLoader

class PINNS_Dataloader(object):

    def __init__(self,
        field_dataset: Dataset,
        initial_dataset: Dataset,
        boundary_dataset: Dataset,
        field_batch_size: int,
        initial_batch_size: int,
        boundary_batch_size: int,
        num_workers: int = 1,
        shuffle: bool = True,
        drop_last: bool = True
    ):
        self.field_loader = DataLoader(
            field_dataset,
            batch_size = field_batch_size,
            drop_last = drop_last,
            shuffle = shuffle,
            num_workers = num_workers
        )

        self.initial_loader = DataLoader(
            initial_dataset,
            batch_size = initial_batch_size,
            drop_last = drop_last,
            shuffle = shuffle,
            num_workers = num_workers
        )

        self.boundary_loader = DataLoader(
            boundary_dataset,
            batch_size=boundary_batch_size,
            drop_last=drop_last,
            num_workers = num_workers
        )
        self.field_iter = None
        self.boundar_iter = None
        self.initial_iter = None

    def __len__(self):
        return len(self.field_loader)

    def __iter__(self):
        self.field_iter = self.field_loader.__iter__()
        self.boundar_iter = self.boundary_loader.__iter__()
        self.initial_iter = self.initial_loader.__iter__()
        return self

    def __next__(self):
        field_data = next(self.field_iter)

        try:
            initial_data = next(self.initial_iter)
        except StopIteration:
            self.boundar_iter = self.initial_loader.__iter__()
            initial_data = next(self.initial_iter)

        try:
            boundary_data = next(self.boundar_iter)
        except StopIteration:
            self.boundar_iter = self.boundary_loader.__iter__()
            boundary_data = next(self.boundar_iter)

        return field_data, initial_data, boundary_data
