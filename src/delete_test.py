from multiprocessing import Manager

import torch
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, shared_dict):
        self.shared_dict = shared_dict
        self.length = 100

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self.shared_dict[index] = index
        return index


if __name__ == "__main__":
    manager = Manager()
    shared_dict = manager.dict()

    my_dict = {3000: 3000}
    shared_dict.update(my_dict)

    dataset = CustomDataset(shared_dict)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4)

    for batch in dataloader:
        print(batch)

    print(shared_dict)
