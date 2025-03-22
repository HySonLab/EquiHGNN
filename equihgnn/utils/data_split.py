import torch_geometric.transforms as T
from torch.utils.data import random_split

from equihgnn.data import OneTarget, OPVBase, PCQM4Mv2Base
from equihgnn.utils.create import create_data


def create_train_val_test_set_and_normalize(target: int, data_name: str, data_dir: str):
    transform = T.Compose([OneTarget(target=target)])
    data_cls = create_data(data_name=data_name)

    print(f"Use {data_cls.__name__} dataset")

    if issubclass(data_cls, OPVBase):
        if target in [0, 1, 2, 3]:
            polymer = False
        elif target in [4, 5, 6, 7]:
            polymer = True
        else:
            raise Exception("Invalid target value!")

        train_dataset = data_cls(
            root=data_dir,
            polymer=polymer,
            partition="train",
            transform=transform,
        )
        valid_dataset = data_cls(
            root=data_dir,
            polymer=polymer,
            partition="valid",
            transform=transform,
        )
        test_dataset = data_cls(
            root=data_dir,
            polymer=polymer,
            partition="test",
            transform=transform,
        )

        # Normalize targets to mean = 0 and std = 1.
        mean = train_dataset._data.y.mean(dim=0, keepdim=True)
        std = train_dataset._data.y.std(dim=0, keepdim=True)
        train_dataset._data.y = (train_dataset._data.y - mean) / std
        valid_dataset._data.y = (valid_dataset._data.y - mean) / std
        test_dataset._data.y = (test_dataset._data.y - mean) / std

    else:
        if isinstance(data_cls, PCQM4Mv2Base):
            dataset = data_cls(root=data_dir)
        else:
            dataset = data_cls(root=data_dir, transform=transform)

        train_ratio = 0.8
        valid_ratio = 0.1

        # Calculate the number of samples in each set
        train_size = int(train_ratio * len(dataset))
        valid_size = int(valid_ratio * len(dataset))
        test_size = len(dataset) - train_size - valid_size

        # Split the dataset
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset, [train_size, valid_size, test_size]
        )

        # Normalize targets to mean = 0 and std = 1.
        mean = train_dataset.dataset.y.mean(dim=0, keepdim=True)
        std = train_dataset.dataset.y.std(dim=0, keepdim=True)
        train_dataset.dataset.y = (train_dataset.dataset.y - mean) / std
        valid_dataset.dataset.y = (valid_dataset.dataset.y - mean) / std
        test_dataset.dataset.y = (test_dataset.dataset.y - mean) / std

    if mean.dim() != 1:
        mean, std = mean[:, target].item(), std[:, target].item()
    else:
        mean, std = mean.item(), std.item()

    return train_dataset, valid_dataset, test_dataset, std
