from typing import Tuple
from random import randint

import numpy as np
from flex.data import Dataset, FedDataDistribution, FedDataset, FedDatasetConfig
from flexclash.data import data_poisoner


def get_dataset(dataset: str) -> Tuple[FedDataset, Dataset]:
    match dataset:
        case "emnist_non_iid":
            return _emnist_non_iid()
        case "mnist":
            return _emnist_iid()
        case "fashion":
            return _fashion_emnist()
        case "fasion_non_iid":
            return _fashion_non_iid()
        case "celeba_iid":
            return _celeba_iid()
        case "celeba":
            return _celeba_non_iid()
        case "cifar_10":
            return _cifar_10_iid()
        case _:
            raise ValueError(f"Unknown dataset: {dataset}")


def _emnist_non_iid() -> Tuple[FedDataset, Dataset]:
    from flex.datasets.federated_datasets import federated_emnist

    flex_dataset, test_data = federated_emnist(return_test=True)
    nodes = list(flex_dataset.keys())
    for node in nodes:
        if len(flex_dataset[node]) < 30:
            del flex_dataset[node]

    return flex_dataset, test_data


def _emnist_iid() -> Tuple[FedDataset, Dataset]:
    from flex.datasets.standard_datasets import emnist

    flex_dataset, test_data = emnist()

    config = FedDatasetConfig(seed=0)
    config.replacement = False
    config.n_nodes = 200

    flex_dataset = FedDataDistribution.from_config(flex_dataset, config)

    return flex_dataset, test_data


def _fashion_emnist() -> Tuple[FedDataset, Dataset]:
    from torchvision.datasets import FashionMNIST

    train_data = FashionMNIST(root=".", train=True, download=True, transform=None)

    test_data = FashionMNIST(root=".", train=False, download=True, transform=None)
    flex_dataset = Dataset.from_torchvision_dataset(train_data)
    test_data = Dataset.from_torchvision_dataset(test_data)

    config = FedDatasetConfig(seed=0)
    config.replacement = False
    config.n_nodes = 200

    flex_dataset = FedDataDistribution.from_config(flex_dataset, config)

    data_threshold = 30
    cids = list(flex_dataset.keys())
    for k in cids:
        if len(flex_dataset[k]) < data_threshold:
            del flex_dataset[k]

    assert isinstance(flex_dataset, FedDataset)

    return flex_dataset, test_data


def _fashion_non_iid() -> Tuple[FedDataset, Dataset]:
    import dill as pickle
    from torchvision.datasets import FashionMNIST

    try:
        flex_dataset = pickle.load(open("fashion_non_iid_fed.pck", "rb"))
        test_data = pickle.load(open("fashion_non_iid_test.pck", "rb"))
        return flex_dataset, test_data
    except FileNotFoundError:
        pass

    train_data = FashionMNIST(root=".", train=True, download=True, transform=None)

    test_data = FashionMNIST(root=".", train=False, download=True, transform=None)
    flex_dataset = Dataset.from_torchvision_dataset(train_data)
    test_data = Dataset.from_torchvision_dataset(test_data)

    config = FedDatasetConfig(seed=0)
    config.labels_per_node = (2, 8)
    config.replacement = True
    config.n_nodes = 200

    flex_dataset = FedDataDistribution.from_config(flex_dataset, config)

    data_threshold = 30
    cids = list(flex_dataset.keys())
    for k in cids:
        if len(flex_dataset[k]) < data_threshold:
            del flex_dataset[k]

    assert isinstance(flex_dataset, FedDataset)

    pickle.dump(flex_dataset, open("fashion_non_iid_fed.pck", "wb"))
    pickle.dump(test_data, open("fashion_non_iid_test.pck", "wb"))

    return flex_dataset, test_data


def _celeba_iid():
    from torchvision.datasets import CelebA

    train_data = CelebA(root="..", split="train", download=True, transform=None)
    test_data = CelebA(root="..", split="test", download=True, transform=None)
    flex_dataset = Dataset.from_torchvision_dataset(train_data)
    test_data = Dataset.from_torchvision_dataset(test_data)

    config = FedDatasetConfig(seed=0)
    config.replacement = False
    config.n_nodes = 200
    flex_dataset = FedDataDistribution.from_config(flex_dataset, config)

    def select_label(dataset: Dataset):
        smiling_index = -9
        y_data = (
            [y[1] for y in dataset.y_data]
            if isinstance(dataset.y_data[0], tuple)
            else dataset.y_data
        )

        y_data = [np.eye(2)[label[smiling_index]] for label in y_data]
        return Dataset(X_data=dataset.X_data, y_data=y_data)

    flex_dataset = flex_dataset.apply(select_label)
    test_data = select_label(test_data)

    return flex_dataset, test_data


def _celeba_non_iid():
    import dill as pickle
    from flex.datasets.federated_datasets import federated_celeba

    try:
        with open("celeba_fed.pck", "rb") as f:
            flex_dataset = pickle.load(f)
        with open("celeba_test.pck", "rb") as f:
            test_data = pickle.load(f)
    except FileNotFoundError:
        flex_dataset, test_data = federated_celeba("..", return_test=True)
        pickle.dump(flex_dataset, open("celeba_fed.pck", "wb"))
        pickle.dump(test_data, open("celeba_test.pck", "wb"))

    def select_label(dataset: Dataset):
        smiling_index = -9
        y_data = (
            [y[1] for y in dataset.y_data]
            if isinstance(dataset.y_data[0], tuple)
            else dataset.y_data
        )

        y_data = [np.eye(2)[label[smiling_index]] for label in y_data]
        return Dataset(X_data=dataset.X_data, y_data=y_data)

    flex_dataset = flex_dataset.apply(select_label)
    test_data = select_label(test_data)

    return flex_dataset, test_data


def _cifar_10_iid():
    from torchvision.datasets import CIFAR10

    train_data = CIFAR10(root=".", train=True, download=True, transform=None)
    test_data = CIFAR10(root=".", train=False, download=True, transform=None)
    flex_dataset = Dataset.from_torchvision_dataset(train_data)
    test_data = Dataset.from_torchvision_dataset(test_data)

    config = FedDatasetConfig(seed=0)
    config.replacement = False
    config.n_nodes = 200

    flex_dataset = FedDataDistribution.from_config(flex_dataset, config)

    data_threshold = 30
    cids = list(flex_dataset.keys())
    for k in cids:
        if len(flex_dataset[k]) < data_threshold:
            del flex_dataset[k]

    assert isinstance(flex_dataset, FedDataset)

    return flex_dataset, test_data


def poison_dataset(dataset: FedDataset, num_classes: int, poison_ratio):
    poisoned_ids = list(dataset.keys())[: int(poison_ratio * len(dataset))]

    @data_poisoner
    def _label_flipping(img_array, label):
        while True:
            new_label = np.random.randint(0, num_classes)
            if new_label != label:
                break
        return img_array, new_label

    return dataset.apply(_label_flipping, node_ids=poisoned_ids), poisoned_ids


def poison_binary_dataset(dataset: FedDataset, poison_ratio):
    poisoned_ids = list(dataset.keys())[: int(poison_ratio * len(dataset))]

    @data_poisoner
    def _label_flipping(img_array, label):
        return img_array, np.eye(2)[randint(0, 1)]

    return dataset.apply(_label_flipping, node_ids=poisoned_ids), poisoned_ids
