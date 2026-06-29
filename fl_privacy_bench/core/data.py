from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

try:
    from fl_privacy_bench.data.data_handling import (
        get_dataloader,
        get_dataloader_cifar10,
        HAS_FLWR_DATASETS,
        split_cifar10_dirichlet_flwr,
        split_mnist_dirichlet_flwr,
    )
except ModuleNotFoundError:
    HAS_FLWR_DATASETS = False

from .config import ExperimentConfig


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dirichlet_partition_indices(labels: np.ndarray, num_clients: int, alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for cls in range(num_classes):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        splits = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        chunks = np.split(cls_idx, splits)
        for cid, chunk in enumerate(chunks):
            if chunk.size > 0:
                client_indices[cid].extend(chunk.tolist())

    sizes = [len(idx) for idx in client_indices]
    for cid in range(num_clients):
        if sizes[cid] > 0:
            continue
        donor = int(np.argmax(sizes))
        if sizes[donor] <= 1:
            continue
        client_indices[cid].append(client_indices[donor].pop())
        sizes[cid] += 1
        sizes[donor] -= 1

    return {f"client_{cid}": sorted(indices) for cid, indices in enumerate(client_indices)}


def _torchvision_trainset(cfg: ExperimentConfig):
    if cfg.dataset == "fashion":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        trainset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    else:
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

    labels = np.asarray(trainset.targets, dtype=np.int64)
    partitions = _dirichlet_partition_indices(
        labels=labels,
        num_clients=cfg.num_clients,
        alpha=cfg.alpha_dirichlet,
        seed=cfg.seed,
    )
    return trainset, partitions


def build_dataset_partition(cfg: ExperimentConfig):
    if HAS_FLWR_DATASETS:
        if cfg.dataset == "fashion":
            return split_mnist_dirichlet_flwr(
                num_clients=cfg.num_clients,
                alpha=cfg.alpha_dirichlet,
                seed=cfg.seed,
            )
        return split_cifar10_dirichlet_flwr(
            num_clients=cfg.num_clients,
            alpha=cfg.alpha_dirichlet,
            seed=cfg.seed,
        )
    return _torchvision_trainset(cfg)


def build_train_loader(cfg: ExperimentConfig, client_partition, trainset=None):
    if HAS_FLWR_DATASETS:
        if cfg.dataset == "fashion":
            return get_dataloader(client_partition, batch_size=cfg.batch_size)
        return get_dataloader_cifar10(client_partition, batch_size=cfg.batch_size)
    if trainset is None:
        raise ValueError("trainset is required when flwr-datasets is unavailable.")
    subset = Subset(trainset, client_partition)
    return DataLoader(subset, batch_size=cfg.batch_size, shuffle=True)


def build_centralized_testset(dataset: str):
    if dataset == "fashion":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        return datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    return datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
