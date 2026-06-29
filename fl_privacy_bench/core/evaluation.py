from __future__ import annotations

import os
from typing import Callable

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import DEVICE, ExperimentConfig


def metric_filename(metric_name: str, metric_prefix: str | None = None) -> str:
    if metric_prefix:
        return f"{metric_prefix}_{metric_name}.txt"
    return f"{metric_name}.txt"


def write_metric(
    results_dir: str,
    metric_name: str,
    value: float,
    server_round: int,
    metric_prefix: str | None = None,
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, metric_filename(metric_name, metric_prefix))
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"round={server_round},value={value:.8f}\n")


def build_centralized_evaluate_fn(
    cfg: ExperimentConfig,
    model_factory: Callable[[], nn.Module],
    testset,
):
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    results_dir = cfg.ensure_results_dir()
    metric_prefix = cfg.metric_prefix

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        model = model_factory().to(DEVICE)
        model.eval()

        model_keys = list(model.state_dict().keys())
        state_dict = {}
        current_state = model.state_dict()
        for key, weight in zip(model_keys, parameters):
            weight_tensor = torch.tensor(weight, device=DEVICE)
            if weight_tensor.shape == current_state[key].shape:
                state_dict[key] = weight_tensor
        model.load_state_dict(state_dict, strict=False)

        criterion = nn.CrossEntropyLoss()
        correct, total, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total else 0.0
        avg_loss = total_loss / total if total else 0.0
        write_metric(results_dir, "centralized_accuracy", accuracy, server_round, metric_prefix)
        write_metric(results_dir, "centralized_loss", avg_loss, server_round, metric_prefix)
        return avg_loss, {"accuracy": accuracy}

    return evaluate
