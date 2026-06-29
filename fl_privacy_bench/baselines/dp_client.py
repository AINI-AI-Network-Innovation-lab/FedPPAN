import flwr as fl
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fl_privacy_bench.metrics.metrics import compute_privacy_leakage, compute_distortion
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple
from flwr.common import NDArrays
from fl_privacy_bench.core.config import DEVICE


def add_gaussian_noise_to_parameters(parameters, epsilon, delta, sensitivity, max_param_abs=10.0):
    sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon
    noisy_parameters = [
        (
            np.clip(
                np.nan_to_num(param + np.random.normal(loc=0.0, scale=sigma, size=param.shape)),
                -max_param_abs,
                max_param_abs,
            ).astype(param.dtype, copy=False)
            if np.issubdtype(param.dtype, np.floating)
            else param
        )
        for param in parameters
    ]
    return noisy_parameters


class PrivacyClient(fl.client.NumPyClient):
    def __init__(
        self,
        model,
        train_loader,
        epsilon=1.0,
        delta=1e-5,
        sensitivity=1.0,
        learning_rate=0.01,
        max_grad_norm=1.0,
        max_param_abs=10.0,
    ):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.param_shapes = [p.shape for p in self.model.parameters()]
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.max_grad_norm = max_grad_norm
        self.max_param_abs = max_param_abs

    def get_parameters(self, config=None):
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        current_state = self.model.state_dict()
        state_dict = {
            key: torch.tensor(value, dtype=current_state[key].dtype, device=DEVICE)
            for key, value in zip(current_state.keys(), parameters)
        }
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, str] = {}) -> Tuple[NDArrays, int, Dict[str, float]]:
        self.set_parameters(parameters)
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in self.train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        accuracy = float(total_correct / max(1, total_samples))
        avg_loss = float(total_loss / max(1, total_samples))

        # Lấy tham số mô hình và thêm nhiễu DP
        with torch.no_grad():
            for p in self.model.parameters():
                p.data = torch.nan_to_num(p.data).clamp_(-self.max_param_abs, self.max_param_abs)
        updated_parameters = self.get_parameters()
        noisy_parameters = add_gaussian_noise_to_parameters(
            updated_parameters,
            self.epsilon,
            self.delta,
            self.sensitivity,
            max_param_abs=self.max_param_abs,
        )

        # Tính privacy leakage và distortion
        float_pairs = [
            (orig, noisy)
            for orig, noisy in zip(updated_parameters, noisy_parameters)
            if np.issubdtype(orig.dtype, np.floating)
        ]
        flat_params = np.concatenate([p.flatten() for p, _ in float_pairs])
        encrypted_np = np.concatenate([p.flatten() for _, p in float_pairs])

        privacy_leakage = float(compute_privacy_leakage(encrypted_np, flat_params))
        distortion = float(compute_distortion(flat_params, encrypted_np))

        return noisy_parameters, total_samples, {
            "loss": avg_loss,
            "accuracy": accuracy,
            "privacy_leakage": privacy_leakage,
            "distortion": distortion,
        }
