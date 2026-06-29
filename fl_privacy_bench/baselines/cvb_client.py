from typing import Dict, Tuple
from collections import OrderedDict

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import NDArrays, Scalar

from fl_privacy_bench.baselines.cvb_config import CVB_BETA, DEVICE
from fl_privacy_bench.metrics.metrics import compute_distortion, compute_privacy_leakage


class CVBPrivacyClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, learning_rate: float):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config=None):
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_keys = list(self.model.state_dict().keys())
        state_dict = OrderedDict()
        for key, arr in zip(state_keys, parameters):
            state_dict[key] = torch.tensor(arr, device=DEVICE)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        self.model.train()

        total_loss = 0.0
        total_ce = 0.0
        total_kl = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in self.train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            ce_loss = self.criterion(outputs, labels)
            kl_loss = self.model.get_kl_loss()
            loss = ce_loss + CVB_BETA * kl_loss
            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_ce += ce_loss.item() * batch_size
            total_kl += kl_loss.item() * batch_size
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()

        updated_parameters = self.get_parameters()
        flat_params = np.concatenate([p.flatten() for p in updated_parameters])
        noise_scale = float(max(1e-6, np.sqrt(CVB_BETA) * 1e-3))
        protected_params = flat_params + np.random.normal(0.0, noise_scale, size=flat_params.shape)

        privacy_leakage = float(compute_privacy_leakage(protected_params, flat_params))
        distortion = float(compute_distortion(flat_params, protected_params))

        return updated_parameters, total_samples, {
            "loss": float(total_loss / total_samples),
            "ce_loss": float(total_ce / total_samples),
            "kl_loss": float(total_kl / total_samples),
            "accuracy": float(total_correct / total_samples),
            "privacy_leakage": privacy_leakage,
            "distortion": distortion,
        }
