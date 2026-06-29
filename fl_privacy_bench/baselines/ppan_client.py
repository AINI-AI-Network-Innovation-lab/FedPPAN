

import flwr as fl
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fl_privacy_bench.models.privacy_mechanism import PrivacyMechanism
from fl_privacy_bench.metrics.metrics import compute_privacy_leakage, compute_distortion
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
from flwr.common import NDArrays, Scalar, Parameters
from typing import Dict, Tuple
from fl_privacy_bench.core.config import DEVICE



# Define PrivacyClient for Federated Learning
class PrivacyClient(fl.client.NumPyClient):
    def __init__(
        self,
        model,
        train_loader,
        privacy_weight,
        learning_rate=0.01,
        noise_scale=0.01,
        max_privacy_weight=1.0,
        max_privacy_loss=1.0,
        distortion_weight=0.01,
        max_grad_norm=1.0,
        max_param_abs=10.0,
    ):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.param_shapes = [p.shape for p in self.model.parameters()]
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.privacy_mech = PrivacyMechanism(self.total_params, noise_scale=noise_scale, device=DEVICE).to(DEVICE)
        self.optimizer = optim.SGD(
            list(self.model.parameters()) + list(self.privacy_mech.parameters()),
            lr=learning_rate
        )
        self.privacy_weight = privacy_weight
        self.effective_privacy_weight = min(float(privacy_weight), max_privacy_weight)
        self.max_privacy_loss = max_privacy_loss
        self.distortion_weight = distortion_weight
        self.max_grad_norm = max_grad_norm
        self.max_param_abs = max_param_abs

    def get_parameters(self, config=None):
        """Trả về tham số của mô hình dưới dạng danh sách NumPy arrays"""
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Cập nhật mô hình với danh sách NumPy arrays"""
        current_state = self.model.state_dict()
        state_dict = {
            key: torch.tensor(value, dtype=current_state[key].dtype, device=DEVICE)
            for key, value in zip(current_state.keys(), parameters)
        }
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        self.model.train()
        self.privacy_mech.train()

        total_loss, correct = 0, 0

        for images, labels in self.train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            self.optimizer.zero_grad()

            # Forward qua mô hình chính
            task_outputs = self.model(images)
            task_loss = F.cross_entropy(task_outputs, labels)
            if not torch.isfinite(task_loss):
                continue

            # Lấy trọng số mô hình → flatten → mã hóa + giải mã
            flat_weights = torch.cat([p.view(-1) for p in self.model.parameters()], dim=0).unsqueeze(0).to(DEVICE)
            encrypted, decoded = self.privacy_mech(flat_weights)

            # Privacy loss: MSE giữa bản giải mã và bản gốc (khó phục hồi là tốt)
            reconstruction_loss = F.mse_loss(decoded, flat_weights.detach())
            privacy_loss = -torch.clamp(reconstruction_loss, max=self.max_privacy_loss)

            # Distortion loss: giữ cho bản mã hóa gần bản gốc
            distortion_loss = F.mse_loss(encrypted, flat_weights.detach())

            total_batch_loss = (
                task_loss
                + self.effective_privacy_weight * privacy_loss
                + self.distortion_weight * distortion_loss
            )
            if not torch.isfinite(total_batch_loss):
                continue
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.privacy_mech.parameters()),
                self.max_grad_norm,
            )
            self.optimizer.step()

            total_loss += total_batch_loss.item()
            correct += (task_outputs.argmax(dim=1) == labels).sum().item()

        avg_loss = total_loss / max(1, len(self.train_loader))
        accuracy = correct / max(1, len(self.train_loader.dataset))

        # Mã hóa tham số mô hình để gửi về server
        with torch.no_grad():
            for p in self.model.parameters():
                p.data = torch.nan_to_num(p.data).clamp_(-self.max_param_abs, self.max_param_abs)
            server_params = self.get_parameters()
            trainable_params = [p.detach().cpu().numpy() for p in self.model.parameters()]
            flat_params = np.concatenate([p.flatten() for p in trainable_params])
            flat_tensor = torch.tensor(flat_params, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            encrypted_params = self.privacy_mech.encrypt(flat_tensor)
            encrypted_np = encrypted_params.detach().cpu().numpy()

            privacy_leakage = float(compute_privacy_leakage(encrypted_np, flat_params))
            distortion = float(compute_distortion(flat_params, encrypted_np.flatten()))

        return server_params, len(self.train_loader.dataset), {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
            "privacy_leakage": privacy_leakage,
            "distortion": distortion
        }
   
