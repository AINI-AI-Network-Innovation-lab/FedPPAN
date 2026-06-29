from typing import Dict, Tuple
from collections import OrderedDict
from contextlib import nullcontext
import time

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import NDArrays, Scalar

from fl_privacy_bench.baselines.dcs2_config import (
    DCS2_DCS_ITER,
    DCS2_DCS_LR,
    DCS2_EARLY_STOP,
    DCS2_ENABLE_AMP,
    DCS2_ENABLE_COMPILE,
    DCS2_EPSILON,
    DCS2_LAMBDA_G,
    DCS2_LAMBDA_X,
    DCS2_LAMBDA_Y,
    DCS2_LAMBDA_Z,
    DCS2_LOG_TIMING,
    DCS2_MIXUP,
    DCS2_NUM_SEN,
    DCS2_PER_ADV,
    DCS2_PROJECT,
    DCS2_STARTPOINT,
    DCS2_XSIM_THR,
    DEVICE,
)
from fl_privacy_bench.metrics.metrics import compute_distortion, compute_privacy_leakage
from fl_privacy_bench.models.dcs2 import DCS2Defender


class DCS2PrivacyClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, proxy_loader, learning_rate: float, num_classes: int):
        self.model = model.to(DEVICE)
        if DCS2_ENABLE_COMPILE and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass
        self.train_loader = train_loader
        self.proxy_loader = proxy_loader
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        self.use_amp = bool(DCS2_ENABLE_AMP and DEVICE.type == "cuda")
        self.defender = DCS2Defender(
            model=self.model,
            criterion=self.criterion,
            lambda_g=DCS2_LAMBDA_G,
            lambda_x=DCS2_LAMBDA_X,
            lambda_z=DCS2_LAMBDA_Z,
            epsilon=DCS2_EPSILON,
            dcs_iter=DCS2_DCS_ITER,
            dcs_lr=DCS2_DCS_LR,
            num_sen=DCS2_NUM_SEN,
            per_adv=DCS2_PER_ADV,
            lambda_y=DCS2_LAMBDA_Y,
            xsim_thr=DCS2_XSIM_THR,
            project=DCS2_PROJECT,
            mixup=DCS2_MIXUP,
            startpoint=DCS2_STARTPOINT,
            early_stop=DCS2_EARLY_STOP,
        )

    def get_parameters(self, config=None):
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def _autocast(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    @staticmethod
    def _forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        out = model(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

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
        total_correct = 0
        total_samples = 0
        conceal_obj_total = 0.0
        grad_cos_total = 0.0
        proj_ratio_total = 0.0
        dcs_time_total = 0.0
        num_batches = 0

        proxy_iter = iter(self.proxy_loader)

        for images, labels in self.train_loader:
            try:
                proxy_images, proxy_labels = next(proxy_iter)
            except StopIteration:
                proxy_iter = iter(self.proxy_loader)
                proxy_images, proxy_labels = next(proxy_iter)

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            proxy_images = proxy_images.to(DEVICE, non_blocking=True)
            proxy_labels = proxy_labels.to(DEVICE, non_blocking=True)

            if DCS2_STARTPOINT == "noise":
                proxy_images = torch.randn_like(proxy_images)

            self.optimizer.zero_grad()

            start = time.perf_counter()
            with self._autocast():
                grads, dcs_stats = self.defender.defense_optim(images, labels, proxy_images, proxy_labels)
            dcs_time_total += time.perf_counter() - start

            params = [p for p in self.model.parameters() if p.requires_grad]
            for p, g in zip(params, grads):
                p.grad = g
            self.optimizer.step()

            with torch.no_grad():
                with self._autocast():
                    outputs = self._forward_logits(self.model, images)
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)

            batch_size = labels.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_correct += (predicted == labels).sum().item()
            conceal_obj_total += dcs_stats["conceal_obj"]
            grad_cos_total += dcs_stats["grad_cosine"]
            proj_ratio_total += dcs_stats["proj_applied_ratio"]
            num_batches += 1

        updated_parameters = self.get_parameters()
        flat_params = np.concatenate([p.flatten() for p in updated_parameters])
        jitter = np.random.normal(0.0, 1e-6, size=flat_params.shape)
        privacy_leakage = float(compute_privacy_leakage(flat_params + jitter, flat_params))
        distortion = float(compute_distortion(flat_params, flat_params + jitter))

        return updated_parameters, total_samples, {
            "loss": float(total_loss / max(1, total_samples)),
            "accuracy": float(total_correct / max(1, total_samples)),
            "privacy_leakage": privacy_leakage,
            "distortion": distortion,
            "conceal_obj": float(conceal_obj_total / max(1, num_batches)),
            "grad_cosine": float(grad_cos_total / max(1, num_batches)),
            "proj_applied_ratio": float(proj_ratio_total / max(1, num_batches)),
            "dcs_time_sec": float(dcs_time_total) if DCS2_LOG_TIMING else 0.0,
        }
