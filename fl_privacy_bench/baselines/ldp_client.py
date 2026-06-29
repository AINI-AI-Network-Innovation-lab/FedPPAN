from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import NDArrays, Scalar

from fl_privacy_bench.baselines.ldp_config import DEVICE
from fl_privacy_bench.metrics.metrics import compute_distortion, compute_privacy_leakage


def _layer_id_from_key(key: str) -> str:
    parts = key.split(".")
    if len(parts) > 1 and parts[1].isdigit():
        return f"{parts[0]}.{parts[1]}"
    return parts[0]


def _make_rounds_per_cycle(total_rounds: int, num_cycles: int) -> List[int]:
    num_cycles = max(1, min(num_cycles, total_rounds))
    base = total_rounds // num_cycles
    rem = total_rounds % num_cycles
    rounds = [base] * num_cycles
    for i in range(rem):
        rounds[i] += 1
    return rounds


def _allocate_rounds_to_layers(layer_sizes: List[int], cycle_rounds: int) -> List[int]:
    num_layers = len(layer_sizes)
    if num_layers == 0:
        return []
    if cycle_rounds <= 0:
        return [0] * num_layers

    if cycle_rounds < num_layers:
        # Rare fallback when rounds are fewer than layers.
        order = sorted(range(num_layers), key=lambda i: layer_sizes[i], reverse=True)
        alloc = [0] * num_layers
        for i in order[:cycle_rounds]:
            alloc[i] = 1
        return alloc

    alloc = [1] * num_layers
    remaining = cycle_rounds - num_layers
    total_size = max(1, sum(layer_sizes))

    weighted = [remaining * s / total_size for s in layer_sizes]
    floors = [int(np.floor(w)) for w in weighted]
    for i, f in enumerate(floors):
        alloc[i] += f

    used = sum(floors)
    residual = remaining - used
    if residual > 0:
        fracs = [(weighted[i] - floors[i], i) for i in range(num_layers)]
        fracs.sort(reverse=True)
        for _, idx in fracs[:residual]:
            alloc[idx] += 1
    return alloc


class LDPFedClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        model: nn.Module,
        train_loader,
        learning_rate: float,
        num_rounds: int,
        num_cycles: int,
        total_alpha: float,
        precision_rho: int,
        clip_c: float,
        sampling_ratio: float = 1.0,
        use_sampling_amplification: bool = True,
        local_epochs: int = 1,
        seed: int = 42,
    ):
        self.cid = int(cid)
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        self.num_rounds = int(num_rounds)
        self.num_cycles = max(1, int(num_cycles))
        self.total_alpha = float(total_alpha)
        self.precision_rho = int(precision_rho)
        self.clip_c = float(clip_c)
        self.sampling_ratio = float(max(1e-12, sampling_ratio))
        self.use_sampling_amplification = bool(use_sampling_amplification)
        self.local_epochs = max(1, int(local_epochs))
        self.rng = np.random.default_rng(seed + 1009 * self.cid)

        state_dict = self.model.state_dict()
        self.param_keys = list(state_dict.keys())
        trainable_keys = set(dict(self.model.named_parameters()).keys())
        # Only perturb trainable parameters; keep buffers (e.g., BN running stats) untouched.
        self.float_mask = [key in trainable_keys for key in self.param_keys]

        layer_to_indices: "OrderedDict[str, List[int]]" = OrderedDict()
        for idx, key in enumerate(self.param_keys):
            if not self.float_mask[idx]:
                continue
            layer_id = _layer_id_from_key(key)
            layer_to_indices.setdefault(layer_id, []).append(idx)

        self.layer_names = list(layer_to_indices.keys())
        self.layer_to_indices = layer_to_indices
        self.layer_sizes = []
        for layer in self.layer_names:
            size = 0
            for idx in self.layer_to_indices[layer]:
                size += int(state_dict[self.param_keys[idx]].numel())
            self.layer_sizes.append(size)
        self.rounds_per_cycle = _make_rounds_per_cycle(self.num_rounds, self.num_cycles)

    def get_parameters(self, config=None):
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = OrderedDict()
        for key, arr in zip(self.param_keys, parameters):
            state_dict[key] = torch.tensor(arr, device=DEVICE)
        self.model.load_state_dict(state_dict, strict=True)

    def _get_cycle_context(self, server_round: int) -> Tuple[int, int, int]:
        ridx = min(max(server_round - 1, 0), self.num_rounds - 1)
        acc = 0
        for cycle_idx, cycle_rounds in enumerate(self.rounds_per_cycle):
            if ridx < acc + cycle_rounds:
                return cycle_idx, ridx - acc, cycle_rounds
            acc += cycle_rounds
        return len(self.rounds_per_cycle) - 1, self.rounds_per_cycle[-1] - 1, self.rounds_per_cycle[-1]

    def _select_layer_for_round(self, server_round: int) -> Tuple[int, int, float, float]:
        cycle_idx, pos_in_cycle, cycle_rounds = self._get_cycle_context(server_round)
        alloc = _allocate_rounds_to_layers(self.layer_sizes, cycle_rounds)

        schedule: List[int] = []
        # Backward stepping: output layer -> earlier layers.
        for layer_idx in range(len(self.layer_names) - 1, -1, -1):
            schedule.extend([layer_idx] * alloc[layer_idx])
        if not schedule:
            schedule = [max(0, len(self.layer_names) - 1)]

        selected_layer_idx = schedule[min(pos_in_cycle, len(schedule) - 1)]
        alpha_cycle = self.total_alpha / max(1, self.num_cycles)
        alpha_round = alpha_cycle / max(1, cycle_rounds)
        if self.use_sampling_amplification:
            # Paper: alpha = sum_i q * alpha_i, q = k / N.
            alpha_round = alpha_round / self.sampling_ratio
        alpha_per_param = alpha_round / max(1, self.layer_sizes[selected_layer_idx])
        return selected_layer_idx, cycle_idx, alpha_round, alpha_per_param

    def _ordinal_cldp_perturb(self, delta: np.ndarray, alpha_per_param: float) -> Tuple[np.ndarray, np.ndarray]:
        scale = 10 ** self.precision_rho
        bound = int(round(self.clip_c * scale))
        if bound <= 0:
            clipped = np.clip(delta, -self.clip_c, self.clip_c)
            return clipped, clipped

        clipped = np.clip(delta, -self.clip_c, self.clip_c)
        integers = np.rint(clipped * scale).astype(np.int64)

        # EM over ordinal distance has two-sided geometric form; clip to finite universe.
        p = max(1e-12, 1.0 - np.exp(-0.5 * alpha_per_param))
        g1 = self.rng.geometric(p, size=integers.shape) - 1
        g2 = self.rng.geometric(p, size=integers.shape) - 1
        noise = g1.astype(np.int64) - g2.astype(np.int64)
        perturbed_int = np.clip(integers + noise, -bound, bound)

        perturbed = perturbed_int.astype(np.float64) / scale
        clipped_float = integers.astype(np.float64) / scale
        return perturbed.astype(delta.dtype, copy=False), clipped_float.astype(delta.dtype, copy=False)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        server_round = int(config.get("round", 1))
        self.set_parameters(parameters)

        global_params = [np.array(p, copy=True) for p in parameters]

        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for _ in range(self.local_epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                bsz = labels.size(0)
                total_samples += bsz
                total_loss += float(loss.item()) * bsz
                _, predicted = torch.max(outputs, 1)
                total_correct += int((predicted == labels).sum().item())

        local_params = self.get_parameters()
        selected_layer_idx, cycle_idx, alpha_round, alpha_per_param = self._select_layer_for_round(server_round)
        selected_layer_name = self.layer_names[selected_layer_idx]
        selected_indices = set(self.layer_to_indices[selected_layer_name])

        outgoing_params: List[np.ndarray] = []
        selected_original: List[np.ndarray] = []
        selected_perturbed: List[np.ndarray] = []

        for idx, (global_arr, local_arr) in enumerate(zip(global_params, local_params)):
            if not self.float_mask[idx]:
                outgoing_params.append(global_arr)
                continue

            delta = local_arr - global_arr
            if idx in selected_indices:
                perturbed_delta, clipped_delta = self._ordinal_cldp_perturb(delta, alpha_per_param)
                outgoing_params.append((global_arr + perturbed_delta).astype(local_arr.dtype, copy=False))
                selected_original.append(clipped_delta.reshape(-1))
                selected_perturbed.append(perturbed_delta.reshape(-1))
            else:
                outgoing_params.append(global_arr.astype(local_arr.dtype, copy=False))

        if selected_original:
            original_vec = np.concatenate(selected_original)
            perturbed_vec = np.concatenate(selected_perturbed)
            privacy_leakage = float(compute_privacy_leakage(perturbed_vec, original_vec))
            distortion = float(compute_distortion(original_vec, perturbed_vec))
        else:
            privacy_leakage, distortion = 0.0, 0.0

        avg_loss = total_loss / max(1, total_samples)
        accuracy = total_correct / max(1, total_samples)
        return outgoing_params, total_samples, {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
            "privacy_leakage": float(privacy_leakage),
            "distortion": float(distortion),
            "alpha_round": float(alpha_round),
            "alpha_per_param": float(alpha_per_param),
            "selected_layer_idx": float(selected_layer_idx),
            "cycle_idx": float(cycle_idx),
        }
