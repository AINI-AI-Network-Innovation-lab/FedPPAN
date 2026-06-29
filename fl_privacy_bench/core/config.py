from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASELINE_PROTOCOLS: Dict[str, Dict[str, Any]] = {
    "fashion": {
        "dataset": "fashion",
        "num_clients": 100,
        "clients_per_round": 10,
        "num_rounds": 200,
        "batch_size": 16,
        "learning_rate": 0.01,
        "num_channel": 1,
        "num_classes": 10,
        "model_name": "net",
    },
    "cifar": {
        "dataset": "cifar",
        "num_clients": 50,
        "clients_per_round": 10,
        "num_rounds": 500,
        "batch_size": 16,
        "learning_rate": 0.03,
        "num_channel": 3,
        "num_classes": 10,
        "model_name": "cnn4",
    },
}


@dataclass
class ExperimentConfig:
    algorithm: str
    dataset: str
    seed: int = 42
    profile: str = "baseline"
    num_clients: int = 100
    clients_per_round: int = 10
    num_rounds: int = 200
    batch_size: int = 16
    learning_rate: float = 0.01
    num_channel: int = 1
    num_classes: int = 10
    model_name: str = "net"
    alpha_dirichlet: float = 0.5
    results_root: str = "results"
    results_dir: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def fraction_fit(self) -> float:
        return self.clients_per_round / self.num_clients

    @property
    def fraction_evaluate(self) -> float:
        return self.clients_per_round / self.num_clients

    @staticmethod
    def _safe_name(value: str) -> str:
        return (
            str(value)
            .strip()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(".", "p")
        )

    @property
    def metric_prefix(self) -> str:
        if self.profile and self.profile != "baseline":
            return f"{self._safe_name(self.algorithm)}_{self._safe_name(self.profile)}"
        return self._safe_name(self.algorithm)

    def ensure_results_dir(self) -> str:
        if self.results_dir is None:
            self.results_dir = os.path.join(self.results_root, self.dataset, f"seed_{self.seed}")
        os.makedirs(self.results_dir, exist_ok=True)
        return self.results_dir

    def validate(self, check_rounds: bool = True, enforce_protocol: bool = True) -> None:
        if self.dataset not in BASELINE_PROTOCOLS:
            raise ValueError(f"Unsupported dataset '{self.dataset}'. Use 'fashion' or 'cifar'.")
        if self.num_clients <= 0:
            raise ValueError("num_clients must be positive.")
        if self.clients_per_round <= 0:
            raise ValueError("clients_per_round must be positive.")
        if self.clients_per_round > self.num_clients:
            raise ValueError("clients_per_round cannot be greater than num_clients.")
        if self.num_rounds <= 0:
            raise ValueError("num_rounds must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.alpha_dirichlet <= 0:
            raise ValueError("alpha_dirichlet must be positive.")

        if not enforce_protocol:
            return

        expected = BASELINE_PROTOCOLS[self.dataset]
        protocol_keys = ("num_clients", "clients_per_round", "num_rounds", "learning_rate")
        for key in protocol_keys:
            if key == "num_rounds" and not check_rounds:
                continue
            if getattr(self, key) != expected[key]:
                raise ValueError(
                    f"Protocol mismatch for {self.dataset}: {key}={getattr(self, key)} != {expected[key]}"
                )

    def to_legacy_dict(self) -> Dict[str, Any]:
        cfg = {
            "ALGORITHM": self.algorithm,
            "DATASET": self.dataset,
            "PROFILE": self.profile,
            "SEED": self.seed,
            "NUM_CLIENTS": self.num_clients,
            "CLIENTS_PER_ROUND": self.clients_per_round,
            "NUM_ROUNDS": self.num_rounds,
            "BATCH_SIZE": self.batch_size,
            "LEARNING_RATE": self.learning_rate,
            "NUM_CHANNEL": self.num_channel,
            "NUM_CLASSES": self.num_classes,
            "MODEL_NAME": self.model_name,
            "ALPHA_DIRICHLET": self.alpha_dirichlet,
            "FRACTION_FIT": self.fraction_fit,
            "FRACTION_EVALUATE": self.fraction_evaluate,
            "RESULTS_DIR": self.ensure_results_dir(),
            "METRIC_PREFIX": self.metric_prefix,
        }
        cfg.update(self.extras)
        return cfg


def build_experiment_config(
    algorithm: str,
    dataset: str,
    seed: int = 42,
    profile: str = "baseline",
    overrides: Optional[Dict[str, Any]] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> ExperimentConfig:
    dataset = dataset.lower()
    if dataset not in BASELINE_PROTOCOLS:
        raise ValueError(f"Unsupported dataset '{dataset}'. Use 'fashion' or 'cifar'.")

    base = dict(BASELINE_PROTOCOLS[dataset])
    if overrides:
        base.update({k: v for k, v in overrides.items() if v is not None})

    cfg = ExperimentConfig(
        algorithm=algorithm,
        dataset=base["dataset"],
        seed=int(seed),
        profile=profile,
        num_clients=int(base["num_clients"]),
        clients_per_round=int(base["clients_per_round"]),
        num_rounds=int(base["num_rounds"]),
        batch_size=int(base["batch_size"]),
        learning_rate=float(base["learning_rate"]),
        num_channel=int(base["num_channel"]),
        num_classes=int(base["num_classes"]),
        model_name=str(base["model_name"]),
        alpha_dirichlet=float(base.get("alpha_dirichlet", 0.5)),
        extras=dict(extras or {}),
    )
    cfg.ensure_results_dir()
    return cfg


def parse_float_list(raw: str | Iterable[float]) -> list[float]:
    if isinstance(raw, str):
        return [float(item.strip()) for item in raw.split(",") if item.strip()]
    return [float(item) for item in raw]
