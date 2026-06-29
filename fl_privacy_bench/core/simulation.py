from __future__ import annotations

from typing import Callable, Iterable

import flwr as fl
import torch

from fl_privacy_bench.server.aggregation import aggregate_evaluate_metrics, aggregate_fit_metrics
from fl_privacy_bench.server.client_manager import SimpleClientManager

from .config import ExperimentConfig
from .data import build_centralized_testset, build_dataset_partition, set_global_seed
from .evaluation import build_centralized_evaluate_fn
from .strategy import MetricsFedAvg, make_evaluate_config, make_fit_config


def client_resources() -> dict:
    resources = {"num_cpus": 1}
    if torch.cuda.is_available():
        resources["num_gpus"] = 0.1
    return resources


def run_federated_simulation(
    *,
    cfg: ExperimentConfig,
    model_factory: Callable[[], torch.nn.Module],
    client_fn_factory: Callable[[dict, object], Callable],
    persisted_fit_metrics: Iterable[str] = ("privacy_leakage", "distortion"),
) -> None:
    set_global_seed(cfg.seed)
    trainset, federated_data = build_dataset_partition(cfg)
    testset = build_centralized_testset(cfg.dataset)

    strategy = MetricsFedAvg(
        fraction_fit=cfg.fraction_fit,
        fraction_evaluate=0.0,
        min_fit_clients=cfg.clients_per_round,
        min_evaluate_clients=0,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=make_fit_config(cfg),
        on_evaluate_config_fn=make_evaluate_config(cfg),
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        evaluate_fn=build_centralized_evaluate_fn(cfg, model_factory, testset),
        results_dir=cfg.ensure_results_dir(),
        metric_prefix=cfg.metric_prefix,
        persisted_fit_metrics=persisted_fit_metrics,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn_factory(federated_data, trainset),
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_manager=SimpleClientManager(),
        client_resources=client_resources(),
    )
