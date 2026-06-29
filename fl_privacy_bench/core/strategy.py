from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from .evaluation import write_metric


class MetricsFedAvg(fl.server.strategy.FedAvg):
    """FedAvg with standardized server-side metric persistence."""

    def __init__(
        self,
        *,
        results_dir: str,
        metric_prefix: str | None = None,
        persisted_fit_metrics: Iterable[str] = ("privacy_leakage", "distortion"),
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 0,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.results_dir = results_dir
        self.metric_prefix = metric_prefix
        self.persisted_fit_metrics = tuple(persisted_fit_metrics)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        aggregated_result = super().aggregate_fit(server_round, results, failures)
        if aggregated_result is None:
            return None, {}

        parameters_aggregated, _ = aggregated_result
        ndarrays = parameters_to_ndarrays(parameters_aggregated)

        metrics_aggregated: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        for metric_name in self.persisted_fit_metrics:
            values = [float(res.metrics.get(metric_name, 0.0)) for _, res in results]
            write_metric(
                self.results_dir,
                metric_name,
                float(np.mean(values)) if values else 0.0,
                server_round,
                self.metric_prefix,
            )

        return ndarrays_to_parameters(ndarrays), metrics_aggregated

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        if self.fraction_evaluate <= 0.0 or self.min_evaluate_clients <= 0:
            return []
        config = {"round": server_round}
        if self.on_evaluate_config_fn:
            config.update(self.on_evaluate_config_fn(server_round))
        evaluate_ins = EvaluateIns(parameters=parameters, config=config)
        return [(client, evaluate_ins) for client in client_manager.all().values()]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        metrics_aggregated: Dict[str, float] = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)

        avg_loss = metrics_aggregated.get("loss")
        if avg_loss is None:
            avg_loss = float(np.mean([res.metrics.get("loss", 0.0) for _, res in results]))
        return avg_loss, metrics_aggregated


def make_fit_config(cfg) -> Callable[[int], Dict[str, Scalar]]:
    def _fit_config(server_round: int) -> Dict[str, Scalar]:
        return {
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "round": server_round,
        }

    return _fit_config


def make_evaluate_config(cfg) -> Callable[[int], Dict[str, Scalar]]:
    def _evaluate_config(server_round: int) -> Dict[str, Scalar]:
        return {"batch_size": cfg.batch_size, "round": server_round}

    return _evaluate_config
