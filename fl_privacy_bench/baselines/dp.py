import argparse
import os
import sys

import flwr as fl
from flwr.common import Context

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fl_privacy_bench.baselines.dp_client import PrivacyClient
from fl_privacy_bench.core.cli import add_common_experiment_args, collect_common_overrides, validate_or_skip_protocol
from fl_privacy_bench.core.config import build_experiment_config
from fl_privacy_bench.core.data import build_train_loader
from fl_privacy_bench.core.simulation import run_federated_simulation
from fl_privacy_bench.models.mnist_model import CNN4, Net


def parse_args():
    parser = argparse.ArgumentParser(description="Run Gaussian DP-FedAvg with the synchronized FL Privacy Bench protocol.")
    add_common_experiment_args(parser)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--sensitivity", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-param-abs", type=float, default=10.0)
    return parser.parse_args()


def model_factory(cfg):
    if cfg.dataset == "fashion":
        return Net
    return lambda: CNN4(num_channel=3, num_classes=cfg.num_classes)


def client_fn_factory(cfg):
    def _factory(federated_data, trainset):
        def client_fn(context: Context) -> fl.client.Client:
            partition_id = int(context.node_config["partition-id"])
            key = f"client_{partition_id}"
            if key not in federated_data:
                raise ValueError(f"Client ID {partition_id} does not exist in federated_data")
            train_loader = build_train_loader(cfg, federated_data[key], trainset=trainset)
            return PrivacyClient(
                model_factory(cfg)(),
                train_loader,
                epsilon=cfg.extras["epsilon"],
                delta=cfg.extras["delta"],
                sensitivity=cfg.extras["sensitivity"],
                learning_rate=cfg.learning_rate,
                max_grad_norm=cfg.extras["max_grad_norm"],
                max_param_abs=cfg.extras["max_param_abs"],
            ).to_client()

        return client_fn

    return _factory


def main():
    args = parse_args()
    cfg = build_experiment_config(
        "dp_fl",
        args.dataset,
        args.seed,
        profile=f"epsilon_{args.epsilon:g}",
        overrides=collect_common_overrides(args),
        extras={
            "epsilon": args.epsilon,
            "delta": args.delta,
            "sensitivity": args.sensitivity,
            "max_grad_norm": args.max_grad_norm,
            "max_param_abs": args.max_param_abs,
        },
    )
    validate_or_skip_protocol(cfg, args)
    run_federated_simulation(
        cfg=cfg,
        model_factory=model_factory(cfg),
        client_fn_factory=client_fn_factory(cfg),
        persisted_fit_metrics=("privacy_leakage", "distortion"),
    )


if __name__ == "__main__":
    main()
