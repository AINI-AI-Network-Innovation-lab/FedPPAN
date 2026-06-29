import argparse
import os
import sys

import flwr as fl
from flwr.common import Context

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fl_privacy_bench.baselines.ppan_config import (
    DISTORTION_WEIGHT,
    MAX_GRAD_NORM,
    MAX_PARAM_ABS,
    MAX_PRIVACY_LOSS,
    MAX_PRIVACY_WEIGHT,
    NOISE_SCALE,
    PRIVACY_WEIGHT,
)
from fl_privacy_bench.baselines.ppan_client import PrivacyClient
from fl_privacy_bench.core.cli import (
    add_common_experiment_args,
    collect_common_overrides,
    validate_or_skip_protocol,
)
from fl_privacy_bench.core.config import build_experiment_config, parse_float_list
from fl_privacy_bench.core.data import build_train_loader
from fl_privacy_bench.core.simulation import run_federated_simulation
from fl_privacy_bench.models.mnist_model import CNN4, Net


def parse_args():
    parser = argparse.ArgumentParser(description="Run PPAN-FedAvg with the synchronized FL Privacy Bench protocol.")
    add_common_experiment_args(parser)
    parser.add_argument(
        "--privacy-weights",
        default=",".join(str(x) for x in PRIVACY_WEIGHT),
        help="Comma-separated privacy weights. Each value runs one simulation.",
    )
    parser.add_argument("--noise-scale", type=float, default=NOISE_SCALE)
    parser.add_argument("--max-privacy-weight", type=float, default=MAX_PRIVACY_WEIGHT)
    parser.add_argument("--max-privacy-loss", type=float, default=MAX_PRIVACY_LOSS)
    parser.add_argument("--distortion-weight", type=float, default=DISTORTION_WEIGHT)
    parser.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM)
    parser.add_argument("--max-param-abs", type=float, default=MAX_PARAM_ABS)
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
                privacy_weight=cfg.extras["privacy_weight"],
                learning_rate=cfg.learning_rate,
                noise_scale=cfg.extras["noise_scale"],
                max_privacy_weight=cfg.extras["max_privacy_weight"],
                max_privacy_loss=cfg.extras["max_privacy_loss"],
                distortion_weight=cfg.extras["distortion_weight"],
                max_grad_norm=cfg.extras["max_grad_norm"],
                max_param_abs=cfg.extras["max_param_abs"],
            ).to_client()

        return client_fn

    return _factory


def run_one(args, privacy_weight: float) -> None:
    cfg = build_experiment_config(
        "ppan_fl",
        args.dataset,
        args.seed,
        profile=f"privacy_{privacy_weight:g}",
        overrides=collect_common_overrides(args),
        extras={
            "privacy_weight": privacy_weight,
            "noise_scale": args.noise_scale,
            "max_privacy_weight": args.max_privacy_weight,
            "max_privacy_loss": args.max_privacy_loss,
            "distortion_weight": args.distortion_weight,
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


def main():
    args = parse_args()
    for privacy_weight in parse_float_list(args.privacy_weights):
        run_one(args, privacy_weight)


if __name__ == "__main__":
    main()
