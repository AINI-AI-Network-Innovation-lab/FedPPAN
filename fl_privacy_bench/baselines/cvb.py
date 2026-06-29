import argparse
import os
import sys

import flwr as fl
from flwr.common import Context

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fl_privacy_bench.baselines.cvb_config import CVB_KERNEL_SIZE, CVB_SCALE
from fl_privacy_bench.baselines.cvb_client import CVBPrivacyClient
from fl_privacy_bench.core.cli import add_common_experiment_args, collect_common_overrides, validate_or_skip_protocol
from fl_privacy_bench.core.config import build_experiment_config
from fl_privacy_bench.core.data import build_train_loader
from fl_privacy_bench.core.simulation import run_federated_simulation
from fl_privacy_bench.models.cvb import CVBCNN4, CVBNet


def parse_args():
    parser = argparse.ArgumentParser(description="Run the CVB baseline with the synchronized FL Privacy Bench protocol.")
    add_common_experiment_args(parser)
    return parser.parse_args()


def model_factory(cfg):
    if cfg.dataset == "fashion":
        return lambda: CVBNet(cvb_scale=CVB_SCALE, cvb_kernel_size=CVB_KERNEL_SIZE)
    return lambda: CVBCNN4(
        num_classes=cfg.num_classes,
        cvb_scale=CVB_SCALE,
        cvb_kernel_size=CVB_KERNEL_SIZE,
    )


def client_fn_factory(cfg):
    def _factory(federated_data, trainset):
        def client_fn(context: Context) -> fl.client.Client:
            partition_id = int(context.node_config["partition-id"])
            key = f"client_{partition_id}"
            if key not in federated_data:
                raise ValueError(f"Client ID {partition_id} does not exist in federated_data")
            train_loader = build_train_loader(cfg, federated_data[key], trainset=trainset)
            return CVBPrivacyClient(
                model_factory(cfg)(),
                train_loader,
                learning_rate=cfg.learning_rate,
            ).to_client()

        return client_fn

    return _factory


def main():
    args = parse_args()
    cfg = build_experiment_config("cvb_fl", args.dataset, args.seed, overrides=collect_common_overrides(args))
    validate_or_skip_protocol(cfg, args)
    run_federated_simulation(
        cfg=cfg,
        model_factory=model_factory(cfg),
        client_fn_factory=client_fn_factory(cfg),
        persisted_fit_metrics=("privacy_leakage", "distortion"),
    )


if __name__ == "__main__":
    main()
