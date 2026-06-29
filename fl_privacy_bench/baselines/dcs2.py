import argparse
import os
import sys

import flwr as fl
from flwr.common import Context
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fl_privacy_bench.baselines.dcs2_config import DCS2_PER_ADV
from fl_privacy_bench.baselines.dcs2_client import DCS2PrivacyClient
from fl_privacy_bench.core.cli import add_common_experiment_args, collect_common_overrides, validate_or_skip_protocol
from fl_privacy_bench.core.config import build_experiment_config
from fl_privacy_bench.core.data import build_train_loader
from fl_privacy_bench.core.simulation import run_federated_simulation
from fl_privacy_bench.models.mnist_model import CNN4, Net


def parse_args():
    parser = argparse.ArgumentParser(description="Run the DCS2 baseline with the synchronized FL Privacy Bench protocol.")
    add_common_experiment_args(parser)
    return parser.parse_args()


def model_factory(cfg):
    if cfg.dataset == "fashion":
        return Net
    return lambda: CNN4(num_channel=3, num_classes=cfg.num_classes)


def build_dcs2_loaders(cfg, client_partition, trainset):
    train_loader = build_train_loader(cfg, client_partition, trainset=trainset)
    proxy_batch_size = max(1, cfg.batch_size * DCS2_PER_ADV)
    proxy_loader = DataLoader(
        train_loader.dataset,
        batch_size=proxy_batch_size,
        shuffle=True,
        drop_last=(len(train_loader.dataset) >= proxy_batch_size),
        pin_memory=False,
    )
    return train_loader, proxy_loader


def client_fn_factory(cfg):
    def _factory(federated_data, trainset):
        def client_fn(context: Context) -> fl.client.Client:
            partition_id = int(context.node_config["partition-id"])
            key = f"client_{partition_id}"
            if key not in federated_data:
                raise ValueError(f"Client ID {partition_id} does not exist in federated_data")
            train_loader, proxy_loader = build_dcs2_loaders(cfg, federated_data[key], trainset)
            return DCS2PrivacyClient(
                model_factory(cfg)(),
                train_loader,
                proxy_loader,
                learning_rate=cfg.learning_rate,
                num_classes=cfg.num_classes,
            ).to_client()

        return client_fn

    return _factory


def main():
    args = parse_args()
    cfg = build_experiment_config("dcs2_fl", args.dataset, args.seed, overrides=collect_common_overrides(args))
    validate_or_skip_protocol(cfg, args)
    run_federated_simulation(
        cfg=cfg,
        model_factory=model_factory(cfg),
        client_fn_factory=client_fn_factory(cfg),
        persisted_fit_metrics=("privacy_leakage", "distortion", "conceal_obj", "proj_applied_ratio"),
    )


if __name__ == "__main__":
    main()
