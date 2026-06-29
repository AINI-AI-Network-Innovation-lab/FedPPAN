import argparse
import os
import sys

import flwr as fl
from flwr.common import Context

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fl_privacy_bench.baselines.ldp_config import build_ldp_experiment_config
from fl_privacy_bench.baselines.ldp_client import LDPFedClient
from fl_privacy_bench.core.cli import add_common_experiment_args
from fl_privacy_bench.core.data import build_train_loader
from fl_privacy_bench.core.simulation import run_federated_simulation
from fl_privacy_bench.models.mnist_model import CNN4, LDPFedFashionNet, Net


def parse_args():
    parser = argparse.ArgumentParser(description="Run LDP-Fed with the FL Privacy Bench runtime.")
    add_common_experiment_args(parser, include_profile=True, default_profile="auto")
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--rho", type=int, default=None)
    parser.add_argument("--clip-c", type=float, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--disable-sampling-amplification", action="store_true")
    return parser.parse_args()


def apply_args(cfg, args):
    if args.num_rounds is not None:
        cfg.num_rounds = int(args.num_rounds)
        if args.cycles is None:
            cfg.extras["NUM_CYCLES"] = min(cfg.extras["NUM_CYCLES"], cfg.num_rounds)
    if args.num_clients is not None:
        cfg.num_clients = int(args.num_clients)
    if args.clients_per_round is not None:
        cfg.clients_per_round = int(args.clients_per_round)
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.learning_rate is not None:
        cfg.learning_rate = float(args.learning_rate)
    if args.alpha_dirichlet is not None:
        cfg.alpha_dirichlet = float(args.alpha_dirichlet)
    if args.cycles is not None:
        cfg.extras["NUM_CYCLES"] = max(1, int(args.cycles))
    if args.alpha is not None:
        cfg.extras["TOTAL_ALPHA"] = float(args.alpha)
    if args.rho is not None:
        cfg.extras["PRECISION_RHO"] = int(args.rho)
    if args.clip_c is not None:
        cfg.extras["CLIP_C"] = float(args.clip_c)
    if args.local_epochs is not None:
        cfg.extras["LOCAL_EPOCHS"] = max(1, int(args.local_epochs))
    if args.disable_sampling_amplification:
        cfg.extras["USE_SAMPLING_AMPLIFICATION"] = False
    cfg.ensure_results_dir()
    return cfg


def model_factory(cfg):
    if cfg.dataset == "fashion":
        if cfg.model_name == "ldp_paper":
            return LDPFedFashionNet
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
            return LDPFedClient(
                cid=partition_id,
                model=model_factory(cfg)(),
                train_loader=train_loader,
                learning_rate=cfg.learning_rate,
                num_rounds=cfg.num_rounds,
                num_cycles=cfg.extras["NUM_CYCLES"],
                total_alpha=cfg.extras["TOTAL_ALPHA"],
                precision_rho=cfg.extras["PRECISION_RHO"],
                clip_c=cfg.extras["CLIP_C"],
                sampling_ratio=cfg.clients_per_round / cfg.num_clients,
                use_sampling_amplification=cfg.extras["USE_SAMPLING_AMPLIFICATION"],
                local_epochs=cfg.extras["LOCAL_EPOCHS"],
                seed=cfg.seed,
            ).to_client()

        return client_fn

    return _factory


def main():
    args = parse_args()
    cfg = apply_args(build_ldp_experiment_config(args.dataset, args.seed, profile=args.profile), args)
    if cfg.extras["NUM_CYCLES"] > cfg.num_rounds:
        raise ValueError("NUM_CYCLES cannot be greater than num_rounds.")
    cfg.validate(check_rounds=False, enforce_protocol=(cfg.profile == "baseline"))
    run_federated_simulation(
        cfg=cfg,
        model_factory=model_factory(cfg),
        client_fn_factory=client_fn_factory(cfg),
        persisted_fit_metrics=(
            "privacy_leakage",
            "distortion",
            "alpha_round",
            "alpha_per_param",
            "selected_layer_idx",
        ),
    )


if __name__ == "__main__":
    main()
