from __future__ import annotations

import argparse
from typing import Any, Dict, Iterable, Optional

from .config import ExperimentConfig


def add_common_experiment_args(
    parser: argparse.ArgumentParser,
    *,
    default_dataset: str = "fashion",
    include_profile: bool = False,
    default_profile: str = "baseline",
) -> argparse.ArgumentParser:
    parser.add_argument("--dataset", choices=["fashion", "cifar"], default=default_dataset)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--clients-per-round", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--alpha-dirichlet", type=float, default=None)
    parser.add_argument("--skip-protocol-check", action="store_true")
    if include_profile:
        parser.add_argument("--profile", choices=["auto", "baseline", "paper"], default=default_profile)
    return parser


def collect_common_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "num_rounds": args.num_rounds,
        "num_clients": args.num_clients,
        "clients_per_round": args.clients_per_round,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "alpha_dirichlet": args.alpha_dirichlet,
    }


def validate_or_skip_protocol(cfg: ExperimentConfig, args: argparse.Namespace) -> None:
    if getattr(args, "skip_protocol_check", False):
        cfg.validate(check_rounds=False, enforce_protocol=False)
        return
    cfg.validate(check_rounds=(getattr(args, "num_rounds", None) is None))


def add_float_list_arg(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: Optional[Iterable[float]] = None,
    help_text: str = "",
) -> None:
    default_text = None if default is None else ",".join(str(x) for x in default)
    parser.add_argument(name, default=default_text, help=help_text)
