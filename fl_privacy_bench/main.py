from __future__ import annotations

import argparse
import sys
from typing import Callable

from fl_privacy_bench.baselines import cvb, dcs2, dp, ldp, ppan


RUNNERS: dict[str, Callable[[], None]] = {
    "cvb_fl": cvb.main,
    "dcs2_fl": dcs2.main,
    "dp_fl": dp.main,
    "ppan_fl": ppan.main,
    "ldp_fed": ldp.main,
}

ALIASES = {
    "cvb": "cvb_fl",
    "dcs2": "dcs2_fl",
    "dp": "dp_fl",
    "ppan": "ppan_fl",
    "ldp": "ldp_fed",
}


def parse_dispatch_args(argv: list[str] | None = None) -> tuple[str, list[str]]:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Run a FL Privacy Bench baseline from the unified package.",
        add_help=True,
    )
    parser.add_argument("--algo", required=True, choices=sorted(RUNNERS | ALIASES))
    if not argv or argv in (["-h"], ["--help"]):
        parser.parse_args(argv)

    dispatch_parser = argparse.ArgumentParser(add_help=False)
    dispatch_parser.add_argument("--algo", required=True, choices=sorted(RUNNERS | ALIASES))
    args, remaining = dispatch_parser.parse_known_args(argv)
    return ALIASES.get(args.algo, args.algo), remaining


def main(argv: list[str] | None = None) -> None:
    algo, remaining = parse_dispatch_args(argv)
    sys.argv = [f"fl_privacy_bench {algo}", *remaining]
    RUNNERS[algo]()


if __name__ == "__main__":
    main()
