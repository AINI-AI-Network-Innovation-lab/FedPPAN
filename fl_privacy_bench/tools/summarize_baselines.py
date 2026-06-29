import argparse
import os
from glob import glob

import numpy as np


def parse_metric_file(path: str) -> np.ndarray:
    values = []
    latest_by_round = {}
    if not os.path.exists(path):
        return np.array(values, dtype=np.float64)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            round_idx = None
            value = None
            parts = line.strip().split(",")
            for p in parts:
                if p.startswith("round="):
                    try:
                        round_idx = int(p.split("=", 1)[1])
                    except ValueError:
                        pass
                if p.startswith("value="):
                    try:
                        value = float(p.split("=", 1)[1])
                    except ValueError:
                        pass
            if value is None:
                continue
            if round_idx is None:
                values.append(value)
            else:
                latest_by_round[round_idx] = value
    if latest_by_round:
        return np.array([latest_by_round[k] for k in sorted(latest_by_round)], dtype=np.float64)
    return np.array(values, dtype=np.float64)


def metric_path(seed_dir: str, prefix: str, metric_name: str) -> str:
    filename = f"{prefix}_{metric_name}.txt" if prefix else f"{metric_name}.txt"
    return os.path.join(seed_dir, filename)


def discover_runs(algo: str, dataset: str) -> list[tuple[str, str]]:
    runs = []

    # New shared layout: results/<dataset>/seed_<seed>/<algo>[_profile]_<metric>.txt
    for seed_dir in glob(os.path.join("results", dataset, "seed_*")):
        suffix = "_centralized_accuracy.txt"
        for acc_path in glob(os.path.join(seed_dir, f"{algo}*{suffix}")):
            filename = os.path.basename(acc_path)
            runs.append((seed_dir, filename[: -len(suffix)]))

    # Legacy layout kept readable for existing experiments.
    for seed_dir in glob(os.path.join("results", algo, dataset, "seed_*")):
        if os.path.exists(metric_path(seed_dir, "", "centralized_accuracy")):
            runs.append((seed_dir, ""))
    for seed_dir in glob(os.path.join("results", algo, dataset, "*", "seed_*")):
        if os.path.exists(metric_path(seed_dir, "", "centralized_accuracy")):
            runs.append((seed_dir, ""))

    return sorted(set(runs))


def summarize_algo_dataset(algo: str, dataset: str) -> dict:
    runs = discover_runs(algo, dataset)
    if not runs:
        return {}

    accs = []
    losses = []
    leaks = []
    valid_runs = 0
    for seed_dir, prefix in runs:
        acc = parse_metric_file(metric_path(seed_dir, prefix, "centralized_accuracy"))
        loss = parse_metric_file(metric_path(seed_dir, prefix, "centralized_loss"))
        leak = parse_metric_file(metric_path(seed_dir, prefix, "privacy_leakage"))
        if acc.size == 0 and loss.size == 0 and leak.size == 0:
            continue
        valid_runs += 1
        if acc.size > 0:
            accs.append(acc[-1])
        if loss.size > 0:
            losses.append(np.mean(loss[-15:]))
        if leak.size > 0:
            leaks.append(np.mean(leak))

    def _mean_std(x):
        if len(x) == 0:
            return (np.nan, np.nan)
        return (float(np.mean(x)), float(np.std(x)))

    acc_m, acc_s = _mean_std(accs)
    loss_m, loss_s = _mean_std(losses)
    leak_m, leak_s = _mean_std(leaks)
    return {
        "algo": algo,
        "dataset": dataset,
        "num_seeds": valid_runs,
        "accuracy_mean": acc_m,
        "accuracy_std": acc_s,
        "loss_last15_mean": loss_m,
        "loss_last15_std": loss_s,
        "leakage_mean": leak_m,
        "leakage_std": leak_s,
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize CVB/DCS2 baseline results.")
    parser.add_argument("--algos", nargs="+", default=["cvb_fl", "dcs2_fl"])
    parser.add_argument("--datasets", nargs="+", default=["fashion", "cifar"])
    args = parser.parse_args()

    print("algo,dataset,num_seeds,acc_mean,acc_std,loss15_mean,loss15_std,leak_mean,leak_std")
    for algo in args.algos:
        for dataset in args.datasets:
            row = summarize_algo_dataset(algo, dataset)
            if not row:
                continue
            print(
                f"{row['algo']},{row['dataset']},{row['num_seeds']},"
                f"{row['accuracy_mean']:.6f},{row['accuracy_std']:.6f},"
                f"{row['loss_last15_mean']:.6f},{row['loss_last15_std']:.6f},"
                f"{row['leakage_mean']:.6f},{row['leakage_std']:.6f}"
            )


if __name__ == "__main__":
    main()
