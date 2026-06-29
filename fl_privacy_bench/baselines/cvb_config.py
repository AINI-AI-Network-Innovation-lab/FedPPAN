from fl_privacy_bench.core.config import DEVICE, build_experiment_config

LEAKAGE_METRIC_MODE = "legacy"
NUM_REPEATS = 3

CVB_POSITION = 1
CVB_KERNEL_SIZE = 5
CVB_SCALE = 0.5
CVB_BETA = 0.1


def get_experiment_config(dataset: str, seed: int) -> dict:
    cfg = build_experiment_config("cvb_fl", dataset, seed)
    return cfg.to_legacy_dict()
