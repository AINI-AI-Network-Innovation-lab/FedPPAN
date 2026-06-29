from fl_privacy_bench.core.config import DEVICE, ExperimentConfig, build_experiment_config

TOTAL_ALPHA = 1.0
PRECISION_RHO = 4
CLIP_C = 0.1
NUM_CYCLES = 5
ALPHA_DIRICHLET = 0.5
LOCAL_EPOCHS = 1
USE_SAMPLING_AMPLIFICATION = True


def _paper_fashion_config(seed: int) -> ExperimentConfig:
    cfg = ExperimentConfig(
        algorithm="ldp_fed",
        dataset="fashion",
        profile="paper",
        seed=seed,
        num_clients=50,
        clients_per_round=9,
        num_rounds=80,
        batch_size=16,
        learning_rate=0.01,
        num_channel=1,
        num_classes=10,
        model_name="ldp_paper",
        alpha_dirichlet=ALPHA_DIRICHLET,
    )
    cfg.ensure_results_dir()
    return cfg


def build_ldp_experiment_config(dataset: str, seed: int, profile: str = "auto") -> ExperimentConfig:
    dataset = dataset.lower()
    profile = profile.lower()
    if dataset == "fashion" and profile in ("auto", "paper"):
        cfg = _paper_fashion_config(seed)
    elif dataset == "cifar" and profile == "paper":
        raise ValueError("Paper profile is only available for Fashion-MNIST in this repository.")
    else:
        cfg = build_experiment_config("ldp_fed", dataset, seed, profile="baseline")

    cfg.extras.update(
        {
            "TOTAL_ALPHA": TOTAL_ALPHA,
            "PRECISION_RHO": PRECISION_RHO,
            "CLIP_C": CLIP_C,
            "NUM_CYCLES": NUM_CYCLES,
            "LOCAL_EPOCHS": LOCAL_EPOCHS,
            "USE_SAMPLING_AMPLIFICATION": USE_SAMPLING_AMPLIFICATION,
        }
    )
    return cfg


def get_experiment_config(dataset: str, seed: int, profile: str = "auto") -> dict:
    cfg = build_ldp_experiment_config(dataset, seed, profile)
    return cfg.to_legacy_dict()
