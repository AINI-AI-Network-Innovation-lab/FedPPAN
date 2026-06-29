from fl_privacy_bench.core.config import DEVICE, build_experiment_config

LEAKAGE_METRIC_MODE = "legacy"
NUM_REPEATS = 3

DCS2_LAMBDA_G = 0.7
DCS2_LAMBDA_X = 0.01
DCS2_LAMBDA_Z = 0.01
DCS2_EPSILON = 0.01
DCS2_DCS_ITER = 1000
DCS2_DCS_LR = 0.1
DCS2_NUM_SEN = 16
DCS2_PER_ADV = 1
DCS2_LAMBDA_Y = DCS2_LAMBDA_G
DCS2_XSIM_THR = 150.0
DCS2_PROJECT = True
DCS2_MIXUP = True
DCS2_STARTPOINT = "none"
DCS2_EARLY_STOP = True

DCS2_SYNTH_STEPS = DCS2_DCS_ITER
DCS2_SYNTH_LR = DCS2_DCS_LR
DCS2_INIT_MODE = DCS2_STARTPOINT

DCS2_ENABLE_AMP = False
DCS2_ENABLE_COMPILE = False
DCS2_LOG_TIMING = True


def get_experiment_config(dataset: str, seed: int) -> dict:
    cfg = build_experiment_config("dcs2_fl", dataset, seed)
    return cfg.to_legacy_dict()
