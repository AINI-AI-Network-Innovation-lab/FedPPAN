# Configuration Reference

Tai lieu nay mo ta config chung va config rieng cho cac baseline trong `fl_privacy_bench`.

## Source Config

| Phan | File |
| --- | --- |
| Protocol chung | `fl_privacy_bench/core/config.py` |
| CLI flag chung | `fl_privacy_bench/core/cli.py` |
| CVB config | `fl_privacy_bench/baselines/cvb_config.py` |
| DCS2 config | `fl_privacy_bench/baselines/dcs2_config.py` |
| DP config | `fl_privacy_bench/baselines/dp_config.py` |
| PPAN config | `fl_privacy_bench/baselines/ppan_config.py` |
| LDP-Fed config | `fl_privacy_bench/baselines/ldp_config.py` |

## Protocol Chung

`BASELINE_PROTOCOLS` trong `fl_privacy_bench/core/config.py` dinh nghia protocol mac dinh.

Fashion-MNIST:

| Key | Value |
| --- | --- |
| `num_clients` | `100` |
| `clients_per_round` | `10` |
| `num_rounds` | `200` |
| `batch_size` | `16` |
| `learning_rate` | `0.01` |
| `num_channel` | `1` |
| `num_classes` | `10` |
| `model_name` | `net` |

CIFAR-10:

| Key | Value |
| --- | --- |
| `num_clients` | `50` |
| `clients_per_round` | `10` |
| `num_rounds` | `500` |
| `batch_size` | `16` |
| `learning_rate` | `0.03` |
| `num_channel` | `3` |
| `num_classes` | `10` |
| `model_name` | `cnn4` |

## CLI Flag Chung

Moi baseline runner dung `add_common_experiment_args`.

| Flag | Type | Mac dinh | Anh huong |
| --- | --- | --- | --- |
| `--dataset` | choice | `fashion` | Chon `fashion` hoac `cifar` |
| `--seed` | int | `42` | Seed cho partition/sampling/model init |
| `--num-rounds` | int | protocol | Override so FL rounds |
| `--num-clients` | int | protocol | Override tong clients |
| `--clients-per-round` | int | protocol | Override sampled clients/round |
| `--batch-size` | int | protocol | Override local batch size |
| `--learning-rate` | float | protocol | Override local optimizer LR |
| `--alpha-dirichlet` | float | `0.5` | Dirichlet alpha cho non-IID split |
| `--skip-protocol-check` | bool | off | Cho phep chay config lech protocol full-run |

Khi chay smoke test voi `--num-rounds`, protocol check bo qua so round nhung van check cac key con lai. Neu muon override nhieu hon, them `--skip-protocol-check`.

## ExperimentConfig

`ExperimentConfig` la dataclass trung tam. Cac field quan trong:

| Field | Y nghia |
| --- | --- |
| `algorithm` | Ten algo: `cvb_fl`, `dcs2_fl`, `dp_fl`, `ppan_fl`, `ldp_fed` |
| `dataset` | `fashion` hoac `cifar` |
| `profile` | Sub-run/profile, vi du `epsilon_0.1`, `privacy_1`, `paper` |
| `num_clients` | Tong clients |
| `clients_per_round` | So clients sampled moi round |
| `num_rounds` | So FL rounds |
| `batch_size` | Local batch size |
| `learning_rate` | Local learning rate |
| `alpha_dirichlet` | Non-IID partition alpha |
| `extras` | Dict chua config rieng cua baseline |

Derived fields:

| Property | Y nghia |
| --- | --- |
| `fraction_fit` | `clients_per_round / num_clients` |
| `fraction_evaluate` | `clients_per_round / num_clients` |
| `metric_prefix` | Prefix ten file metric, gom algorithm/profile |
| `results_dir` | `results/<dataset>/seed_<seed>/` |

## Output Naming

Ket qua moi run ghi vao:

```text
results/<dataset>/seed_<seed>/
```

Ten file:

```text
<metric_prefix>_<metric_name>.txt
```

Vi du:

```text
cvb_fl_centralized_accuracy.txt
dp_fl_epsilon_0p1_privacy_leakage.txt
ppan_fl_privacy_1_distortion.txt
ldp_fed_paper_alpha_round.txt
```

## Config Rieng Theo Baseline

### CVB

File: `fl_privacy_bench/baselines/cvb_config.py`

| Key | Value |
| --- | --- |
| `CVB_POSITION` | `1` |
| `CVB_KERNEL_SIZE` | `5` |
| `CVB_SCALE` | `0.5` |
| `CVB_BETA` | `0.1` |

### DCS2

File: `fl_privacy_bench/baselines/dcs2_config.py`

| Key | Value |
| --- | --- |
| `DCS2_LAMBDA_G` | `0.7` |
| `DCS2_LAMBDA_X` | `0.01` |
| `DCS2_LAMBDA_Z` | `0.01` |
| `DCS2_EPSILON` | `0.01` |
| `DCS2_DCS_ITER` | `1000` |
| `DCS2_DCS_LR` | `0.1` |
| `DCS2_NUM_SEN` | `16` |
| `DCS2_PER_ADV` | `1` |
| `DCS2_XSIM_THR` | `150.0` |
| `DCS2_PROJECT` | `True` |
| `DCS2_MIXUP` | `True` |
| `DCS2_STARTPOINT` | `none` |
| `DCS2_EARLY_STOP` | `True` |
| `DCS2_ENABLE_AMP` | `False` |
| `DCS2_ENABLE_COMPILE` | `False` |

### DP

File: `fl_privacy_bench/baselines/dp_config.py`

| Key | Value |
| --- | --- |
| `epsilon` | `0.1` |
| `delta` | `1e-5` |
| `sensitivity` | `0.01` |
| `MAX_PARAM_ABS` | `10.0` |
| `MAX_GRAD_NORM` | `1.0` |

Runtime flags can override these values.

### PPAN

File: `fl_privacy_bench/baselines/ppan_config.py`

| Key | Value |
| --- | --- |
| `PRIVACY_WEIGHT` | `[500, 200, 100, 10, 1, 0.1, 0.01, 0.001]` |
| `NOISE_SCALE` | `0.01` |
| `MAX_PRIVACY_WEIGHT` | `1.0` |
| `MAX_PRIVACY_LOSS` | `1.0` |
| `DISTORTION_WEIGHT` | `0.01` |
| `MAX_GRAD_NORM` | `1.0` |
| `MAX_PARAM_ABS` | `10.0` |

Runtime flags can override these values.

### LDP-Fed

File: `fl_privacy_bench/baselines/ldp_config.py`

| Key | Value |
| --- | --- |
| `TOTAL_ALPHA` | `1.0` |
| `PRECISION_RHO` | `4` |
| `CLIP_C` | `0.1` |
| `NUM_CYCLES` | `5` |
| `ALPHA_DIRICHLET` | `0.5` |
| `LOCAL_EPOCHS` | `1` |
| `USE_SAMPLING_AMPLIFICATION` | `True` |

Fashion-MNIST `paper` profile overrides protocol:

| Key | Value |
| --- | --- |
| `num_clients` | `50` |
| `clients_per_round` | `9` |
| `num_rounds` | `80` |
| `model_name` | `ldp_paper` |

## Cach Them Config Moi

1. Them constant vao `fl_privacy_bench/baselines/<algo>_config.py`.
2. Them CLI flag vao runner `fl_privacy_bench/baselines/<algo>.py` neu can runtime override.
3. Dua gia tri vao `extras` khi build `ExperimentConfig`.
4. Doc gia tri trong client factory hoac client class.
5. Cap nhat docs trong `docs/BASELINES.md` va file nay.
