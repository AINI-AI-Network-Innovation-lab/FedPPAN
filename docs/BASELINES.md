# Baseline Reference

Tai lieu nay mo ta cac baseline hien co trong `fl_privacy_bench/`, file code chinh, tham so rieng, metric ghi ra, va lenh chay mau.

## Tong Quan

| Algo | Muc dich | Runner | Client | Config | Model |
| --- | --- | --- | --- | --- | --- |
| `cvb_fl` | Convolutional Variational Bottleneck | `fl_privacy_bench/baselines/cvb.py` | `fl_privacy_bench/baselines/cvb_client.py` | `fl_privacy_bench/baselines/cvb_config.py` | `fl_privacy_bench/models/cvb.py` |
| `dcs2_fl` | Defense by Concealing Sensitive Samples | `fl_privacy_bench/baselines/dcs2.py` | `fl_privacy_bench/baselines/dcs2_client.py` | `fl_privacy_bench/baselines/dcs2_config.py` | `fl_privacy_bench/models/dcs2.py` |
| `dp_fl` | Gaussian DP noise tren model update | `fl_privacy_bench/baselines/dp.py` | `fl_privacy_bench/baselines/dp_client.py` | `fl_privacy_bench/baselines/dp_config.py` | `fl_privacy_bench/models/mnist_model.py` |
| `ppan_fl` | PPAN-style learned privacy mechanism | `fl_privacy_bench/baselines/ppan.py` | `fl_privacy_bench/baselines/ppan_client.py` | `fl_privacy_bench/baselines/ppan_config.py` | `fl_privacy_bench/models/privacy_mechanism.py` |
| `ldp_fed` | Adaptive local DP theo layer/cycle | `fl_privacy_bench/baselines/ldp.py` | `fl_privacy_bench/baselines/ldp_client.py` | `fl_privacy_bench/baselines/ldp_config.py` | `fl_privacy_bench/models/mnist_model.py` |

Tat ca baseline dung chung runtime:

- `fl_privacy_bench/core/config.py`: protocol va `ExperimentConfig`
- `fl_privacy_bench/core/data.py`: partition va dataloader
- `fl_privacy_bench/core/simulation.py`: Flower simulation wiring
- `fl_privacy_bench/core/evaluation.py`: centralized accuracy/loss
- `fl_privacy_bench/core/strategy.py`: FedAvg + metric persistence

## Metric Chung

Tat ca baseline deu ghi:

- `<prefix>_centralized_accuracy.txt`
- `<prefix>_centralized_loss.txt`
- `<prefix>_privacy_leakage.txt`
- `<prefix>_distortion.txt`

Output folder chung:

```text
results/<dataset>/seed_<seed>/
```

`prefix` la ten algorithm, cong them profile neu co. Vi du:

```text
cvb_fl
dcs2_fl
dp_fl_epsilon_0p1
ppan_fl_privacy_1
ldp_fed_paper
```

## `cvb_fl`

CVB chen bottleneck convolutional vao feature map som va toi uu loss:

```text
cross_entropy + CVB_BETA * kl_loss
```

Config rieng:

| Ten | Mac dinh | Y nghia |
| --- | --- | --- |
| `CVB_POSITION` | `1` | Vi tri y tuong cua CVB trong network |
| `CVB_KERNEL_SIZE` | `5` | Kernel size cua encoder/decoder bottleneck |
| `CVB_SCALE` | `0.5` | Ti le latent channels |
| `CVB_BETA` | `0.1` | Trong so KL loss |

Lenh chay:

```bash
python -m fl_privacy_bench.main --algo cvb_fl --dataset fashion --seed 42
python -m fl_privacy_bench.main --algo cvb_fl --dataset cifar --seed 42
```

Metric rieng: khong co; chi dung metric chung.

## `dcs2_fl`

DCS2 toi uu gradient bao ve dua tren sensitive samples/proxy samples, co mixup va projection.

Config rieng:

| Ten | Mac dinh | Y nghia |
| --- | --- | --- |
| `DCS2_LAMBDA_G` | `0.7` | He so chinh cho conceal/protected gradient |
| `DCS2_LAMBDA_X` | `0.01` | He so cho image-space distance |
| `DCS2_LAMBDA_Z` | `0.01` | He so cho feature/logit term |
| `DCS2_EPSILON` | `0.01` | Margin trong objective |
| `DCS2_DCS_ITER` | `1000` | So buoc toi uu proxy/adversarial samples |
| `DCS2_DCS_LR` | `0.1` | Learning rate cho DCS optimization |
| `DCS2_NUM_SEN` | `16` | So sensitive samples moi batch |
| `DCS2_PER_ADV` | `1` | So proxy/adversarial sample tren moi sensitive sample |
| `DCS2_XSIM_THR` | `150.0` | Early-stop threshold cho image distance |
| `DCS2_PROJECT` | `True` | Bat/tat projection khi gradient conflict |
| `DCS2_MIXUP` | `True` | Bat/tat mixup voi sensitive image |
| `DCS2_STARTPOINT` | `"none"` | Khoi tao proxy image; `"noise"` dung noise |
| `DCS2_EARLY_STOP` | `True` | Bat/tat early stop |

Lenh chay:

```bash
python -m fl_privacy_bench.main --algo dcs2_fl --dataset fashion --seed 42
python -m fl_privacy_bench.main --algo dcs2_fl --dataset cifar --seed 42
```

Metric rieng:

- `<prefix>_conceal_obj.txt`
- `<prefix>_proj_applied_ratio.txt`

## `dp_fl`

DP baseline train local model binh thuong, clip gradient/param, roi them Gaussian noise vao model parameters truoc khi tra ve server.

CLI/config rieng:

| Flag | Mac dinh | Y nghia |
| --- | --- | --- |
| `--epsilon` | `0.1` | DP epsilon |
| `--delta` | `1e-5` | DP delta |
| `--sensitivity` | `0.01` | Sensitivity dung tinh sigma |
| `--max-grad-norm` | `1.0` | Clip gradient norm |
| `--max-param-abs` | `10.0` | Clamp absolute parameter value |

Noise sigma:

```text
sigma = sensitivity * sqrt(2 * log(1.25 / delta)) / epsilon
```

Lenh chay:

```bash
python -m fl_privacy_bench.main --algo dp_fl --dataset fashion --seed 42 --epsilon 0.1
python -m fl_privacy_bench.main --algo dp_fl --dataset cifar --seed 42 --epsilon 0.1
```

Metric rieng: khong co; chi dung metric chung.

## `ppan_fl`

PPAN baseline train model chinh cung learned privacy mechanism gom encoder/adversary. Co the chay nhieu privacy weights trong mot command.

CLI/config rieng:

| Flag | Mac dinh | Y nghia |
| --- | --- | --- |
| `--privacy-weights` | `500,200,100,10,1,0.1,0.01,0.001` | Danh sach privacy weights, moi gia tri chay mot simulation |
| `--noise-scale` | `0.01` | Noise trong privacy encoder khi training |
| `--max-privacy-weight` | `1.0` | Cap tren cho privacy weight hieu dung |
| `--max-privacy-loss` | `1.0` | Clamp privacy reconstruction loss |
| `--distortion-weight` | `0.01` | Trong so distortion loss |
| `--max-grad-norm` | `1.0` | Clip gradient norm |
| `--max-param-abs` | `10.0` | Clamp absolute parameter value |

Lenh chay:

```bash
python -m fl_privacy_bench.main --algo ppan_fl --dataset fashion --seed 42 --privacy-weights 1
python -m fl_privacy_bench.main --algo ppan_fl --dataset fashion --seed 42 --privacy-weights 500,200,100,10,1,0.1
```

Metric rieng: khong co; chi dung metric chung.

## `ldp_fed`

LDP-Fed baseline train local model, lay delta giua local/global params, roi perturb theo ordinal CLDP tren layer duoc schedule theo cycle.

Profile:

| Profile | Dataset | Protocol |
| --- | --- | --- |
| `auto` | Fashion-MNIST | Tu dong dung `paper` |
| `paper` | Fashion-MNIST | `50` clients, `9` clients/round, `80` rounds, model `LDPFedFashionNet` |
| `baseline` | Fashion-MNIST/CIFAR-10 | Dong bo voi protocol chung |

CLI/config rieng:

| Flag | Mac dinh | Y nghia |
| --- | --- | --- |
| `--profile` | `auto` | `auto`, `baseline`, hoac `paper` |
| `--cycles` | `5` | So cycle schedule layer |
| `--alpha` | `1.0` | Tong privacy budget alpha |
| `--rho` | `4` | Decimal precision rho cho ordinal perturbation |
| `--clip-c` | `0.1` | Clip delta truoc perturbation |
| `--local-epochs` | `1` | Local epochs moi client |
| `--disable-sampling-amplification` | off | Tat sampling amplification |

Lenh chay:

```bash
python -m fl_privacy_bench.main --algo ldp_fed --dataset fashion --profile auto --seed 42
python -m fl_privacy_bench.main --algo ldp_fed --dataset fashion --profile baseline --seed 42
python -m fl_privacy_bench.main --algo ldp_fed --dataset cifar --profile baseline --seed 42
```

Metric rieng:

- `<prefix>_alpha_round.txt`
- `<prefix>_alpha_per_param.txt`
- `<prefix>_selected_layer_idx.txt`
