# FL Privacy Bench

Federated learning baselines for privacy-preserving image classification on Fashion-MNIST and CIFAR-10.

## Baselines

- `dp_fl`: FedAvg with Gaussian differential privacy noise on model updates
- `ppan_fl`: FedAvg with a PPAN-style learned privacy mechanism
- `ldp_fed`: adaptive local differential privacy by layer/cycle
- `cvb_fl`: Convolutional Variational Bottleneck
- `dcs2_fl`: Defense by Concealing Sensitive Samples

All source code now lives under the unified `fl_privacy_bench/` package.

## Documentation

- [Baseline reference](docs/BASELINES.md)
- [Configuration reference](docs/CONFIG.md)
- [Code structure](docs/CODE_STRUCTURE.md)
- [Run guide](docs/HUONG_DAN_CHAY.md)

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run a smoke test from the repository root:

```bash
python -m fl_privacy_bench.main --algo cvb_fl --dataset fashion --seed 1 --num-rounds 1
python -m fl_privacy_bench.main --algo dcs2_fl --dataset fashion --seed 1 --num-rounds 1
python -m fl_privacy_bench.main --algo dp_fl --dataset fashion --seed 1 --num-rounds 1
python -m fl_privacy_bench.main --algo ppan_fl --dataset fashion --seed 1 --num-rounds 1 --privacy-weights 1
python -m fl_privacy_bench.main --algo ldp_fed --dataset fashion --profile baseline --seed 1 --num-rounds 1
```

Results are written under:

```text
results/<dataset>/seed_<seed>/
```

Metric files are prefixed by algorithm/profile, for example:

```text
results/fashion/seed_42/cvb_fl_centralized_accuracy.txt
results/fashion/seed_42/dp_fl_epsilon_0p1_privacy_leakage.txt
results/fashion/seed_42/ppan_fl_privacy_1_distortion.txt
```

For baseline-synchronized full runs, omit `--num-rounds`.
