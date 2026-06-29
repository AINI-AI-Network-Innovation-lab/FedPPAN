# Huong Dan Chay FL Privacy Bench

Tai lieu nay mo ta cach chay cac baseline sau refactor. Source chinh nam trong package chung `fl_privacy_bench/`.

Tai lieu lien quan:

- `docs/BASELINES.md`: mo ta tung baseline
- `docs/CONFIG.md`: config chung va config rieng
- `docs/CODE_STRUCTURE.md`: cau truc source code va cach mo rong

## 1) Cai dat

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Neu khong cai duoc `quadprog`, DCS2 van co fallback closed-form cho truong hop constraint don.

## 2) Protocol mac dinh

Fashion-MNIST baseline:

- `NUM_CLIENTS=100`
- `CLIENTS_PER_ROUND=10`
- `NUM_ROUNDS=200`
- `BATCH_SIZE=16`
- `LEARNING_RATE=0.01`

CIFAR-10 baseline:

- `NUM_CLIENTS=50`
- `CLIENTS_PER_ROUND=10`
- `NUM_ROUNDS=500`
- `BATCH_SIZE=16`
- `LEARNING_RATE=0.03`

Entrypoint chung:

```bash
python -m fl_privacy_bench.main --algo <algo> [baseline flags]
```

`--algo` ho tro: `cvb_fl`, `dcs2_fl`, `dp_fl`, `ppan_fl`, `ldp_fed`.

Moi baseline deu co cac flag chung:

```bash
--dataset fashion|cifar
--seed 42
--num-rounds N
--num-clients N
--clients-per-round N
--batch-size N
--learning-rate LR
--alpha-dirichlet A
--skip-protocol-check
```

Mac dinh protocol check se bat loi neu baseline full-run bi lech. Khi smoke test voi `--num-rounds`, check so round duoc nới ra.

## 3) Chay smoke test

```bash
python -m fl_privacy_bench.main --algo cvb_fl --dataset fashion --seed 1 --num-rounds 1
python -m fl_privacy_bench.main --algo dcs2_fl --dataset fashion --seed 1 --num-rounds 1
python -m fl_privacy_bench.main --algo dp_fl --dataset fashion --seed 1 --num-rounds 1
python -m fl_privacy_bench.main --algo ppan_fl --dataset fashion --seed 1 --num-rounds 1 --privacy-weights 1
python -m fl_privacy_bench.main --algo ldp_fed --dataset fashion --profile baseline --seed 1 --num-rounds 1
```

## 4) Chay tung baseline

### DP

```bash
python -m fl_privacy_bench.main --algo dp_fl --dataset fashion --seed 42
python -m fl_privacy_bench.main --algo dp_fl --dataset cifar --seed 42
```

Flag rieng:

```bash
--epsilon 0.1
--delta 1e-5
--sensitivity 0.01
--max-grad-norm 1.0
--max-param-abs 10.0
```

### PPAN

```bash
python -m fl_privacy_bench.main --algo ppan_fl --dataset fashion --seed 42 --privacy-weights 500,200,100,10,1,0.1
python -m fl_privacy_bench.main --algo ppan_fl --dataset cifar --seed 42 --privacy-weights 1
```

Flag rieng:

```bash
--privacy-weights 500,200,100,10,1,0.1,0.01,0.001
--noise-scale 0.01
--max-privacy-weight 1.0
--max-privacy-loss 1.0
--distortion-weight 0.01
```

### LDP-Fed

```bash
python -m fl_privacy_bench.main --algo ldp_fed --dataset fashion --profile auto --seed 42
python -m fl_privacy_bench.main --algo ldp_fed --dataset fashion --profile baseline --seed 42
python -m fl_privacy_bench.main --algo ldp_fed --dataset cifar --profile baseline --seed 42
```

`--profile auto` tren Fashion-MNIST dung setup paper: `50` clients, `9` clients/round, `80` rounds, model `LDPFedFashionNet`. De dong bo voi baseline khac, dung `--profile baseline`.

Flag rieng:

```bash
--cycles 5
--alpha 1.0
--rho 4
--clip-c 0.1
--local-epochs 1
--disable-sampling-amplification
```

### CVB

```bash
python -m fl_privacy_bench.main --algo cvb_fl --dataset fashion --seed 42
python -m fl_privacy_bench.main --algo cvb_fl --dataset cifar --seed 42
```

Tham so CVB nam trong `fl_privacy_bench/baselines/cvb_config.py`.

### DCS2

```bash
python -m fl_privacy_bench.main --algo dcs2_fl --dataset fashion --seed 42
python -m fl_privacy_bench.main --algo dcs2_fl --dataset cifar --seed 42
```

Tham so DCS2 nam trong `fl_privacy_bench/baselines/dcs2_config.py`.

## 5) Output

Output chuan:

```text
results/<dataset>/seed_<seed>/
```

Moi baseline ghi vao cung folder dataset/seed. Ten file metric co prefix algorithm/profile de khong ghi de nhau, vi du:

```text
results/fashion/seed_42/cvb_fl_centralized_accuracy.txt
results/fashion/seed_42/dcs2_fl_privacy_leakage.txt
results/fashion/seed_42/dp_fl_epsilon_0p1_distortion.txt
results/fashion/seed_42/ppan_fl_privacy_1_centralized_loss.txt
results/fashion/seed_42/ldp_fed_paper_alpha_round.txt
```

Metric chung theo tung prefix:

- `<prefix>_centralized_accuracy.txt`
- `<prefix>_centralized_loss.txt`
- `<prefix>_privacy_leakage.txt`
- `<prefix>_distortion.txt`

Metric rieng:

- DCS2: `<prefix>_conceal_obj.txt`, `<prefix>_proj_applied_ratio.txt`
- LDP-Fed: `<prefix>_alpha_round.txt`, `<prefix>_alpha_per_param.txt`, `<prefix>_selected_layer_idx.txt`

## 6) Tong hop ket qua

```bash
python -m fl_privacy_bench.tools.summarize_baselines --algos cvb_fl dcs2_fl dp_fl ppan_fl ldp_fed --datasets fashion cifar
```
