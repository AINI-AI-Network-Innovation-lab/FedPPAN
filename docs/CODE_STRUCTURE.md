# Code Structure

Source code chinh nam trong package:

```text
fl_privacy_bench/
  main.py
  baselines/
  core/
  data/
  metrics/
  models/
  server/
  tools/
```

## Package Map

| Folder | Vai tro |
| --- | --- |
| `fl_privacy_bench/main.py` | Dispatcher CLI: `--algo <algo>` |
| `fl_privacy_bench/baselines/` | Runner, client, config rieng tung baseline |
| `fl_privacy_bench/core/` | Runtime chung: config, CLI, data, evaluation, strategy, simulation |
| `fl_privacy_bench/data/` | Dataset partition va dataloader helpers |
| `fl_privacy_bench/metrics/` | Privacy leakage/distortion estimators |
| `fl_privacy_bench/models/` | Model architectures va privacy mechanism modules |
| `fl_privacy_bench/server/` | Flower client manager va metric aggregation |
| `fl_privacy_bench/tools/` | Tools nhu result summarizer |

## Runtime Flow

1. User goi:

```bash
python -m fl_privacy_bench.main --algo cvb_fl --dataset fashion --seed 42
```

2. `fl_privacy_bench/main.py` dispatch sang runner trong `fl_privacy_bench/baselines/`.
3. Runner parse CLI, build `ExperimentConfig`, validate protocol.
4. Runner truyen `model_factory` va `client_fn_factory` vao `run_federated_simulation`.
5. `core/simulation.py` tao dataset partition, centralized testset, strategy, va Flower simulation.
6. Client train local model va tra ve parameters + metrics.
7. `core/strategy.py` aggregate FedAvg va ghi fit metrics.
8. `core/evaluation.py` evaluate centralized va ghi accuracy/loss.

## Baseline Module Pattern

Moi baseline nen co 3 file:

```text
fl_privacy_bench/baselines/<algo>.py
fl_privacy_bench/baselines/<algo>_client.py
fl_privacy_bench/baselines/<algo>_config.py
```

Trong runner `<algo>.py` nen co:

- `parse_args()`
- `model_factory(cfg)`
- `client_fn_factory(cfg)`
- `main()`

Trong client `<algo>_client.py` nen co class ke thua `fl.client.NumPyClient`.

Trong config `<algo>_config.py` nen chi chua default constants va optional helper config builder.

## Them Baseline Moi

1. Tao `fl_privacy_bench/baselines/my_algo_config.py`.
2. Tao `fl_privacy_bench/baselines/my_algo_client.py`.
3. Tao `fl_privacy_bench/baselines/my_algo.py` theo pattern runner hien co.
4. Them algo vao `RUNNERS` va `ALIASES` trong `fl_privacy_bench/main.py`.
5. Neu can model moi, them vao `fl_privacy_bench/models/`.
6. Neu can metric moi, client return metric trong `fit`, va runner them ten metric vao `persisted_fit_metrics`.
7. Cap nhat `docs/BASELINES.md` va `docs/CONFIG.md`.

## Naming Convention

- Algorithm ids: lowercase snake_case, vi du `cvb_fl`, `dcs2_fl`, `dp_fl`.
- Metric files: `<metric_prefix>_<metric_name>.txt`.
- Profile names nen ngan va deterministic, vi du `epsilon_0.1`, `privacy_1`, `paper`.
- New shared behavior nen vao `core/`; algorithm-specific behavior nen o `baselines/`.

## Debug Checklist

Lenh kiem tra nhanh:

```bash
python -c "import ast, pathlib; [ast.parse(p.read_text()) for p in pathlib.Path('fl_privacy_bench').rglob('*.py')]"
python -c "import fl_privacy_bench.main"
python -m fl_privacy_bench.main --algo cvb_fl --help
python -m fl_privacy_bench.tools.summarize_baselines --algos cvb_fl dcs2_fl --datasets fashion cifar
```

Smoke test training:

```bash
python -m fl_privacy_bench.main --algo cvb_fl --dataset fashion --seed 1 --num-rounds 1
```
