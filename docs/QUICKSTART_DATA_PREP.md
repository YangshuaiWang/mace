# Quickstart: foolproof data prep for sweeps

This guide shows exactly how to generate the four files needed for sweep/eval/MD scripts:

1. `new_domain_train_batches.pt`
2. `new_domain_valid_batches.pt` (optional)
3. `old_benchmark_retention_batches.pt`
4. `seed_structure.xyz` (or `.extxyz`)

## 0) Put files in expected locations

```bash
mkdir -p checkpoints data
```

Put your files here:

- Foundation checkpoint: `./checkpoints/foundation.model`
- New-domain extxyz:
  - either split files: `./data/new_domain_train.extxyz` (+ optional `./data/new_domain_valid.extxyz`)
  - or one file to split: `./data/new_domain.extxyz`
- Old-domain held-out benchmark extxyz: `./data/old_domain_retention.extxyz`

## 1) Build train/valid batch `.pt` files

### Option A: train + valid extxyz are already separate

```bash
PYTHONPATH=. python -m scripts.make_batches_pt \
  --train_extxyz ./data/new_domain_train.extxyz \
  --valid_extxyz ./data/new_domain_valid.extxyz \
  --out_train_pt ./data/new_domain_train_batches.pt \
  --out_valid_pt ./data/new_domain_valid_batches.pt \
  --foundation_checkpoint ./checkpoints/foundation.model \
  --seed 123
```

### Option B: one extxyz, deterministic split

```bash
PYTHONPATH=. python -m scripts.make_batches_pt \
  --extxyz ./data/new_domain.extxyz \
  --valid_fraction 0.1 \
  --out_train_pt ./data/new_domain_train_batches.pt \
  --out_valid_pt ./data/new_domain_valid_batches.pt \
  --foundation_checkpoint ./checkpoints/foundation.model \
  --seed 123
```

### Smoke-test sized data prep (fast)

```bash
PYTHONPATH=. python -m scripts.make_batches_pt \
  --extxyz ./data/new_domain.extxyz \
  --valid_fraction 0.1 \
  --out_train_pt ./data/new_domain_train_batches.pt \
  --out_valid_pt ./data/new_domain_valid_batches.pt \
  --foundation_checkpoint ./checkpoints/foundation.model \
  --max_train_structures 50 \
  --max_valid_structures 50 \
  --seed 123
```

## 2) Build retention benchmark `.pt`

```bash
PYTHONPATH=. python -m scripts.make_retention_batches_pt \
  --retention_extxyz ./data/old_domain_retention.extxyz \
  --out_pt ./data/old_benchmark_retention_batches.pt \
  --foundation_checkpoint ./checkpoints/foundation.model \
  --seed 123
```

Smoke-test sized retention set:

```bash
PYTHONPATH=. python -m scripts.make_retention_batches_pt \
  --retention_extxyz ./data/old_domain_retention.extxyz \
  --out_pt ./data/old_benchmark_retention_batches.pt \
  --foundation_checkpoint ./checkpoints/foundation.model \
  --max_structures 50 \
  --seed 123
```

## 3) Select MD seed structure

Default (median number of atoms):

```bash
PYTHONPATH=. python -m scripts.select_seed_structure \
  --source_extxyz ./data/new_domain_valid.extxyz \
  --out_xyz ./data/seed_structure.xyz \
  --mode median_natoms
```

Other modes:

```bash
# deterministic random
PYTHONPATH=. python -m scripts.select_seed_structure --source_extxyz ./data/new_domain_valid.extxyz --out_xyz ./data/seed_structure.xyz --mode random --seed 123

# lowest-energy (requires energy labels)
PYTHONPATH=. python -m scripts.select_seed_structure --source_extxyz ./data/new_domain_valid.extxyz --out_xyz ./data/seed_structure.extxyz --mode min_energy
```

## 4) Run sweeps/evals end-to-end

Update your config(s) so `data.train_batches_pt` and `data.valid_batches_pt` point at your generated `.pt` files, and `model.foundation_checkpoint` points at `./checkpoints/foundation.model`.

### Baseline sweep

```bash
PYTHONPATH=. python -m scripts.run_sweep --config configs/baseline.yaml --mode baseline --exp_name_prefix baseline --seeds 0,1,2 --run_dir_base runs
```

### E-FGGM sweep

```bash
PYTHONPATH=. python -m scripts.run_sweep --config configs/efggm_module.yaml --mode efggm --exp_name_prefix efggm --seeds 0,1,2 --run_dir_base runs
```

### Retention eval (example on one adapted model)

```bash
PYTHONPATH=. python -m scripts.eval_retention \
  --base_model ./checkpoints/foundation.model \
  --adapted_model runs/baseline_seed0/full_ft/final_model.pt \
  --retention_batches ./data/old_benchmark_retention_batches.pt \
  --out runs/baseline_seed0/full_ft/retention.json \
  --device cpu
```

### MD stability

```bash
PYTHONPATH=. python -m scripts.md_stability \
  --model runs/baseline_seed0/full_ft/final_model.pt \
  --atoms ./data/seed_structure.xyz \
  --steps 20 --temperature 300 --timestep_fs 0.5 \
  --seeds 0,1,2 \
  --out runs/baseline_seed0/full_ft/md_stability.json \
  --device cpu
```

### Aggregate + plot + table

```bash
PYTHONPATH=. python -m scripts.aggregate_runs --runs_glob 'runs/*' --out_csv_raw runs/summary.csv
PYTHONPATH=. python -m scripts.plot_results --input runs/summary.csv --out_dir runs/figures
PYTHONPATH=. python -m scripts.make_paper_table --input runs/summary.csv --output runs/table.tex
```

## 5) Smoke-test training budget

For a quick pipeline sanity check, set in your YAML config:

- `budget.max_steps: 50`

and use `--max_train_structures 50` / `--max_valid_structures 50` in the prep commands above.

Expected outputs under `runs/` include per-run folders with files such as:

- `config.yaml`
- `budget.json`
- `metrics.jsonl`
- `final_model.pt`
- and if you run post-eval: `retention.json`, `md_stability.json`
