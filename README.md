# KuaiLive-M3 Benchmark

This repository provides three independent benchmarks built on the
KuaiLive-M3 (KLM3) dataset. They share the lazy raw-data loader in
[`klm3.py`](klm3.py), but use separate preprocessing outputs and evaluation
protocols. Raw KLM3 data and generated artifacts are not included.

This repository also contains the official KuaiLive-M3 project website under
[`website/`](website/). The dataset is available on
[Hugging Face](https://huggingface.co/imgkkk2004/KuaiLive-M3/tree/main).

## 1. Benchmark Tasks

### Cross-domain recommendation (CDR)

The source domain is short video (`photo_id`) and the target domain is live
streaming authors (`author_id`). The target live interactions use a per-user
chronological 80/10/10 train/validation/test split; the source photo domain is
training-only. Both domains are iteratively 10-core filtered. Evaluation is
full-sort retrieval on the target domain with Recall/NDCG at 10, 20, and 40.

| Model group | Implemented models | Entry point |
| --- | --- | --- |
| Target-only baselines | BPR, NeuMF, NGCF, LightGCN, SASRec | `run_tune.sh` |
| Cross-domain models | CMF, BiTGCF, CLFM, CoNet, DCDCSR, DeepAPF, DisenCDR, DTCDR, EMCDR, MGCCDR, NATR, SSCDR | `run_klm3_cdr.sh`, `run_tune.sh` |

### Live-stream highlight prediction

Each example is a chronological sequence of segments from one live room. The
continuous target is `0.6 × normalized retention + 0.4 × normalized engagement
density`; the binary highlight label is the top 30% of segments within a room.
Rooms are split chronologically by start time (70/10/20). The `hierarchical`
model is offline and may use future segments; the other sequential models are
online/causal.

| Model | Setting | Main metrics |
| --- | --- | --- |
| `mlp` | Segment-wise reference | mAP, F1@50%, Kendall's tau, Spearman's rho |
| `gru` | Online recurrent | mAP, F1@50%, Kendall's tau, Spearman's rho |
| `causal` | Online causal Transformer | mAP, F1@50%, Kendall's tau, Spearman's rho |
| `hierarchical` | Offline local-plus-global Transformer | mAP, F1@50%, Kendall's tau, Spearman's rho |

Three input variants are available for every highlight model: `plain`
(segment embedding), `stats` (causal interaction statistics only), and `es`
(embedding plus an additive causal-statistics branch).

### Questionnaire-based recommendation

This author-level benchmark uses `live_interaction` as clicks and first-level
questionnaire answers as satisfaction feedback. `开播就推`, `适当推荐`, and
`打赏` map to positive feedback; `不想再看` maps to negative feedback. A
full-period author-level 5-core is applied before chronological leave-one-out
splitting. Ranking uses one positive plus 99 unseen sampled authors; MRR is
untruncated and HR/NDCG are reported at configurable cutoffs.

| Model group | Implemented models | History/input |
| --- | --- | --- |
| Click-only sequential baselines | Caser, GRU4Rec, SASRec, FMLPRec, HGN, NARM | Click history from `events.parquet` |
| Questionnaire-aware baselines | FeedRec, GRU4RecM, SASRecM, FMLPRecM, DFN, DMT | Timestamp-mixed CLICK/SATISFIED/DISSATISFIED tokens from `feedrec_events.parquet` |
| Proposed method | SAQRec | Base -> Propensity -> Satisfaction -> SAQRec teacher pipeline |

Questionnaire tokens are ordered immediately after their corresponding click.
Only click tokens are ranking targets, so an answer cannot leak into the
prediction of its own interaction.

## 2. Quick Start

Install the dependencies appropriate for the task. The vendored RecBole and
RecBole-CDR projects retain their upstream licenses and documentation.

```bash
pip install -r RecBole-CDR/requirements.txt
pip install -r SAQRec/requirements.txt
pip install pandas pyarrow scipy scikit-learn
```

Set `/data/klm3` below to the directory containing the KLM3 raw files.

### CDR

The end-to-end command preprocesses data when the RecBole `.inter` files are
absent, then trains and evaluates the requested model.

```bash
# End to end (default: CMF)
bash run_klm3_cdr.sh /data/klm3

# A specific cross-domain model
bash run_klm3_cdr.sh /data/klm3 EMCDR

# Rebuild only the CDR data; use a small sample for a smoke test
python preprocess_klm3.py --data_dir /data/klm3 --sample_ratio 0.01 --min_interactions 5

# Tune one target-only or CDR model after preprocessing
bash run_tune.sh SASRec 0
bash run_tune.sh MGCCDR 0
```

Generated CDR data is stored under `RecBole-CDR/dataset/`; CDR tuning logs are
stored under `log_tune/<MODEL>/`.

### Highlight prediction

The launcher preprocesses the raw data when needed and schedules a grid search
over all four models. Start with a small run before using the default sweep.

```bash
# One-GPU smoke test: one model grid configuration and 500 rooms
LR_LIST='1e-3' WD_LIST='1e-4' EPOCHS=5 JOBS_PER_GPU=1 VARIANTS='plain' \
  bash run_highlight.sh /data/klm3 0 --top_k 500

# Default stats-only and embedding+stats sweeps on two GPUs
bash run_highlight.sh /data/klm3 0,1

# All input variants
VARIANTS='plain stats es' bash run_highlight.sh /data/klm3 0,1

# Summarise completed highlight runs
python highlight/parse_highlight_results.py highlight/highlight_ckpt --best-only
```

Outputs are isolated in `highlight/highlight_data/` and
`highlight/highlight_ckpt/`. Completed sweep runs are resumed automatically.

### Questionnaire-based recommendation

First construct the independent author-level event table and inspect its audit
before launching full training. The preprocessing output does not modify the
CDR dataset.

```bash
# Build events.parquet, questionnaire_events.parquet, and audit.json
python SAQRec/preprocess.py --data_dir /data/klm3 --output_dir SAQRec/data/klm3

# Local synthetic end-to-end validation
python SAQRec/mock_validate.py

# Train one click-only baseline
./SAQRec/run_baseline.sh SASRec SAQRec/data/klm3 0

# Train one questionnaire-aware baseline; its mixed-token file is created automatically
./SAQRec/run_baseline.sh SASRecM SAQRec/data/klm3 0

# Tune a questionnaire-aware baseline
./SAQRec/run_tune_baseline.sh FeedRec 0

# Run one fixed SAQRec configuration (the shared teacher chain is trained first)
LR_VALUES='1e-3' WD_VALUES='1e-5' ./SAQRec/run_tune_baseline.sh SAQRec 0
```

Use comma-separated GPU IDs such as `0,1,2,3` for SAQRec distributed training.
`BATCH_SIZE` is per GPU; begin with a smaller per-GPU value when using multiple
devices. Each run writes a persistent log, `best.pt`, best-validation metrics,
and final test metrics under `SAQRec/outputs/`.

### Project website

The website is maintained in the same repository and is deployed as a static
GitHub Pages site. For local development:

```bash
cd website
npm ci
npm run dev
```

Before publishing the new repository, replace the placeholder benchmark URL in
`website/app/site-config.ts`. To verify the production site locally:

```bash
cd website
npm test
npm run build:github
```

After pushing to `main`, enable **GitHub Actions** as the Pages source in the
repository settings. The workflow in `.github/workflows/deploy-pages.yml`
builds and publishes `website/out/` whenever the website changes.

## Repository Layout

```text
klm3.py                    shared lazy KLM3 raw-data loader
preprocess_klm3.py         CDR preprocessing
run_klm3_cdr.sh            CDR train/evaluate entry point
run_tune.sh                CDR and target-only tuning entry point
highlight/                 highlight preprocessing, models, training, and sweep runner
SAQRec/                    questionnaire preprocessing, models, baselines, and runners
RecBole/                   vendored RecBole dependency
RecBole-CDR/               vendored RecBole-CDR dependency
website/                   KuaiLive-M3 project website and field documentation
.github/workflows/         GitHub Pages deployment workflow
```

For reproducibility, preserve the generated audit/log/result files with each
experiment. Do not commit raw data, generated parquet files, checkpoints, or
local logs; they are excluded by `.gitignore`.
