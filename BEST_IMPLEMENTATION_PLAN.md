# BEST Model Implementation Plan

**Model**: Bilaterally Expanding Subtrace Tree (BEST)  
**Source**: https://github.com/lmu-dbs/BEST  
**Paper**: _"BEST: Bilaterally Expanding Subtrace Tree for Event Sequence Prediction"_ — BPM 2025

---

## Overview

BEST is a probabilistic, tree-based model for Predictive Process Monitoring. Unlike the existing Transformer and GNN models, it requires no neural network training — it builds a hierarchical tree of subtrace patterns from the training log and performs prediction by traversal.

**Supported tasks**:
- **NAP** — Next Activity Prediction → maps to existing `next_activity` task
- **RTP** — Remaining Trace Prediction → new `remaining_trace` task (predicts the full future activity sequence)

**Not supported**:
- Time-based tasks (`event_time`, `remaining_time`) — BEST is activity-sequence only
- Gradient/SHAP/LIME explainability — uses native pattern-based explanations instead

---

## Phase 1 — Dependencies & Module Scaffold ✅ COMPLETE

> **Status**: Done — branch `feature/best-phase-1` · closes issues #1 #2 #3
> **Goal**: Get the library in place and create the Python module structure.

### 1.1 Add BEST to `requirements.txt` ✅

```
git+https://github.com/lmu-dbs/BEST.git
```

> ℹ️ `#subdirectory=src` is not needed — the `pyproject.toml` is at repo root. Library installed and verified.
> Python >3.12 requirement confirmed in `.venv`.

### 1.2 Create `best/` module ✅

| File | Purpose |
|---|---|
| `best/__init__.py` | Empty module marker ✅ |
| `best/predictor.py` | `BESTRunner` adapter class wrapping `BESTPredictor` + `SequenceData` ✅ |

**`BESTRunner` responsibilities**:
- Accept a pandas DataFrame with standard columns (`CaseID`, `Activity`, `Timestamp`)
- Create `SequenceData` objects (train + test) mapped to BEST's column identifier names
- Execute the full fit/predict lifecycle
- Compute NAP metrics (accuracy, balanced accuracy) and RTP metrics (NDLS similarity)
- Save predictions CSV + pickle model artifact

### 1.3 Create `explainability/best/` module ✅

| File | Purpose |
|---|---|
| `explainability/best/__init__.py` | Exports `BESTExplainer` ✅ |
| `explainability/best/best_explainer.py` | `BESTExplainer` — reads `choice_tracker_nap`/`choice_tracker_rtp` from the fitted model after prediction; produces pattern probability distributions, pattern length charts, top-pattern frequency tables ✅ |

**Outputs** (saved to `artifacts/explainability/`):
- `pattern_probabilities.png` — histogram of chosen pattern conditional probabilities
- `pattern_lengths.png` — bar chart of chosen pattern sizes
- `rpif_distances.png` — distribution of RPIF distance scores
- `top_patterns.csv` — most frequently chosen patterns with stats

---

## Phase 2 — Pipeline Functions

> **Goal**: Wire BEST into `ppm_pipeline.py` using the same conventions as Transformer and GNN functions.

### 2.1 Add to `ppm_pipeline.py`

Three new top-level functions:

#### `run_best_nap_prediction(dataset_path, output_dir, config, split)`
```
Load CSV → detect_and_standardize_columns()
         → SequenceData(data=df, case_identifier="CaseID", activity_identifier="Activity", timestamp_identifier="Timestamp")
         → train_test_split(train_pct=1 - test_size)
         → BESTPredictor(max_pattern_size, process_stage_width_percentage, min_freq, prune_func=None)
         → load_data(train, test)
         → prepare_train()
         → fit()
         → prepare_test(act_encoder=train.act_encoder, filter_sequences=config['filter_sequences'])
         → predict(task='nap', eval_pattern_size, break_buffer, filter_tokens, ncores)
         → NAPEvaluator → accuracy + balanced_accuracy
         → save predictions CSV
         → pickle model to artifacts/best_model.pkl
         → save metrics.json
         → (optional) run_best_explainability()
```

#### `run_best_rtp_prediction(dataset_path, output_dir, config, split)`
Same lifecycle as NAP but uses `task='rtp'` and `RTPEvaluator` → NDLS metric.

#### `run_best_explainability(model, output_dir, task)`
Instantiates `BESTExplainer` and runs all explanation outputs.

### 2.2 Default BEST config

```python
default_best_config = {
    'max_pattern_size_train': 21,          # odd int > 1; tree depth = (size-1)/2
    'max_pattern_size_eval': 21,           # odd int <= max_pattern_size_train
    'process_stage_width_percentage': 0.2, # float 0.0–1.0; 0=n stages, 1=single stage
    'min_freq': 1e-14,                     # cutoff for subtrace patterns (near zero = no filtering)
    'break_buffer': 1.2,                   # RTP loop break factor × max_prefix_len
    'filter_sequences': True,              # filter START/END padding tokens from output
    'ncores': 1,                           # parallelism (>1 may have issues on Windows)
}
```

### 2.3 Metrics contract

| Task | Metric keys in `metrics.json` |
|---|---|
| NAP | `nap_accuracy`, `nap_balanced_accuracy`, `none_share` |
| RTP | `rtp_ndls`, `none_share` |

> Note: No validation split — BEST does not use gradient descent, so there is no need for a val set.

---

## Phase 3 — Backend Integration

> **Goal**: Accept `model_type="best"` and `task="remaining_trace"` through the API and dispatch to the new pipeline functions.

### 3.1 `backend/runner/run_job.py`

Add a new `elif model_type == 'best':` dispatch block:

```python
elif model_type == 'best':
    if task in ('next_activity',):
        from ppm_pipeline import run_best_nap_prediction
        metrics = run_best_nap_prediction(
            dataset_path=dataset_path,
            output_dir=artifacts_dir,
            config=config,
            split=split,
            explainability=explainability,
        )
    elif task == 'remaining_trace':
        from ppm_pipeline import run_best_rtp_prediction
        metrics = run_best_rtp_prediction(
            dataset_path=dataset_path,
            output_dir=artifacts_dir,
            config=config,
            split=split,
            explainability=explainability,
        )
```

### 3.2 `backend/main.py`

- Add `"best"` to any explicit `model_type` validation (if enum-based)
- Add `"remaining_trace"` to any explicit `task` validation

> Both are currently loose `str` fields — changes may be minimal.

---

## Phase 4 — Frontend Integration

> **Goal**: Surface BEST in the wizard UI with correct task filtering, config fields, and explainability options.

### 4.1 `Step2Model.tsx` — Add BEST model card

```typescript
{
  value: 'best',
  label: 'BEST',
  description: 'Bilaterally Expanding Subtrace Tree — a probabilistic, training-free tree model for activity sequence prediction. No neural network required.',
}
```

### 4.2 `Step3Prediction.tsx` — Add `remaining_trace` task + filter by model

New task option:
```typescript
{
  value: 'remaining_trace',
  label: 'Remaining Trace Prediction',
  description: 'Predict the complete sequence of future activities for a running case (activity sequences, not duration).',
}
```

Task visibility filter:
| Model | Visible tasks |
|---|---|
| `transformer` | `next_activity`, `custom_activity`, `event_time`, `remaining_time` |
| `gnn` | `next_activity`, `custom_activity`, `event_time`, `remaining_time`, `unified` |
| `best` | `next_activity`, `remaining_trace` |

### 4.3 `Step4Explainability.tsx` — BEST-specific options

When `model_type === 'best'`:
- **Hide**: SHAP, LIME, Gradient options
- **Show**: "Pattern Analysis" — native BEST explanations showing pattern probabilities, lengths, RPIF distances, and top patterns

### 4.4 `Step5Config.tsx` — BEST config form

Add `BestConfig` TypeScript interface:

```typescript
interface BestConfig {
  max_pattern_size_train: number;          // default: 21
  max_pattern_size_eval: number;           // default: 21
  process_stage_width_percentage: number; // default: 0.2, range: 0.0–1.0
  min_freq: number;                        // default: 1e-14
  break_buffer: number;                    // default: 1.2 (RTP only)
  filter_sequences: boolean;              // default: true
  ncores: number;                          // default: 1
}
```

> No `epochs`, `batch_size`, `learning_rate`, or `patience` — BEST is not a neural model.

### 4.5 Results view — handle `remaining_trace`

- Display predicted future sequences as comma-separated activity lists per case
- Show **NDLS** (Normalized Damerau-Levenshtein Similarity) as the primary metric instead of accuracy
- Show `none_share` (share of cases where no prediction could be found)

---

## File Change Summary

| File | Type | Change | Status |
|---|---|---|---|
| `requirements.txt` | Modify | Add BEST git dependency | ✅ Done |
| `best/__init__.py` | Create | Module marker | ✅ Done |
| `best/predictor.py` | Create | `BESTRunner` adapter class | ✅ Done |
| `explainability/best/__init__.py` | Create | Module marker | ✅ Done |
| `explainability/best/best_explainer.py` | Create | `BESTExplainer` class | ✅ Done |
| `ppm_pipeline.py` | Modify | Add 3 functions | ⏳ Phase 2 |
| `backend/runner/run_job.py` | Modify | Add BEST dispatch branch | ⏳ Phase 3 |
| `backend/main.py` | Modify | Accept new model_type + task | ⏳ Phase 3 |
| `frontend/src/components/steps/Step2Model.tsx` | Modify | Add BEST card | ⏳ Phase 4 |
| `frontend/src/components/steps/Step3Prediction.tsx` | Modify | Add task + model filter | ⏳ Phase 4 |
| `frontend/src/components/steps/Step4Explainability.tsx` | Modify | BEST-specific options | ⏳ Phase 4 |
| `frontend/src/components/steps/Step5Config.tsx` | Modify | BestConfig interface + form | ⏳ Phase 4 |

---

## Verification Checklist

- [ ] Python >3.12 confirmed in `.venv`
- [ ] `pip install git+https://github.com/lmu-dbs/BEST.git#subdirectory=src` succeeds
- [ ] `from best4ppm.models.best import BESTPredictor` imports without error
- [ ] `run_best_nap_prediction()` runs end-to-end on a small test dataset → `artifacts/` written
- [ ] `run_best_rtp_prediction()` runs → `metrics.json` contains `rtp_ndls`
- [ ] Backend API: POST `/runs` with `model_type="best"`, `task="next_activity"` → `status.json` → `"succeeded"`
- [ ] Backend API: POST `/runs` with `model_type="best"`, `task="remaining_trace"` → `status.json` → `"succeeded"`
- [ ] Frontend wizard: BEST model card visible → task filter works → config shows BEST fields → explainability shows Pattern Analysis only → results display NDLS

---

## Known Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Python version incompatibility (BEST requires >3.12) | Check `python --version` in the venv before starting; update if needed |
| Windows multiprocessing issues with `ncores > 1` | Default `ncores=1` in the UI config form |
| Long RTP sequences in results table | Truncate displayed sequences with a "show more" toggle |
| BEST `prepare_train()` can be slow on large logs (pattern generation) | Show a progress indicator in the frontend during the run |
| `max_pattern_size_eval` must be odd and ≤ `max_pattern_size_train` | Add frontend validation on the config form |
