# Salsa_utils

Utilities to run the **InterGen** baseline on the **Salsa** dataset: cache building, dataset loading, and visualization.

**Run all commands from the InterGen root** (`Baselines/Salsa_InterGen`).

---

## 1. Create the Salsa InterGen cache

Builds an LMDB cache from the source Salsa pair LMDB (210-frame windows, stride 50, InterHuman-style motion + raw audio).

```bash
python -m Salsa_utils.salsa_intergen_cache \
  --source_lmdb ../../Salsa-Agent/dataset_processed_New/lmdb_Salsa_pair/lmdb_train \
  --out_lmdb Salsa_utils/lmdb_salsa_intergen_cache
```

**Optional arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--window_frames` | 210 | Window length in frames |
| `--stride_frames` | 50 | Stride between windows |
| `--pose_fps` | 20 | Pose FPS |
| `--dataset_root` | (empty) | Dataset root for annotations |
| `--use_annotations` | off | Use annotations for captions when available |
| `--debug` | off | Print full traceback and shapes on first skip |

Example with custom window/stride and debug:

```bash
python -m Salsa_utils.salsa_intergen_cache \
  --source_lmdb ../../Salsa-Agent/dataset_processed_New/lmdb_Salsa_pair/lmdb_train \
  --out_lmdb Salsa_utils/lmdb_salsa_intergen_cache \
  --window_frames 210 \
  --stride_frames 50 \
  --debug
```

After a successful run, the cache is at `Salsa_utils/lmdb_salsa_intergen_cache/`. Regenerate this directory if you change window/stride or fix audio/mono logic.

---

## 2. Visualization (Gradio)

From the InterGen root:

```bash
python -m Salsa_utils.visualization.app
```

Then open the URL (e.g. `http://0.0.0.0:7860`). In **Ground truth**, choose dataset **Salsa** to load samples from the cache (motion + audio + metadata). See `Salsa_utils/visualization/README.md` for details.

---

## 3. Training

From the InterGen root:

```bash
python tools/train.py --dataset salsa
```

Use the same model/config as the base InterGen setup. Ensure the cache exists and `configs/datasets_salsa.yaml` has `DATA_ROOT` pointing to `Salsa_utils/lmdb_salsa_intergen_cache`.

### Recorded metrics

**Training (WandB / TensorBoard)** — all are losses (lower is better):

| Metric | Meaning |
|--------|--------|
| **total** | Sum of all motion losses; main training objective. |
| **simple** | Masked L1/L2 between predicted and target normalized motion (raw diffusion target). |
| **RO** | Relative orientation: mismatch between the two people’s facing directions. Keeps interaction orientation correct. |
| **DM** | Distance map: MSE between predicted and target inter-person joint distance matrices (only for joints &lt; 1 m apart). Enforces proximity/contact. |
| **JA** | Joint affinity: penalizes predicted inter-person distances where the target has close joints; keeps interaction geometry. |
| **POSE_0 / POSE_1** | Local joint positions per person. Pose accuracy. |
| **VEL_0 / VEL_1** | Joint velocity error per person. Temporal smoothness. |
| **FC_0 / FC_1** | Foot contact: feet velocity when grounded. Reduces foot sliding. |
| **BL_0 / BL_1** | Bone length error per person. Keeps skeleton rigid. |
| **TR_0 / TR_1** | Root (pelvis) XZ trajectory. Global translation. |
| **text_ce_from_motion**, **text_ce_from_d**, **text_mixed_ce** | Text–motion embedding alignment (evaluator); cross-entropy losses. |

**Offline evaluation** (`tools/eval.py`):

| Metric | Meaning | Direction |
|--------|--------|-----------|
| **MM Distance** | Mean distance between text and motion embeddings (same index). | Lower = better text–motion match. |
| **R_precision** | For each text, fraction of times the correct motion is in the top-k nearest. | Higher = better retrieval. |
| **FID** | Frechét distance between distributions of motion embeddings (GT vs generated). | Lower = generated motions closer to real distribution. |
| **Diversity** | Mean pairwise distance between generated motion embeddings. | Higher = more variety (avoids mode collapse). |
| **MultiModality** | For the same text, mean pairwise distance between multiple generated motions. | Higher = more diverse motions per prompt. |

---

## 4. Other commands and instructions

*(Add commands and notes here as you go.)*

---

## Files

| File | Purpose |
|------|---------|
| `salsa_intergen_cache.py` | Script: source pair LMDB → cache LMDB (windows, InterHuman motion, raw audio). |
| `salsa_intergen_dataset.py` | PyTorch `Dataset` over the cache; `collate_salsa_intergen` for batching. |
| `interhuman_utils.py` | Salsa → InterHuman conversion, rigid transform, alignment. |
| `salsa_caption.py` | Caption generation for Salsa samples. |
| `visualization/` | Gradio app for ground-truth and inference. |
| `ROADMAP_SALSA_INTERGEN.md` | Plan and status for Salsa–InterGen integration. |
