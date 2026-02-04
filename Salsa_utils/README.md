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

## 3. Other commands and instructions

*(Add commands and notes here as you go.)*

- **Training**: From InterGen root run `python tools/train.py --dataset salsa`. Same model/config as base; ensure cache exists and `configs/datasets_salsa.yaml` DATA_ROOT points to `Salsa_utils/lmdb_salsa_intergen_cache`.*

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
