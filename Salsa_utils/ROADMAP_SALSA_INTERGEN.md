# Roadmap: Integrating Salsa Data into InterGen Baseline

This document is the single source of truth for the plan, analysis, and progress of bringing our Salsa dataset into the InterGen framework for training and evaluation. Update this file as we move forward.

---

## 1. Summary of InterGen Data Pipeline

- **Input**: Two people’s motions in the same canonical frame, each **(T, 262)**:
  - 22×3 positions, 22×3 velocities, 21×6 rotations, 2+2 foot contact (from `process_motion_np` in `utils/utils.py`).
- **Pipeline**: Load per-person **(T, 192)** from `motions_processed/person1/`, `person2/` → `process_motion_np` → **(T−1, 262)** → align person2 to person1 with **rigid_transform(relative, motion2)** using frame-0 relative pose (angle, x, z).
- **Training**: `batch["motions"] = cat(motion1, motion2)` → **(B, T, 524)**; `batch["text"]`, `batch["motion_lens"]`; normalization via `data/global_mean.npy`, `global_std.npy`.
- **Split**: `train.txt` / `val.txt` / `test.txt` under `DATA_ROOT` list motion names; dataset filters by `MODE`.

---

## 2. Summary of Salsa Data (Our Side)

- **Cache**: Pair LMDB written by `DataPreprocessor` in `salsa_dataloader.py`. Each LMDB key = one window; value includes **interhuman_data**.
- **InterHuman in cache**: For each window:
  - **leader_motion_ih**, **follower_motion_ih**: **(19, 262)** (20-frame window → T−1=19).
  - **relationship_features**: **(19, 4)**; **root_quat_init_L/F**, **root_pos_init_L/F**.
- **262-dim source**: Built via **salsa_to_interhuman** (uses in2IN’s **process_motion_interhuman**). Same 262 layout as InterGen; **compatible**.

---

## 3. Decisions and Options

| Topic | Decision |
|-------|----------|
| **Integration** | **Option A**: Config switch + branch in dataset build so `python tools/train.py` works with Salsa. Use InterGen codebase functions as much as possible for consistency. |
| **Option B (audio encoder, etc.)** | Leave for later; minimal changes for now. |
| **Normalization** | Use InterGen’s **global mean/std** for now. **Later**: may use Salsa-driven global mean/std (documented here). |
| **Export to InterGen folder layout** (§5) | **Optional for future** (export motions_processed + annots + split). Not required for first training. |
| **Config** | Create a **new config file** for Salsa (e.g. `configs/datasets_salsa.yaml` or similar). |
| **Text for each sample** | Salsa caption + annotations (as suggested). May later update with more sophisticated captions. |
| **Split** | Use **Compas3D official train/val/test**; user will share split info when we implement. |

---

## 4. Data Flow: Window Length and Where to Read From

- **Salsa cache (current)**: Built from **source** pair LMDB. Cache creation uses **short windows** (e.g. `n_poses=20`, `subdivision_stride=20`) for VQVAE/training; each cache entry = one 20-frame window, with `interhuman_data` = (19, 262) per person for that window.
- **InterGen needs longer samples**: e.g. **~200 frames** (or up to 300) per sample for training/visualization, not 20. So we **do not** use the existing cache entries as-is for InterGen; we need **longer contiguous windows**.
- **Intended flow**:
  1. **Pick a window of appropriate length** (e.g. 200 frames) from a **full take** (the long clip, ~4 min).
  2. **Then** convert that window to InterHuman: one conversion for the whole 200-frame segment → (199, 262) per person, then align follower with **rigid_transform**.
- **Data source**: Use the **source** pair LMDB (the same `clip_lmdb_dir` used as **input** to `DataPreprocessor` in `salsa_dataloader.py`). That LMDB has one entry per video; each value = `video` with `vid`, `clips`; each **clip** has full-length `keypoints3d_L`, `rotmat_L`, `keypoints3d_F`, `rotmat_F` for the whole take. We subdivide on the fly with **window_size = 200** (or configurable, e.g. 15–300) and **stride** (e.g. 50 or 100), and convert each window to InterHuman. We **do not** read from the **output** cache (the 20-frame-per-entry LMDB) for InterGen samples.

---

## 5. Reuse vs Implement

### 5.1 Reuse from InterGen (no new code; use as-is)

| What | Where | Use for |
|------|--------|--------|
| **rigid_transform** | `utils/utils.py` | Align follower (262-dim) to leader’s frame using frame-0 relative (angle, x, z). Same as in `datasets/interhuman.py`. |
| **process_motion_np** | `utils/utils.py` | Not used for Salsa (we use salsa_to_interhuman). Only for InterHuman .npy. |
| **MotionNormalizer** | `utils/utils.py` | Inference denormalization. |
| **plot_3d_motion**, **t2m_kinematic_chain** | `utils/plot_script.py`, `utils/paramUtil.py` | Visualization in Gradio (GT and generated). |
| **InterHumanDataset** | `datasets/interhuman.py` | InterHuman data only (current GT tab). |
| **Model, config, inference** | `models/`, `configs/`, `tools/infer.py` | Inference tab. |

### 5.2 Reuse from Salsa / motion_representation (existing)

| What | Where | Use for |
|------|--------|--------|
| **salsa_to_interhuman** | `Salsa-Agent/motion_representation/utils/relationship_features.py` | Convert one person’s (keypoints3d, rotmat) for a window of length T → **(T−1, 262)** + root_quat_init, root_pos_init. Same 262 layout as InterGen. |
| **extract_interhuman_relationship_features** | Same file (with `return_aligned_follower=True`) | Optional: get aligned follower (T, 262) from two canonical motions + root inits; uses in2IN’s **rigid_transform**. For InterGen we prefer **InterGen’s** rigid_transform for consistency, so we only need root inits from salsa_to_interhuman and then call InterGen’s rigid_transform. |
| **Source LMDB schema** | `salsa_dataloader.py`: `DataPreprocessor.run()` | Know how to read: iterate keys; value = `pyarrow.deserialize(value)` → `video['vid']`, `video['clips']`; each clip has `keypoints3d_L`, `rotmat_L`, `keypoints3d_F`, `rotmat_F` (full-length arrays). |
| **Annotation / caption helpers** | `salsa_dataloader.py`: `_extract_take_id_from_vid`, `_load_annotation_file_for_take`, `_match_annotations_to_window`, `_get_annotations_for_window`; `PAIR2LEVEL`, `SALSA_CAPTIONS` | Text for a (vid, start_time, end_time) window: level-based caption or annotation-based description. Can copy or import from Salsa-Agent if path is set. |

### 5.3 Implement new (in Salsa_utils)

| Item | Purpose |
|------|--------|
| **SalsaInterGenDataset** (or similar) | PyTorch `Dataset` that: (1) Opens the **source** pair LMDB (path from config/env). (2) Iterates over videos and clips; for each clip with length ≥ window_size (e.g. 200), yields sliding windows (start, start+window_size) with stride. (3) For each window: slice `keypoints3d_L[start:end]`, `rotmat_L[start:end]`, same for F; call **salsa_to_interhuman** for L and F → (T−1, 262), root_quat_L, root_pos_L, root_quat_F, root_pos_F; compute relative (angle, x, z) from frame-0 roots; call **InterGen’s rigid_transform(relative, motion_F)** → motion2; pad to max_gt_length if needed. (4) Returns (name, text, motion1, motion2, gt_length) in InterGen format. |
| **get_salsa_caption(vid, start_time, end_time)** | Returns a string for the window: e.g. from PAIR2LEVEL + SALSA_CAPTIONS, or from annotations (reuse or reimplement logic from `_get_annotations_for_window` + format moves/descriptions). |
| **Config / env for source LMDB** | Path to source pair LMDB (same as `clip_lmdb_dir` for cache creation). E.g. in `Salsa_utils` config or env var. |
| **Split filtering** | When Compas3D official split is available: filter which (vid, clip_idx) or which windows are train/val/test (e.g. by take ID or list file). |
| **Gradio: Salsa in GT tab** | In the visualization app: option (e.g. dropdown) to choose “InterHuman” or “Salsa”; when Salsa is selected, load sample from SalsaInterGenDataset instead of InterHumanDataset (same output format: caption, video). |

### 5.4 Summary

- **Reuse**: InterGen’s rigid_transform, plotting, normalizer, model; Salsa’s salsa_to_interhuman, source LMDB schema, and annotation/caption logic.
- **Implement**: One dataset class that reads **source** LMDB, extracts **long windows** (e.g. 200 frames), converts each window with **salsa_to_interhuman** + **InterGen rigid_transform**, and yields (name, text, motion1, motion2, gt_length); plus caption helper and Gradio switch for Salsa.
- **Do not** use the existing 20-frame cache entries as InterGen samples; we need longer windows and a single conversion per window.

---

## 6. Optional for Future

- **Export Salsa to InterGen folder layout**: Script under Salsa_utils that writes `motions_processed/person1/`, `person2/`, `annots/`, `train/val/test.txt` so the original InterHumanDataset can be used without a custom dataset class. Deferred.
- **Option B**: Separate train script under Salsa_utils (e.g. with audio encoder); minimal changes for now.
- **Salsa-driven normalization**: Compute global mean/std from Salsa (524,) and use when training on Salsa; keep in plan for later.

---

## 6. File Layout Under Salsa_utils

- `Salsa_utils/`
  - **ROADMAP_SALSA_INTERGEN.md** (this file)
  - **visualization/** – Gradio app and helpers for viewing GT and inference (see §7).
  - **interhuman_utils.py** – `process_motion_interhuman`, `rigid_transform`, `salsa_to_interhuman`, `align_follower_to_leader`; uses InterGen or in2IN when available.
  - **salsa_caption.py** – `get_salsa_caption`, PAIR2LEVEL, SALSA_CAPTIONS.
  - **salsa_intergen_cache.py** – Script: source pair LMDB → 210-frame windows (stride 50), full-window InterHuman conversion, raw audio, dict samples → output LMDB.
  - **salsa_intergen_dataset.py** – PyTorch Dataset over cache LMDB; returns dict (name, text, motion1, motion2, gt_length, raw_audio, metadata); `collate_salsa_intergen` for batching.
  - **split.py** or **splits/** – (Optional) Generate/read train/val/test indices; dataset supports split_dir + train/val/test.txt.
  - **data/** – (Optional) Salsa global mean/std.
- `configs/datasets_salsa.yaml` – Salsa dataset config (DATA_ROOT = cache LMDB path).

---

## 7. Visualization Interface (First Deliverable)

- **Location**: `Salsa_utils/visualization/`
- **Tool**: Gradio web app.
- **Tabs**:
  1. **Ground truth**: Load sample from InterGen dataset (by index or random), show caption, visualize motion using **InterGen’s** `plot_3d_motion` (and paramUtil kinematic chain). Option to “Use caption for inference” (copy text to Inference tab).
  2. **Inference**: Text prompt (and optional window size); run pretrained InterGen model; show output video using same InterGen visualization.
- **Requirement**: Use InterGen base code as much as possible (especially `utils.plot_script`, `utils.paramUtil`, inference logic from `tools.infer.py`) so visualization is consistent and verifiable.
- **Later**: Add Salsa data to the same interface (e.g. load Salsa sample, visualize, use for inference) via SalsaInterGenDataset and dataset selector (§5.3).

---

## 9. What’s Left and Next Steps (Text-to-Motion Training)

**Goal:** Train the same as base InterGen — **text-to-motion** only (no audio in the model). Visualization and Salsa dataloader (motion, captions, audio) are verified; next is wiring the Salsa cache into the training pipeline.

**Remaining work (in order):**

1. **DataModule branch** (`datasets/__init__.py`)
   - When `data_cfg.NAME == "salsa_intergen"`, build `SalsaInterGenDataset(cfg)` instead of `InterHumanDataset`.
   - Resolve `DATA_ROOT` to the cache LMDB path (e.g. `Salsa_utils/lmdb_salsa_intergen_cache` or path in `configs/datasets_salsa.yaml`).
   - Use a **collate** that converts Salsa dict batches into the format expected by `LitTrainModel.forward(batch_data)`:
     - Training expects `batch_data = (name, text, motion1, motion2, motion_lens)` (same as InterHuman).
     - `SalsaInterGenDataset` returns a dict (`name`, `text`, `motion1`, `motion2`, `gt_length`, `raw_audio`, …). Collate must produce `(list of names, list of texts, motion1 tensor, motion2 tensor, motion_lens tensor)` so no changes are needed in `train.py` or the model.

2. **Train script config selection** (`tools/train.py`)
   - Load Salsa data config instead of InterHuman when training on Salsa (e.g. `get_config("configs/datasets_salsa.yaml").salsa_train` or a CLI flag to choose dataset).

3. **Config** (`configs/datasets_salsa.yaml`)
   - Set `DATA_ROOT` to the actual cache path (e.g. `Salsa_utils/lmdb_salsa_intergen_cache` when run from InterGen root). Optionally add `SPLIT_DIR` when train/val/test splits are used.

4. **Splits (optional for first run)**
   - Without split files, use all cache samples as train (or single split). When Compas3D/official splits are available, add `train.txt` / `val.txt` / `test.txt` (one sample `name` per line) and set `SPLIT_DIR`; `SalsaInterGenDataset(split=..., split_dir=...)` already supports this.

5. **Run training**
   - Same as base: `python tools/train.py` (with config pointing to Salsa). Model and loss stay text-to-motion; no code changes in `LitTrainModel` or `InterGen` model.

**Deferred / optional later:**

- **Salsa-driven normalization:** Compute global mean/std from Salsa (524-dim) and use in training; for first run use InterGen’s existing `global_mean.npy` / `global_std.npy`.
- **Export to InterGen folder layout:** Optional script to write `motions_processed/`, `annots/`, split files so base `InterHumanDataset` could be used.
- **Option B (e.g. audio encoder):** Not required for text-to-motion.

---

## 10. Progress Log

| Date | Step | Status | Notes |
|------|------|--------|--------|
| (today) | Roadmap and plan documented | Done | ROADMAP_SALSA_INTERGEN.md created. |
| (today) | Visualization Gradio app (InterGen only) | Done | `Salsa_utils/visualization/`: app.py, intergen_vis_utils.py, README. Tab 1: GT; Tab 2: Inference. |
| | User verification of visualization | Done | User confirmed visualization and dataloader (motion, captions, audio) are correct. |
| (today) | Salsa dataset class + cache + interhuman_utils | Done | salsa_intergen_dataset.py, salsa_intergen_cache.py, interhuman_utils.py (210 frames, stride 50; full-window InterHuman conversion; raw audio; dict output). |
| | Config + DataModule branch for Salsa | Pending | datasets_salsa.yaml in place; DataModule must instantiate SalsaInterGenDataset when NAME=salsa_intergen and use collate → (name, text, motion1, motion2, motion_lens). |
| | Train script: use Salsa config | Pending | train.py loads salsa_train from datasets_salsa.yaml (or switch) so DATA_ROOT points to cache. |
| | Train on Salsa (text-to-motion) | Pending | Run training; optional: Salsa normalization later. |
| | Optional: Export to InterGen folder layout | Future | |
| | Optional: Option B (e.g. audio encoder) | Future | |

---

*Last updated: roadmap updated with “What’s left”, next steps for text-to-motion training, and progress log.*
