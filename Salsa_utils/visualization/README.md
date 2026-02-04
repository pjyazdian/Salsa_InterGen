# InterGen Visualization (Gradio)

Web UI for **ground-truth** visualization and **inference** using the InterGen codebase. All plotting and inference logic comes from InterGen (`utils.plot_script`, `utils.paramUtil`, `tools.infer.py`).

## Setup

1. Use an environment where **InterGen dependencies** are already installed (e.g. the same env where `python tools/infer.py` runs).
2. From the **InterGen root** (`Baselines/Salsa_InterGen`), install Gradio:

```bash
pip install gradio
```

(InterGen’s own `requirements.txt` does not include Gradio; install it separately.)

## Run the app

From the **InterGen root**:

```bash
# Option 1: as module (recommended)
python -m Salsa_utils.visualization.app

# Option 2: as script
python Salsa_utils/visualization/app.py
```

Then open the URL shown (default: `http://0.0.0.0:7860`).

## Tabs

1. **Ground truth**  
   - Load a sample from the InterGen **test** set by index.  
   - Shows caption and a two-person motion video (using InterGen’s `plot_3d_motion`).  
   - **“Use caption for inference”** copies the caption into the Inference tab prompt.

2. **Inference**  
   - Enter a text prompt (or use the caption from the Ground truth tab).  
   - Optionally set window size (frames).  
   - **“Generate”** runs the pretrained InterGen model and shows the result video (same visualization pipeline as `tools/infer.py`).

## Requirements

- **InterHuman data**: `configs/datasets.yaml` points `DATA_ROOT` to the InterHuman folder (e.g. `../InterHuman_DATASET` when running from Salsa_InterGen). That folder must contain `motions_processed/person1/`, `person2/`, `annots/`, and `train.txt`, `val.txt`, `test.txt` (and optionally `ignore_list.txt`). See `Salsa_utils/DATA_SETUP.md` for layout.
- **Pretrained checkpoint** at path set in `configs/model.yaml` (e.g. `checkpoints/intergen.ckpt`) for Inference.

If the dataset is missing or paths are wrong, the Ground truth tab will show an error. Inference can still be used if the checkpoint is present.
