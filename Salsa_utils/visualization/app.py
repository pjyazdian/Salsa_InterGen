"""
Gradio app for InterGen: Ground-truth visualization and Inference.
Uses InterGen codebase for plotting and inference (utils.plot_script, utils.paramUtil, tools.infer).
Run from InterGen root: python -m Salsa_utils.visualization.app
Or: cd Baselines/Salsa_InterGen && python Salsa_utils/visualization/app.py
"""
import os
import sys
import random
import copy
import tempfile

# Run from InterGen root so configs, data paths, and imports resolve
INTERGEN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(INTERGEN_ROOT)
if INTERGEN_ROOT not in sys.path:
    sys.path.insert(0, INTERGEN_ROOT)

import numpy as np
import torch
import gradio as gr
import scipy.ndimage.filters as filters

from collections import OrderedDict

# InterGen imports (after path set)
from configs import get_config
from models import InterGen
from utils.plot_script import plot_3d_motion
from utils import paramUtil
from utils.utils import MotionNormalizer

# Import from same folder (works when run as script or as -m Salsa_utils.visualization.app)
_VIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _VIS_DIR not in sys.path:
    sys.path.insert(0, _VIS_DIR)
try:
    from intergen_vis_utils import plot_gt_motion_to_file, plot_generated_motion_to_file, KINEMATIC_CHAIN
except ImportError:
    from .intergen_vis_utils import plot_gt_motion_to_file, plot_generated_motion_to_file, KINEMATIC_CHAIN

# ---------------------------------------------------------------------------
# Paths for Salsa cache (check exists → else create)
# ---------------------------------------------------------------------------
SALSA_CACHE_DIR = os.path.join(INTERGEN_ROOT, "Salsa_utils", "lmdb_salsa_intergen_cache")
SALSA_SOURCE_LMDB = os.path.normpath(os.path.join(INTERGEN_ROOT, "..", "..", "Salsa-Agent", "dataset_processed_New", "lmdb_Salsa_pair", "lmdb_train"))

# Directories to search for .ckpt files (relative to INTERGEN_ROOT); add more to list as needed
CHECKPOINT_SEARCH_DIRS = ["checkpoints_org", "checkpoints"]

# Salsa global mean/std (used when a Salsa-trained checkpoint is selected)
SALSA_MEAN_PATH = os.path.join(INTERGEN_ROOT, "data", "global_mean_salsa.npy")
SALSA_STD_PATH = os.path.join(INTERGEN_ROOT, "data", "global_std_salsa.npy")


def discover_checkpoints():
    """Return list of (label, path) for dropdown: label = path relative to INTERGEN_ROOT, path = absolute."""
    found = []
    for rel_dir in CHECKPOINT_SEARCH_DIRS:
        base = os.path.join(INTERGEN_ROOT, rel_dir)
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith(".ckpt"):
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, INTERGEN_ROOT)
                    found.append((rel, full))
    found.sort(key=lambda x: x[0])
    return found


def _salsa_cache_exists():
    """True if cache dir exists and has at least one LMDB entry."""
    if not os.path.isdir(SALSA_CACHE_DIR):
        return False
    try:
        import lmdb
        env = lmdb.open(SALSA_CACHE_DIR, readonly=True, lock=False)
        with env.begin(write=False) as txn:
            n = txn.stat()["entries"]
        env.close()
        return n > 0
    except Exception:
        return False


def ensure_salsa_cache():
    """If Salsa cache does not exist, create it. Returns (True, None) on success, (False, error_msg) on failure."""
    if _salsa_cache_exists():
        return True, None
    if not os.path.isdir(SALSA_SOURCE_LMDB):
        return False, (
            f"Salsa source LMDB not found: {SALSA_SOURCE_LMDB}. "
            "Create it first or run from repo: conda activate intergen && python -m Salsa_utils.salsa_intergen_cache "
            f"--source_lmdb <path> --out_lmdb {SALSA_CACHE_DIR}"
        )
    try:
        from Salsa_utils.salsa_intergen_cache import run as run_cache
        os.makedirs(SALSA_CACHE_DIR, exist_ok=True)
        run_cache(
            source_lmdb_dir=SALSA_SOURCE_LMDB,
            out_lmdb_dir=SALSA_CACHE_DIR,
        )
        return _salsa_cache_exists(), None
    except Exception as e:
        import traceback
        return False, f"Cache creation failed: {e}\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# Dataset (lazy) – same pipeline as base: configs/datasets.yaml + InterHumanDataset(opt)
# ---------------------------------------------------------------------------
_dataset = None
_data_cfg = None
_dataset_salsa = None


def get_dataset(mode="test"):
    """Load InterGen dataset exactly as base code (datasets/__init__.py + interhuman.py). Returns (dataset, error_str)."""
    global _dataset, _data_cfg
    if _dataset is not None and _data_cfg is not None and getattr(_data_cfg, "MODE", None) == mode:
        return _dataset, None
    try:
        # Same as train.py: get_config("configs/datasets.yaml") then interhuman_test / interhuman
        cfg = get_config("configs/datasets.yaml")
        data_cfg = getattr(cfg, f"interhuman_{mode}", None) or cfg.interhuman
        # Resolve DATA_ROOT to absolute (base code uses cwd-relative)
        data_root = getattr(data_cfg, "DATA_ROOT", "")
        if data_root and not os.path.isabs(data_root):
            data_root = os.path.abspath(os.path.normpath(os.path.join(INTERGEN_ROOT, data_root)))
        else:
            data_root = os.path.abspath(data_root) if data_root else ""
        # Build opt with same keys as yacs config (base: InterHumanDataset(data_cfg))
        from types import SimpleNamespace
        opt = SimpleNamespace(
            NAME=getattr(data_cfg, "NAME", "interhuman"),
            DATA_ROOT=data_root,
            MOTION_REP=getattr(data_cfg, "MOTION_REP", "global"),
            MODE=mode,
            CACHE=getattr(data_cfg, "CACHE", True),
        )
        _data_cfg = opt  # set before creating dataset so error message can show DATA_ROOT
        # Same as datasets/__init__.py: InterHumanDataset(data_cfg)
        from datasets.interhuman import InterHumanDataset
        _dataset = InterHumanDataset(opt)
        return _dataset, None  # success
    except Exception as e:
        import traceback
        err_msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        traceback.print_exc()
        return None, err_msg


def get_dataset_salsa():
    """Load Salsa cache dataset; ensure cache exists first. Returns (dataset, error_str)."""
    global _dataset_salsa
    if _dataset_salsa is not None:
        return _dataset_salsa, None
    ok, err = ensure_salsa_cache()
    if not ok:
        return None, err or "Salsa cache missing and could not be created."
    try:
        from Salsa_utils.salsa_intergen_dataset import SalsaInterGenDataset
        _dataset_salsa = SalsaInterGenDataset(SALSA_CACHE_DIR, max_gt_length=300, swap_person=False)
        return _dataset_salsa, None
    except Exception as e:
        import traceback
        return None, f"Load Salsa dataset: {e}\n{traceback.format_exc()}"


def _format_salsa_metadata(sample):
    """Format Salsa sample metadata for display (includes annotations when available)."""
    meta = sample.get("metadata", {})
    if not meta:
        return "(no metadata)"
    lines = [
        f"vid: {meta.get('vid', '')}",
        f"clip_idx: {meta.get('clip_idx', '')}  window_idx: {meta.get('window_idx', '')}",
        f"start_frame: {meta.get('start_frame', '')}  end_frame: {meta.get('end_frame', '')}",
        f"start_time: {meta.get('start_time', '')} s  end_time: {meta.get('end_time', '')} s",
        f"name: {sample.get('name', '')}",
    ]
    # Annotations (included when available from dataset_root at cache time)
    ann = meta.get("annotations", {})
    lines.append("\nAnnotations:")
    if ann:
        moves = ann.get("moves", [])
        errors = ann.get("errors", [])
        styling_l = ann.get("styling_leader", [])
        styling_f = ann.get("styling_follower", [])
        if moves:
            lines.append("  Moves:")
            for m in moves[:5]:
                desc = (m.get("description") or "")[:80]
                lines.append(f"    - {desc}")
        if errors:
            lines.append("  Errors:")
            for e in errors[:5]:
                desc = (e.get("description") or "")[:80]
                lines.append(f"    - {desc}")
        if styling_l:
            lines.append("  Styling (leader):")
            for s in styling_l[:2]:
                lines.append(f"    - {(s.get('description') or '')[:80]}")
        if styling_f:
            lines.append("  Styling (follower):")
            for s in styling_f[:2]:
                lines.append(f"    - {(s.get('description') or '')[:80]}")
        if not (moves or errors or styling_l or styling_f):
            lines.append("  (none for this window)")
    else:
        lines.append("  (none or not loaded at cache time)")
    return "\n".join(lines)


def load_gt_sample(index_str, dataset_choice="InterHuman", seed_for_reproduce=True):
    """Load one sample by index. Returns (caption, video_path, error_msg, audio_tuple, metadata_str)."""
    empty_extra = (None, "")  # no audio, no metadata for InterHuman
    try:
        idx = int(index_str)
    except ValueError:
        return None, None, "Please enter a valid integer index.", None, ""

    if dataset_choice == "Salsa":
        dataset, load_err = get_dataset_salsa()
        if dataset is None:
            return None, None, (load_err or "Could not load Salsa dataset."), None, ""
        n = len(dataset)
        if n == 0:
            return None, None, "Salsa cache is empty.", None, ""
        idx = idx % n
        if seed_for_reproduce:
            random.seed(idx)
            np.random.seed(idx)
        try:
            sample = dataset[idx]
        except Exception as e:
            return None, None, f"Error loading sample: {e}", None, ""
        text = sample["text"]
        motion1 = np.asarray(sample["motion1"])
        motion2 = np.asarray(sample["motion2"])
        gt_length = int(sample["gt_length"])
        # Raw audio: mono, 1D, trimmed to motion duration (same as Salsa-Agent; fixes double/repeated playback)
        raw_audio = sample.get("raw_audio")
        audio_sr = int(sample.get("audio_sr", 24000))
        pose_fps = 20
        if raw_audio is not None and getattr(raw_audio, "size", 0) > 0:
            audio_arr = np.asarray(raw_audio, dtype=np.float32)
            # Mono: (samples, channels) -> mean(axis=1); (channels, samples) -> mean(axis=0); else ravel
            if audio_arr.ndim == 2:
                if audio_arr.shape[-1] <= 2 and audio_arr.shape[0] > audio_arr.shape[-1]:
                    audio_arr = np.mean(audio_arr, axis=1)
                else:
                    audio_arr = np.mean(audio_arr, axis=0)
            else:
                audio_arr = audio_arr.ravel()
            # Trim to motion duration (Gradio expects 1D; extra length causes double/repeated playback)
            expected_samples = int(gt_length / pose_fps * audio_sr)
            max_samples = int(210 / pose_fps * audio_sr)
            trim_to = min(expected_samples, max_samples)
            if len(audio_arr) > trim_to:
                audio_arr = audio_arr[:trim_to]
            audio_arr = np.asarray(audio_arr).flatten()
            m = np.abs(audio_arr).max()
            if m > 1.0 and m > 0:
                audio_arr = audio_arr / m
            audio_tuple = (audio_sr, audio_arr)
        else:
            audio_tuple = None
        metadata_str = _format_salsa_metadata(sample)
    else:
        dataset, load_err = get_dataset("test")
        if dataset is None:
            data_root = getattr(_data_cfg, "DATA_ROOT", "") if _data_cfg else ""
            return (
                None, None,
                "Could not load InterGen dataset. DATA_ROOT: "
                f"{data_root!r}\n\nError:\n{load_err or 'unknown'}",
                None, "",
            )
        n = len(dataset)
        if n == 0:
            data_root = getattr(_data_cfg, "DATA_ROOT", "") if _data_cfg else ""
            return (
                None, None,
                "Dataset is empty. Check that DATA_ROOT has train/val/test.txt (one motion name per line) "
                f"and matching .npy/.txt files. DATA_ROOT: {data_root!r}",
                None, "",
            )
        idx = idx % n
        if seed_for_reproduce:
            random.seed(idx)
            np.random.seed(idx)
        try:
            name, text, gt_motion1, gt_motion2, gt_length = dataset[idx]
        except Exception as e:
            return None, None, f"Error loading sample: {e}", None, ""
        gt_length = int(gt_length) if hasattr(gt_length, "__int__") else int(gt_length.item())
        motion1 = gt_motion1.numpy() if isinstance(gt_motion1, torch.Tensor) else np.asarray(gt_motion1)
        motion2 = gt_motion2.numpy() if isinstance(gt_motion2, torch.Tensor) else np.asarray(gt_motion2)
        audio_tuple = None
        metadata_str = ""

    try:
        path = plot_gt_motion_to_file(motion1, motion2, gt_length, text, fps=30)
    except Exception as e:
        return text, None, f"Plot error: {e}", audio_tuple if dataset_choice == "Salsa" else None, metadata_str if dataset_choice == "Salsa" else ""
    return text, path, None, audio_tuple, metadata_str


# ---------------------------------------------------------------------------
# Inference (lazy model load; checkpoint selectable via dropdown)
# ---------------------------------------------------------------------------
_litmodel = None
_model_cfg = None
_infer_cfg = None
_loaded_ckpt_path = None
_loaded_salsa_stats = None  # True if current model/wrapper use Salsa mean/std


def _is_salsa_checkpoint(ckpt_path):
    """True if this checkpoint is Salsa-trained (use Salsa global mean/std for denorm)."""
    if not ckpt_path:
        return False
    path = os.path.normpath(ckpt_path)
    root = os.path.normpath(INTERGEN_ROOT)
    if not path.startswith(root):
        return "salsa" in path.lower()
    rel = os.path.relpath(path, root)
    return "checkpoints" in rel and "checkpoints_org" not in rel


def _apply_normalizer_env(use_salsa):
    """Set INTERGEN_GLOBAL_MEAN/STD so MotionNormalizer uses Salsa or default stats."""
    if use_salsa and os.path.isfile(SALSA_MEAN_PATH) and os.path.isfile(SALSA_STD_PATH):
        os.environ["INTERGEN_GLOBAL_MEAN"] = SALSA_MEAN_PATH
        os.environ["INTERGEN_GLOBAL_STD"] = SALSA_STD_PATH
    else:
        os.environ.pop("INTERGEN_GLOBAL_MEAN", None)
        os.environ.pop("INTERGEN_GLOBAL_STD", None)


def get_litmodel(ckpt_path=None):
    """Load model, using ckpt_path if given else config CHECKPOINT. Reloads when ckpt_path or Salsa vs default stats change."""
    global _litmodel, _model_cfg, _infer_cfg, _loaded_ckpt_path, _loaded_salsa_stats
    path_to_load = (ckpt_path or "").strip() or None
    if path_to_load is None:
        _model_cfg = get_config("configs/model.yaml")
        path_to_load = getattr(_model_cfg, "CHECKPOINT", None)
        if path_to_load and not os.path.isabs(path_to_load):
            path_to_load = os.path.join(INTERGEN_ROOT, path_to_load)
    use_salsa = _is_salsa_checkpoint(path_to_load)
    if _litmodel is not None and _loaded_ckpt_path == path_to_load and _loaded_salsa_stats == use_salsa:
        return _litmodel, None
    _apply_normalizer_env(use_salsa)
    try:
        if _model_cfg is None:
            _model_cfg = get_config("configs/model.yaml")
        _infer_cfg = get_config("configs/infer.yaml")
        model = InterGen(_model_cfg)
        if path_to_load and os.path.exists(path_to_load):
            ckpt = torch.load(path_to_load, map_location="cpu")
            state = ckpt.get("state_dict", {}) or {}
            for k in list(state.keys()):
                if "model." in k:
                    state[k.replace("model.", "")] = state.pop(k)
            model.load_state_dict(state, strict=False)
        _loaded_ckpt_path = path_to_load
        _loaded_salsa_stats = use_salsa
        _litmodel = _LitGenModelWrapper(model, _infer_cfg)
        _litmodel.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        return _litmodel, None
    except Exception as e:
        return None, str(e)


class _LitGenModelWrapper:
    """Thin wrapper around InterGen inference logic (same as tools/infer.py)."""

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.normalizer = MotionNormalizer()

    def to(self, device):
        """Move the underlying model to device (so we can call litmodel.to(device))."""
        self.model = self.model.to(device)
        return self

    def generate_loop(self, prompt, window_size=210):
        self.model.eval()
        device = next(self.model.parameters()).device
        batch = OrderedDict()
        batch["motion_lens"] = torch.zeros(1, 1, dtype=torch.long, device=device)
        batch["motion_lens"][:] = window_size
        batch["text"] = [prompt]
        batch["prompt"] = prompt
        with torch.no_grad():
            batch = self.model.text_process(batch)
            batch.update(self.model.decode_motion(batch))
        motion_output_both = batch["output"][0].reshape(batch["output"][0].shape[0], 2, -1)
        motion_output_both = self.normalizer.backward(motion_output_both.cpu().detach().numpy())
        sequences = [[], []]
        for j in range(2):
            motion_output = motion_output_both[:, j]
            joints3d = motion_output[:, : 22 * 3].reshape(-1, 22, 3)
            joints3d = filters.gaussian_filter1d(joints3d, 1, axis=0, mode="nearest")
            sequences[j].append(joints3d)
        sequences[0] = np.concatenate(sequences[0], axis=0)
        sequences[1] = np.concatenate(sequences[1], axis=0)
        # Convert back to (T, 262) layout for plot_generated_motion_to_file: we only have joints (T,22,3)
        # plot_3d_motion expects list of (T, 22, 3); we have that in sequences
        return [sequences[0], sequences[1]]


def run_inference(prompt, window_size_str, ckpt_path=None):
    """Run InterGen on a text prompt using the selected checkpoint. Returns (video_path, error_msg)."""
    if not (prompt or "").strip():
        return None, "Enter a text prompt."
    try:
        window_size = int(window_size_str) if window_size_str else 210
        window_size = max(15, min(300, window_size))
    except ValueError:
        window_size = 210
    litmodel, err = get_litmodel(ckpt_path=ckpt_path)
    if err:
        return None, f"Model load error: {err}"
    try:
        sequences = litmodel.generate_loop(prompt.strip(), window_size=window_size)
        path = plot_generated_motion_to_file(sequences, prompt.strip(), fps=30)
        return path, None
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="InterGen – Ground truth & Inference", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# InterGen: Ground truth & Inference")
        gr.Markdown("**Ground truth**: Load and visualize a dataset sample. **Inference**: Generate motion from a text prompt using the pretrained model.")

        with gr.Tabs():
            # ----- Tab 1: Ground truth -----
            with gr.TabItem("Ground truth"):
                gr.Markdown("Load a sample by index and visualize both persons. **InterHuman**: config dataset. **Salsa**: cache in Salsa_utils (created automatically if missing). For Salsa, sample audio and metadata are shown below.")
                gt_dataset = gr.Radio(
                    choices=["InterHuman", "Salsa"],
                    value="InterHuman",
                    label="Dataset",
                )
                gt_index = gr.Number(value=0, label="Sample index", precision=0)
                gt_btn = gr.Button("Load sample")
                gt_caption = gr.Textbox(label="Caption", interactive=False)
                gt_video = gr.Video(label="Motion (two persons)")
                gt_error = gr.Textbox(label="Message", interactive=False, visible=True)
                gt_audio = gr.Audio(label="Sample audio (Salsa only)", type="numpy", visible=True)
                gt_metadata = gr.Textbox(label="Metadata (Salsa only)", interactive=False, lines=12, placeholder="Load a Salsa sample to see vid, clip/window indices, time range, annotations…")
                gt_btn.click(
                    fn=lambda i, ds: load_gt_sample(str(int(i)), dataset_choice=ds),
                    inputs=[gt_index, gt_dataset],
                    outputs=[gt_caption, gt_video, gt_error, gt_audio, gt_metadata],
                )
                gr.Markdown("Copy this caption to the **Inference** tab:")
                use_caption_btn = gr.Button("Use caption for inference")

            # ----- Tab 2: Inference -----
            with gr.TabItem("Inference"):
                gr.Markdown("Generate two-person motion from a text prompt. Choose which checkpoint to use (found under `checkpoints_org/` and `checkpoints/`).")
                ckpt_choices = discover_checkpoints()
                if not ckpt_choices:
                    ckpt_choices = [("Config default", "")]
                inf_ckpt = gr.Dropdown(
                    choices=[(label, path) for label, path in ckpt_choices],
                    value=ckpt_choices[0][1],
                    label="Checkpoint",
                    allow_custom_value=False,
                )
                inf_prompt = gr.Textbox(label="Text prompt", placeholder="e.g. Two people embrace each other.")
                inf_window = gr.Number(value=210, label="Window size (frames)", precision=0)
                inf_btn = gr.Button("Generate")
                inf_video = gr.Video(label="Generated motion")
                inf_error = gr.Textbox(label="Message", interactive=False, visible=True)

                def do_inference(p, w, ckpt):
                    v, e = run_inference(p, str(int(w)) if w is not None else "210", ckpt_path=ckpt)
                    return v, e or ""

                inf_btn.click(
                    fn=do_inference,
                    inputs=[inf_prompt, inf_window, inf_ckpt],
                    outputs=[inf_video, inf_error],
                )

        # "Use caption for inference" copies Ground truth caption into Inference prompt
        def copy_caption(cap):
            return cap or ""

        use_caption_btn.click(fn=copy_caption, inputs=[gt_caption], outputs=[inf_prompt])

        gr.Markdown("---\n*Visualization uses InterGen's `plot_3d_motion` and paramUtil. Inference uses the same logic as `tools/infer.py`.*")

    return demo


def main():
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7861)


if __name__ == "__main__":
    main()
