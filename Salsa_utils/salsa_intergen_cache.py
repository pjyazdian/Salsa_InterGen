"""
Build InterGen-style cache from Salsa source pair LMDB.

Reads the SOURCE pair LMDB (full-length clips), extracts 210-frame windows with 50-frame
stride, converts each FULL window to InterHuman in one go (not small chunks then combine),
and writes samples with motion1, motion2, text, raw_audio, and metadata. No audio or
motion tokenization; raw audio per sample for future use.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from pathlib import Path

import lmdb
import numpy as np
import pyarrow
from tqdm import tqdm

# Add InterGen root so Salsa_utils can import interhuman_utils (which may use InterGen's utils)
_SCRIPT_DIR = Path(__file__).resolve().parent
_INTERGEN_ROOT = _SCRIPT_DIR.parent
if str(_INTERGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_INTERGEN_ROOT))

from Salsa_utils.interhuman_utils import (
    align_follower_to_leader,
    salsa_to_interhuman,
)
from Salsa_utils.salsa_caption import get_salsa_caption


# Defaults: 210 frames, stride 50, 20 fps
DEFAULT_WINDOW_FRAMES = 210
DEFAULT_STRIDE_FRAMES = 50
DEFAULT_POSE_FPS = 20


def _extract_take_id(vid: str) -> str | None:
    if "," in vid:
        base = vid.split(",")[0].strip()
    else:
        base = vid.strip()
    for suffix in ("_leader", "_leader_subject", "_follower", "_follower_subject"):
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)].rstrip("_")
            break
    if not base or "pair" not in base.lower().split("_")[0]:
        return None
    return base


def _get_annotations_for_window(
    dataset_root: Path,
    vid: str,
    start_time: float,
    end_time: float,
    annotation_cache: dict,
) -> dict:
    """Return dict with 'moves', 'errors', 'styling_leader', 'styling_follower' (simplified)."""
    out = {"moves": [], "errors": [], "styling_leader": [], "styling_follower": []}
    take_id = _extract_take_id(vid)
    if not take_id:
        return out
    parts = take_id.split("_")
    if len(parts) < 3 or "pair" not in parts[0].lower():
        return out
    ann_path = dataset_root / "compas3d" / parts[0] / take_id / f"{take_id}.txt"
    if not ann_path.exists():
        return out
    if take_id in annotation_cache:
        annotations = annotation_cache[take_id]
    else:
        annotations = []
        try:
            with open(ann_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or len(line.split("\t")) < 8:
                        continue
                    p = line.split("\t")
                    annotations.append({
                        "start_time": float(p[3]),
                        "end_time": float(p[5]),
                        "description": (p[8] if len(p) > 8 else "").strip(),
                        "type": p[0],
                    })
            annotation_cache[take_id] = annotations
        except Exception:
            annotation_cache[take_id] = []
            return out
    for ann in annotations:
        overlap_start = max(start_time, ann["start_time"])
        overlap_end = min(end_time, ann["end_time"])
        if overlap_end <= overlap_start:
            continue
        overlap_ratio = (overlap_end - overlap_start) / max(1e-6, ann["end_time"] - ann["start_time"])
        if overlap_ratio < 0.3:
            continue
        entry = {"description": ann["description"], "start_time": ann["start_time"], "end_time": ann["end_time"]}
        if ann["type"] == "Errors":
            out["errors"].append(entry)
        else:
            out["moves"].append(entry)
    return out


SALSA_MEAN_FNAME = "global_mean_salsa.npy"
SALSA_STD_FNAME = "global_std_salsa.npy"


def compute_and_save_salsa_global_stats(cache_lmdb_dir: str, data_dir: str) -> None:
    """Compute mean/std (262-d) over all valid frames from both persons.
    Same 262-d stats are applied per person in the model; motion2 is already
    aligned to leader (rigid transform) in the cache."""
    os.makedirs(data_dir, exist_ok=True)
    env = lmdb.open(cache_lmdb_dir, readonly=True, lock=False)
    n_dim = 262
    total_sum = np.zeros(n_dim, dtype=np.float64)
    total_sq = np.zeros(n_dim, dtype=np.float64)
    total_count = 0
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for _key, value in cursor:
            sample = pyarrow.deserialize(value)
            m1 = np.asarray(sample["motion1"], dtype=np.float64)[: int(sample["gt_length"])]
            m2 = np.asarray(sample["motion2"], dtype=np.float64)[: int(sample["gt_length"])]
            for block in (m1, m2):
                total_sum += block.sum(axis=0)
                total_sq += (block ** 2).sum(axis=0)
                total_count += block.shape[0]
    env.close()
    if total_count == 0:
        return
    mean = (total_sum / total_count).astype(np.float32)
    var = np.maximum(total_sq / total_count - mean.astype(np.float64) ** 2, 0.0)
    std = np.sqrt(var).astype(np.float32)
    np.save(os.path.join(data_dir, SALSA_MEAN_FNAME), mean)
    np.save(os.path.join(data_dir, SALSA_STD_FNAME), std)
    print(f"Saved Salsa global mean/std (262-d) to {data_dir}")


def ensure_salsa_global_stats(cache_lmdb_dir: str, data_dir: str) -> None:
    """If global_mean_salsa.npy does not exist, compute from cache and save."""
    mean_path = os.path.join(data_dir, SALSA_MEAN_FNAME)
    if os.path.isfile(mean_path):
        return
    compute_and_save_salsa_global_stats(cache_lmdb_dir, data_dir)


def run(
    source_lmdb_dir: str,
    out_lmdb_dir: str,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    stride_frames: int = DEFAULT_STRIDE_FRAMES,
    pose_fps: int = DEFAULT_POSE_FPS,
    dataset_root: str | None = None,
    use_annotations: bool = False,
) -> None:
    """
    Build cache LMDB from source pair LMDB.

    Each sample: 210-frame window (→ 209 InterHuman frames), raw audio for that window,
    caption, and metadata. Conversion is done on the full window (keypoints → InterHuman)
    not on small chunks.
    """
    dataset_root = Path(dataset_root or os.environ.get("SALSA_DATASET_ROOT", os.environ.get("SALSA_DATA_ROOT", "")))
    if not dataset_root.exists():
        dataset_root = _SCRIPT_DIR.parent.parent.parent.parent / "Dataset"

    env_src = lmdb.open(source_lmdb_dir, readonly=True, lock=False)
    map_size = 1024 * 50 * (1 << 20)  # 50 GB
    env_dst = lmdb.open(out_lmdb_dir, map_size=map_size)

    annotation_cache: dict = {}
    n_out = 0
    first_skip_logged = [True]  # list so we can assign from nested scope

    with env_src.begin(write=False) as txn_src:
        keys = list(txn_src.cursor().iternext(values=False))
    total_videos = len(keys)

    for v in tqdm(keys, desc="Videos"):
        with env_src.begin(write=False) as txn:
            value = txn.get(v)
        if value is None:
            continue
        video = pyarrow.deserialize(value)
        vid = video["vid"]
        clips = video.get("clips", [])
        for clip_idx, clip in enumerate(clips):
            kp_L = clip.get("keypoints3d_L")
            rot_L = clip.get("rotmat_L")
            kp_F = clip.get("keypoints3d_F")
            rot_F = clip.get("rotmat_F")
            if kp_L is None or rot_L is None or kp_F is None or rot_F is None:
                continue
            T = len(kp_L)
            if T < window_frames:
                continue
            num_windows = math.floor((T - window_frames) / stride_frames) + 1
            audio_raw = clip.get("audio_raw")
            audio_sr = clip.get("audio_sr", 24000)
            if audio_raw is not None and hasattr(audio_raw, "shape"):
                if audio_raw.ndim == 1:
                    audio_len = len(audio_raw)
                else:
                    audio_len = audio_raw.shape[-1]
            else:
                audio_len = 0

            for i in range(num_windows):
                start_idx = i * stride_frames
                end_idx = start_idx + window_frames
                if end_idx > T:
                    continue

                start_time = start_idx / pose_fps
                end_time = end_idx / pose_fps

                # Slice keypoints/rotmat for this window (full window → one InterHuman conversion)
                kp_L_w = np.asarray(kp_L[start_idx:end_idx])
                rot_L_w = np.asarray(rot_L[start_idx:end_idx])
                kp_F_w = np.asarray(kp_F[start_idx:end_idx])
                rot_F_w = np.asarray(rot_F[start_idx:end_idx])

                try:
                    motion_L, root_quat_L, root_pos_L = salsa_to_interhuman(
                        kp_L_w, rot_L_w, n_joints=22, rotation_deg=90
                    )
                    motion_F, root_quat_F, root_pos_F = salsa_to_interhuman(
                        kp_F_w, rot_F_w, n_joints=22, rotation_deg=90
                    )
                    motion_F_aligned = align_follower_to_leader(
                        motion_L, motion_F,
                        root_quat_L, root_pos_L,
                        root_quat_F, root_pos_F,
                    )
                except Exception as e:
                    if first_skip_logged[0]:
                        first_skip_logged[0] = False
                        tqdm.write(f"Skip window vid={vid} clip={clip_idx} start={start_idx}: {repr(e)}")
                        tqdm.write(
                            f"  kp_L_w.shape={getattr(kp_L_w, 'shape', None)} "
                            f"rot_L_w.shape={getattr(rot_L_w, 'shape', None)} "
                            f"kp_F_w.shape={getattr(kp_F_w, 'shape', None)} "
                            f"rot_F_w.shape={getattr(rot_F_w, 'shape', None)}"
                        )
                        if DEBUG:
                            traceback.print_exc()
                    else:
                        tqdm.write(f"Skip window vid={vid} clip={clip_idx} start={start_idx}: {e}")
                    continue

                gt_length = motion_L.shape[0]

                # Raw audio for this window: fixed sample count (same as Salsa-Agent) so duration = window_frames/pose_fps
                audio_sample_length = int(window_frames / pose_fps * audio_sr)
                if audio_len > 0 and audio_raw is not None and audio_sample_length > 0:
                    audio_start = int(start_idx / T * audio_len)
                    audio_end = min(audio_start + audio_sample_length, audio_len)
                    raw_slice = np.asarray(audio_raw)[..., audio_start:audio_end].copy()
                    # If slice shorter than target (clip boundary), pad with zeros
                    if raw_slice.shape[-1] < audio_sample_length:
                        pad_width = audio_sample_length - raw_slice.shape[-1]
                        if raw_slice.ndim == 1:
                            raw_audio = np.concatenate([raw_slice, np.zeros(pad_width, dtype=raw_slice.dtype)])
                        else:
                            raw_audio = np.concatenate([raw_slice, np.zeros((raw_slice.shape[0], pad_width), dtype=raw_slice.dtype)], axis=-1)
                    else:
                        raw_audio = raw_slice
                    # Ensure mono and 1D: (samples, channels) -> mean(axis=1); (channels, samples) -> mean(axis=0)
                    if raw_audio.ndim == 2:
                        if raw_audio.shape[-1] <= 2 and raw_audio.shape[0] > raw_audio.shape[-1]:
                            raw_audio = np.mean(raw_audio, axis=1)
                        else:
                            raw_audio = np.mean(raw_audio, axis=0)
                    raw_audio = np.asarray(raw_audio).ravel()
                    # Trim to exact length so playback duration matches motion (no double/repeat)
                    if len(raw_audio) > audio_sample_length:
                        raw_audio = raw_audio[:audio_sample_length].copy()
                else:
                    raw_audio = np.zeros(0, dtype=np.float32)

                annotations = _get_annotations_for_window(
                    dataset_root, vid, start_time, end_time, annotation_cache
                ) if dataset_root.exists() else {"moves": [], "errors": []}
                text = get_salsa_caption(
                    vid,
                    use_annotations=use_annotations,
                    annotations_for_window=annotations,
                    randomize=False,
                )

                name = f"{vid}_c{clip_idx}_w{i}"

                sample = {
                    "name": name,
                    "text": text,
                    "motion1": motion_L.astype(np.float32),
                    "motion2": motion_F_aligned.astype(np.float32),
                    "gt_length": gt_length,
                    "raw_audio": raw_audio,
                    "audio_sr": audio_sr,
                    "metadata": {
                        "vid": vid,
                        "clip_idx": clip_idx,
                        "window_idx": i,
                        "start_frame": start_idx,
                        "end_frame": end_idx,
                        "start_time": start_time,
                        "end_time": end_time,
                        "annotations": annotations,
                    },
                }

                with env_dst.begin(write=True) as txn:
                    key = f"{n_out:010d}".encode("ascii")
                    txn.put(key, pyarrow.serialize(sample).to_buffer())
                n_out += 1

    env_src.close()
    env_dst.sync()
    env_dst.close()
    print(f"Wrote {n_out} samples to {out_lmdb_dir}")

    # Salsa global mean/std (same folder as InterGen's data/global_*.npy)
    data_dir = os.path.join(_INTERGEN_ROOT, "data")
    compute_and_save_salsa_global_stats(out_lmdb_dir, data_dir)


def main():
    ap = argparse.ArgumentParser(description="Build Salsa InterGen cache from source pair LMDB")
    ap.add_argument("--source_lmdb", type=str, required=True, help="Source pair LMDB directory")
    ap.add_argument("--out_lmdb", type=str, required=True, help="Output cache LMDB directory")
    ap.add_argument("--window_frames", type=int, default=DEFAULT_WINDOW_FRAMES, help="Window length in frames")
    ap.add_argument("--stride_frames", type=int, default=DEFAULT_STRIDE_FRAMES, help="Stride between windows")
    ap.add_argument("--pose_fps", type=int, default=DEFAULT_POSE_FPS, help="Pose FPS")
    ap.add_argument("--dataset_root", type=str, default="", help="Dataset root for annotations")
    ap.add_argument("--use_annotations", action="store_true", help="Use annotations for caption when available")
    ap.add_argument("--debug", action="store_true", help="Print full traceback and shapes on first skip")
    args = ap.parse_args()
    if args.debug:
        global DEBUG
        DEBUG = True
    run(
        source_lmdb_dir=args.source_lmdb,
        out_lmdb_dir=args.out_lmdb,
        window_frames=args.window_frames,
        stride_frames=args.stride_frames,
        pose_fps=args.pose_fps,
        dataset_root=args.dataset_root or None,
        use_annotations=args.use_annotations,
    )


if __name__ == "__main__":
    main()
