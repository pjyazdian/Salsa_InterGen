"""
InterGen visualization helpers using the InterGen codebase.
All visualization (plot_3d_motion, kinematic chain, inference) comes from InterGen.
Run from InterGen root so imports resolve; this module is used by the Gradio app.
"""
import os
import sys
import tempfile
import numpy as np

# Ensure InterGen root is on path and cwd when this module is imported from the Gradio app
_VIS_ROOT = os.path.dirname(os.path.abspath(__file__))
_INTERGEN_ROOT = os.path.abspath(os.path.join(_VIS_ROOT, "../.."))
if _INTERGEN_ROOT not in sys.path:
    sys.path.insert(0, _INTERGEN_ROOT)

# Use InterGen's plot and paramUtil
from utils.plot_script import plot_3d_motion
from utils import paramUtil

KINEMATIC_CHAIN = paramUtil.t2m_kinematic_chain


def motion_262_to_joints(motion_262: np.ndarray, length: int = None) -> np.ndarray:
    """
    Extract (T, 22, 3) joint positions from (T, 262) InterGen motion.
    Uses same layout as InterGen: first 22*3 dims are positions.
    """
    if length is not None:
        motion_262 = motion_262[:length]
    T = motion_262.shape[0]
    positions = motion_262[:, : 22 * 3].reshape(T, 22, 3)
    return positions.astype(np.float32)


def plot_gt_motion_to_file(motion1: np.ndarray, motion2: np.ndarray, gt_length: int, caption: str, fps: int = 30) -> str:
    """
    Plot two-person ground-truth motion to a temporary MP4 file using InterGen's plot_3d_motion.
    motion1, motion2: (T, 262) each; gt_length: actual length (rest may be padding).
    Returns path to the saved MP4 file.
    """
    joints1 = motion_262_to_joints(motion1, gt_length)
    joints2 = motion_262_to_joints(motion2, gt_length)
    mp_joints = [joints1, joints2]
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        plot_3d_motion(path, KINEMATIC_CHAIN, mp_joints, title=caption, figsize=(10, 10), fps=fps, radius=4)
    except Exception:
        os.unlink(path)
        raise
    return path


def plot_generated_motion_to_file(mp_data_list, caption: str, fps: int = 30) -> str:
    """
    Plot generated two-person motion to a temporary MP4 file.
    mp_data_list: list of two arrays, each (T, 22, 3) joints (as returned by inference)
                  or (T, 262) from which joints are extracted.
    """
    mp_joints = []
    for data in mp_data_list:
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            joint = arr
        else:
            joint = arr[:, : 22 * 3].reshape(-1, 22, 3)
        mp_joints.append(joint)
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        plot_3d_motion(path, KINEMATIC_CHAIN, mp_joints, title=caption, figsize=(10, 10), fps=fps, radius=4)
    except Exception:
        os.unlink(path)
        raise
    return path
