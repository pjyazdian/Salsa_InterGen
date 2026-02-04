"""
InterHuman motion conversion utilities for Salsa → InterGen.

Converts Salsa keypoints3d + rotmat to InterHuman (T-1, 262) representation and
aligns follower to leader via rigid_transform. Uses InterGen's or local implementations
of process_motion_interhuman and rigid_transform so we do not depend on in2IN when
running from the InterGen codebase.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Tuple

import numpy as np

# Joint indices for InterHuman (22 joints): same as InterGen/in2IN
FACE_JOINT_INDX = [2, 1, 17, 16]  # r_hip, l_hip, sdr_r, sdr_l
FID_L = [7, 10]
FID_R = [8, 11]

# Same as InterGen/in2IN: front view flip
TRANS_MATRIX = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float32
)


def _get_quaternion_and_rigid():
    """Import quaternion helpers and rigid_transform from InterGen or in2IN or use local."""
    # Prefer InterGen (when running from Baselines/Salsa_InterGen)
    try:
        intergen_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if intergen_root not in sys.path:
            sys.path.insert(0, intergen_root)
        from common.quaternion import qbetween_np, qinv_np, qrot_np, qmul_np
        from utils.utils import rigid_transform as _rigid_transform
        return qbetween_np, qinv_np, qrot_np, qmul_np, _rigid_transform
    except Exception:
        pass
    # Fallback: in2IN
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        in2in_root = os.path.join(project_root, "Download", "in2IN")
        if in2in_root not in sys.path:
            sys.path.insert(0, in2in_root)
        from in2in.utils.quaternion import qbetween_np, qinv_np, qrot_np, qmul_np
        from in2in.utils.utils import rigid_transform as _rigid_transform
        return qbetween_np, qinv_np, qrot_np, qmul_np, _rigid_transform
    except Exception:
        pass
    raise ImportError(
        "Neither InterGen (common.quaternion, utils.utils) nor in2IN could be imported. "
        "Run from Baselines/Salsa_InterGen or ensure Download/in2IN is available."
    )


# Lazy bindings (set on first use)
_qbetween_np = _qinv_np = _qrot_np = _qmul_np = _rigid_transform = None


def _ensure_quat():
    global _qbetween_np, _qinv_np, _qrot_np, _qmul_np, _rigid_transform
    if _rigid_transform is None:
        _qbetween_np, _qinv_np, _qrot_np, _qmul_np, _rigid_transform = _get_quaternion_and_rigid()


def process_motion_interhuman(
    motion: np.ndarray,
    feet_thre: float,
    prev_frames: int,
    n_joints: int,
    flip: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process raw (T, 192) motion to InterHuman (T-1, 262).
    Ported from in2IN for use without in2IN dependency when InterGen is available.

    Args:
        motion: (T, 192) = positions (66) + rotations (126)
        feet_thre: threshold for foot contact
        prev_frames: frame index for initial root (usually 0)
        n_joints: 22
        flip: apply TRANS_MATRIX (front view)

    Returns:
        data: (T-1, 262)
        root_quat_init: (4,)
        root_pose_init_xz: (1, 3) - XZ used for origin
    """
    _ensure_quat()
    positions = motion[:, : n_joints * 3].reshape(-1, n_joints, 3).astype(np.float64)
    rotations = motion[:, n_joints * 3 :].astype(np.float32)

    if flip:
        positions = np.einsum("mn,tjn->tjm", TRANS_MATRIX, positions)

    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    root_pos_init = positions[prev_frames]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    r_hip, l_hip, sdr_r, sdr_l = FACE_JOINT_INDX
    across = root_pos_init[r_hip] - root_pos_init[l_hip]
    across = across / (np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis] + 1e-8)
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / (np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis] + 1e-8)
    target = np.array([[0, 0, 1]])
    root_quat_init = _qbetween_np(forward_init, target)
    root_quat_init_for_all = np.ones(positions.shape[:-1] + (4,), dtype=np.float64) * root_quat_init
    positions = _qrot_np(root_quat_init_for_all, positions)

    def foot_detect(positions, thres):
        velfactor = np.array([thres, thres])
        heightfactor = np.array([0.12, 0.05])
        feet_l_x = (positions[1:, FID_L, 0] - positions[:-1, FID_L, 0]) ** 2
        feet_l_y = (positions[1:, FID_L, 1] - positions[:-1, FID_L, 1]) ** 2
        feet_l_z = (positions[1:, FID_L, 2] - positions[:-1, FID_L, 2]) ** 2
        feet_l_h = positions[:-1, FID_L, 1]
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float32)
        feet_r_x = (positions[1:, FID_R, 0] - positions[:-1, FID_R, 0]) ** 2
        feet_r_y = (positions[1:, FID_R, 1] - positions[:-1, FID_R, 1]) ** 2
        feet_r_z = (positions[1:, FID_R, 2] - positions[:-1, FID_R, 2]) ** 2
        feet_r_h = positions[:-1, FID_R, 1]
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float32)
        return feet_l, feet_r

    feet_l, feet_r = foot_detect(positions, feet_thre)

    joint_positions = positions.reshape(len(positions), -1)
    joint_vels = (positions[1:] - positions[:-1]).reshape(len(positions) - 1, -1)
    data = joint_positions[:-1]
    data = np.concatenate([data, joint_vels], axis=-1)
    data = np.concatenate([data, rotations[:-1]], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)
    return data.astype(np.float32), root_quat_init.astype(np.float32), root_pose_init_xz[None].astype(np.float32)


def rigid_transform(relative: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    Align person2 motion to person1's frame using relative [angle, x, z].
    Same as InterGen utils.utils.rigid_transform.
    """
    _ensure_quat()
    return _rigid_transform(relative, data)


def rotate_keypoints_deg_x(keypoints3d: np.ndarray, rotation_deg: float) -> np.ndarray:
    """Rotate 3D keypoints by rotation_deg degrees around X-axis (e.g. +90 for front view)."""
    if rotation_deg == 0:
        return np.asarray(keypoints3d, dtype=np.float64)
    orig_shape = np.array(keypoints3d).shape
    kp = np.asarray(keypoints3d, dtype=np.float64)
    if kp.ndim == 2:
        kp = kp.reshape(kp.shape[0], -1, 3)
    T = kp.shape[0]
    theta_rad = math.pi * rotation_deg / 180.0
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    rot_x = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    out = np.einsum("mn,tjn->tjm", rot_x, kp)
    if len(orig_shape) == 2:
        out = out.reshape(orig_shape)
    return out


def salsa_to_interhuman(
    keypoints3d: np.ndarray,
    rotmat: np.ndarray,
    n_joints: int = 22,
    rotation_deg: float = 90,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Salsa keypoints3d and rotmat to InterHuman (T-1, 262).

    Uses full-window conversion: one call for the entire sequence (e.g. 210 frames → 209, 262).
    Do not convert small chunks and concatenate.

    Args:
        keypoints3d: (T, 22, 3)
        rotmat: (T, 498) [trans (3) | flattened rotmats (55*9)]
        n_joints: 22
        rotation_deg: +90 to reverse Salsa preprocessing -90° rotation

    Returns:
        motion_interhuman: (T-1, 262)
        root_quat_init: (4,) or (T-1, 4) from process_motion_interhuman
        root_pos_init: (3,) or (1, 3) - XZ origin
    """
    T = keypoints3d.shape[0]
    keypoints3d = np.asarray(keypoints3d, dtype=np.float64)
    if keypoints3d.ndim == 2:
        if keypoints3d.shape[1] == 66:
            keypoints3d = keypoints3d.reshape(T, 22, 3)
        else:
            keypoints3d = keypoints3d.reshape(T, -1, 3)
    if keypoints3d.shape[1] != 22:
        raise ValueError(f"keypoints3d must have 22 joints, got shape {keypoints3d.shape}")
    if rotation_deg != 0:
        keypoints3d = rotate_keypoints_deg_x(keypoints3d, rotation_deg)

    positions = keypoints3d.reshape(T, -1).astype(np.float32)
    rotmat = np.asarray(rotmat, dtype=np.float32)
    # Support (T, 498) = trans(3) + 55*9, or (T, 495) = 55*9 only
    if rotmat.shape[1] == 498:
        rotmats_flat = rotmat[:, 3:]
    elif rotmat.shape[1] == 495:
        rotmats_flat = rotmat
    else:
        raise ValueError(f"rotmat last dim must be 498 or 495, got {rotmat.shape}")
    rotmats = rotmats_flat.reshape(T, 55, 3, 3)
    body_rotmats = rotmats[:, 1:22, :, :]
    rotations_6d = body_rotmats[:, :, :, :2].reshape(T, 21, 6).reshape(T, -1)
    motion_raw = np.concatenate([positions, rotations_6d], axis=-1)

    data, root_quat_init, root_pose_init_xz = process_motion_interhuman(
        motion_raw, 0.001, 0, n_joints=n_joints
    )
    root_pos_init = root_pose_init_xz[0]
    if root_quat_init.ndim == 2:
        root_quat_init = root_quat_init[0]
    return data, root_quat_init, root_pos_init


def align_follower_to_leader(
    motion_leader: np.ndarray,
    motion_follower: np.ndarray,
    root_quat_L: np.ndarray,
    root_pos_L: np.ndarray,
    root_quat_F: np.ndarray,
    root_pos_F: np.ndarray,
) -> np.ndarray:
    """
    Compute relative transform from frame-0 roots and apply rigid_transform to follower.
    Same logic as InterGen datasets/interhuman.py.
    """
    _ensure_quat()
    if root_quat_L.ndim > 1:
        root_quat_L = root_quat_L[0]
    if root_pos_L.ndim > 1:
        root_pos_L = root_pos_L[0]
    if root_quat_F.ndim > 1:
        root_quat_F = root_quat_F[0]
    if root_pos_F.ndim > 1:
        root_pos_F = root_pos_F[0]

    r_relative = _qmul_np(root_quat_F, _qinv_np(root_quat_L))
    angle = np.arctan2(r_relative[2:3], r_relative[0:1])
    # qrot_np expects matching batch dims: (1, 4) and (1, 3)
    q_L = np.asarray(root_quat_L).reshape(1, 4)
    v_rel = (root_pos_F - root_pos_L).reshape(1, 3)
    xz = _qrot_np(q_L, v_rel)[0, [0, 2]]
    relative = np.concatenate([angle.ravel(), xz]).astype(np.float32)
    return rigid_transform(relative, motion_follower.copy())
