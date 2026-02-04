"""
PyTorch Dataset for Salsa InterGen cache.

Reads the LMDB produced by salsa_intergen_cache.py and returns dict samples with
motion1, motion2, text, raw_audio, metadata, etc., for training or downstream use.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import pyarrow
import torch
from torch.utils.data import Dataset


class SalsaInterGenDataset(Dataset):
    """
    Dataset over Salsa InterGen cache LMDB.

    Each sample is a dict with:
        - name: str
        - text: str
        - motion1: (T, 262) float32
        - motion2: (T, 262) float32
        - gt_length: int (actual length before padding)
        - raw_audio: (1, N) or (N,) float32
        - audio_sr: int
        - metadata: dict (vid, clip_idx, start_frame, end_frame, annotations, ...)

    Motions are padded to max_gt_length so batches can be stacked. Use gt_length
    for masking or loss.
    """

    def __init__(
        self,
        lmdb_dir: str,
        max_gt_length: int = 300,
        min_gt_length: int = 15,
        swap_person: bool = True,
        split: str | None = None,
        split_dir: str | None = None,
    ):
        """
        Args:
            lmdb_dir: Path to cache LMDB directory (from salsa_intergen_cache.py).
            max_gt_length: Pad motions to this length.
            min_gt_length: Skip samples with gt_length < this (if any).
            swap_person: Randomly swap motion1/motion2 in __getitem__.
            split: 'train' | 'val' | 'test' to filter by split file.
            split_dir: Directory containing train.txt, val.txt, test.txt (names one per line).
                      If None, uses all samples in LMDB.
        """
        self.lmdb_dir = Path(lmdb_dir)
        self.max_gt_length = max_gt_length
        self.min_gt_length = min_gt_length
        self.swap_person = swap_person
        self.split = split
        self.split_dir = Path(split_dir) if split_dir else None

        self.env = lmdb.open(
            str(self.lmdb_dir),
            readonly=True,
            lock=False,
        )
        with self.env.begin(write=False) as txn:
            self.n_samples = txn.stat()["entries"]

        self.valid_indices: list[int] | None = None
        if split and split_dir and self.split_dir.exists():
            split_file = self.split_dir / f"{split}.txt"
            if split_file.exists():
                with open(split_file) as f:
                    valid_names = {line.strip() for line in f if line.strip()}
                self.valid_indices = []
                with self.env.begin(write=False) as txn:
                    cursor = txn.cursor()
                    for key, value in cursor:
                        sample = pyarrow.deserialize(value)
                        if sample.get("name", "") in valid_names:
                            idx = int(key.decode("ascii"))
                            self.valid_indices.append(idx)
                self.valid_indices.sort()
                self.n_samples = len(self.valid_indices)

    def _get_index(self, item: int) -> int:
        if self.valid_indices is not None:
            return self.valid_indices[item % len(self.valid_indices)]
        return item % self.n_samples

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, item: int) -> dict[str, Any]:
        idx = self._get_index(item)
        key = f"{idx:010d}".encode("ascii")
        with self.env.begin(write=False) as txn:
            value = txn.get(key)
        if value is None:
            raise KeyError(f"Missing key {key} in LMDB")
        sample = pyarrow.deserialize(value)

        name = sample["name"]
        text = sample["text"].strip() if isinstance(sample["text"], str) else str(sample["text"])
        motion1 = np.asarray(sample["motion1"], dtype=np.float32)
        motion2 = np.asarray(sample["motion2"], dtype=np.float32)
        gt_length = int(sample["gt_length"])
        raw_audio = sample.get("raw_audio")
        if raw_audio is not None:
            raw_audio = np.asarray(raw_audio, dtype=np.float32).ravel()
        else:
            raw_audio = np.zeros(0, dtype=np.float32)
        audio_sr = int(sample.get("audio_sr", 24000))
        metadata = sample.get("metadata", {})

        if gt_length < self.min_gt_length:
            # Pad up to min_gt_length so we don't break batching; still mark gt_length
            pad_len = self.min_gt_length - gt_length
            motion1 = np.concatenate([motion1, np.zeros((pad_len, motion1.shape[1]), dtype=np.float32)], axis=0)
            motion2 = np.concatenate([motion2, np.zeros((pad_len, motion2.shape[1]), dtype=np.float32)], axis=0)
            gt_length = self.min_gt_length

        if self.swap_person and random.random() > 0.5:
            motion1, motion2 = motion2, motion1

        if gt_length < self.max_gt_length:
            pad_len = self.max_gt_length - gt_length
            motion1 = np.concatenate([motion1, np.zeros((pad_len, motion1.shape[1]), dtype=np.float32)], axis=0)
            motion2 = np.concatenate([motion2, np.zeros((pad_len, motion2.shape[1]), dtype=np.float32)], axis=0)

        # Trim if somehow over (shouldn't happen for our cache)
        if motion1.shape[0] > self.max_gt_length:
            motion1 = motion1[: self.max_gt_length]
            motion2 = motion2[: self.max_gt_length]
            gt_length = self.max_gt_length

        return {
            "name": name,
            "text": text,
            "motion1": motion1,
            "motion2": motion2,
            "motions": np.concatenate([motion1, motion2], axis=-1),  # (T, 524) for InterGen
            "gt_length": gt_length,
            "motion_lens": gt_length,
            "raw_audio": raw_audio,
            "audio_sr": audio_sr,
            "metadata": metadata,
        }

    def close(self):
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
            self.env = None

    def __del__(self):
        self.close()


def collate_salsa_intergen(batch: list[dict]) -> dict[str, Any]:
    """
    Collate list of sample dicts into a batch dict.

    Converts numpy to torch and stacks. Keys: name, text, motion1, motion2, motions,
    gt_length, motion_lens, raw_audio, audio_sr, metadata (list).
    """
    out: dict[str, Any] = {}
    out["name"] = [s["name"] for s in batch]
    out["text"] = [s["text"] for s in batch]
    out["motion1"] = torch.from_numpy(np.stack([s["motion1"] for s in batch])).float()
    out["motion2"] = torch.from_numpy(np.stack([s["motion2"] for s in batch])).float()
    out["motions"] = torch.from_numpy(np.stack([s["motions"] for s in batch])).float()
    out["gt_length"] = torch.tensor([s["gt_length"] for s in batch], dtype=torch.long)
    out["motion_lens"] = out["gt_length"]
    # Raw audio can have different lengths; keep as list or pad to max
    raw_audios = [s["raw_audio"] for s in batch]
    max_audio = max(a.size for a in raw_audios)
    if max_audio == 0:
        out["raw_audio"] = torch.zeros(len(batch), 1, dtype=torch.float32)
    else:
        padded = np.zeros((len(batch), max_audio), dtype=np.float32)
        for i, a in enumerate(raw_audios):
            padded[i, : len(a)] = a
        out["raw_audio"] = torch.from_numpy(padded).float()
    out["audio_sr"] = batch[0].get("audio_sr", 24000)
    out["metadata"] = [s.get("metadata", {}) for s in batch]
    return out


def collate_salsa_intergen_for_training(batch: list[dict]) -> tuple:
    """
    Collate Salsa dict batch into the tuple format expected by InterGen training:
    (names, texts, motion1, motion2, motion_lens) — same as default_collate on
    InterHumanDataset's (name, text, gt_motion1, gt_motion2, gt_length).
    Use as DataLoader collate_fn when training with SalsaInterGenDataset.
    """
    names = [s["name"] for s in batch]
    texts = [s["text"] for s in batch]
    motion1 = torch.from_numpy(np.stack([s["motion1"] for s in batch])).float()
    motion2 = torch.from_numpy(np.stack([s["motion2"] for s in batch])).float()
    motion_lens = torch.tensor([s["gt_length"] for s in batch], dtype=torch.long)
    return (names, texts, motion1, motion2, motion_lens)
