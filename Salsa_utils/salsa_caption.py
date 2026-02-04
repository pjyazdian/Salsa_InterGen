"""
Salsa text captions for InterGen samples.

Level-based captions (pair → proficiency) and optional annotation-based descriptions.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

# Pair → proficiency (Dataset README)
PAIR2LEVEL = {
    "pair1": "beginner",
    "pair2": "intermediate",
    "pair3": "beginner",
    "pair4": "intermediate",
    "pair5": "professional",
    "pair6": "intermediate",
    "pair7": "professional",
    "pair8": "beginner",
    "pair9": "professional",
}

SALSA_CAPTIONS = {
    "beginner": [
        "A beginner salsa dancer practices simple steps with careful timing.",
        "A novice salsa dancer moves cautiously to the rhythm.",
        "A beginner salsa dancer performs basic footwork with focused effort.",
        "A new salsa dancer follows the beat with steady and controlled movements.",
        "A first-time salsa dancer attempts a slow and structured routine.",
    ],
    "intermediate": [
        "An intermediate salsa dancer combines footwork and turns with growing confidence.",
        "A mid-level salsa dancer executes a balanced and expressive routine.",
        "An intermediate dancer performs with more rhythm and body coordination.",
        "A salsa dancer at intermediate level adds flair while maintaining structure.",
        "An intermediate-level salsa dancer blends technical steps with smoother transitions.",
    ],
    "professional": [
        "A professional salsa dancer delivers a dynamic and polished performance.",
        "A skilled salsa dancer flows through complex moves with ease.",
        "A professional dancer commands the floor with sharp and expressive motion.",
        "A seasoned salsa dancer performs an intricate routine with confidence.",
        "An expert salsa dancer dazzles with swift, precise, and rhythmic movements.",
    ],
}


def extract_take_id_from_vid(vid: str) -> Optional[str]:
    """Vid e.g. 'Pair1_song1_take1_leader,Pair1_song1_take1_follower' → 'Pair1_song1_take1'."""
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
    parts = base.split("_")
    if len(parts) < 3:
        return None
    return base


def get_salsa_caption(
    vid: str,
    use_annotations: bool = False,
    annotations_for_window: Optional[dict] = None,
    randomize: bool = True,
) -> str:
    """
    Return a text caption for a Salsa sample.

    Args:
        vid: Video id (e.g. 'Pair1_song1_take1_leader,Pair1_song1_take1_follower').
        use_annotations: If True and annotations_for_window provided, build from moves/errors.
        annotations_for_window: Dict with 'moves', 'errors', etc. (from cache or loader).
        randomize: If True, pick random caption from level list.

    Returns:
        Caption string.
    """
    take_id = extract_take_id_from_vid(vid)
    pair_key = "pair1"
    if take_id:
        parts = take_id.split("_")
        if parts:
            pair_key = parts[0].lower()
    level = PAIR2LEVEL.get(pair_key, "intermediate")

    if use_annotations and annotations_for_window:
        moves = annotations_for_window.get("moves", [])
        errors = annotations_for_window.get("errors", [])
        if moves or errors:
            parts = []
            if moves:
                descs = [m.get("description", "") for m in moves if m.get("description")]
                if descs:
                    parts.append(" ".join(descs[:3]))
            if errors:
                descs = [e.get("description", "") for e in errors if e.get("description")]
                if descs:
                    parts.append("Errors: " + " ".join(descs[:2]))
            if parts:
                return " ".join(parts).strip() or level

    list_captions = SALSA_CAPTIONS.get(level, SALSA_CAPTIONS["intermediate"])
    if randomize:
        return random.choice(list_captions)
    return list_captions[0]
