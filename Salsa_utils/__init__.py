# Salsa_utils: Salsa–InterGen integration and visualization

from .interhuman_utils import (
    align_follower_to_leader,
    process_motion_interhuman,
    rigid_transform,
    salsa_to_interhuman,
)
from .salsa_caption import get_salsa_caption, PAIR2LEVEL, SALSA_CAPTIONS

try:
    from .salsa_intergen_dataset import (
        SalsaInterGenDataset,
        collate_salsa_intergen,
    )
    _dataset_available = True
except ImportError:
    SalsaInterGenDataset = None  # type: ignore[misc, assignment]
    collate_salsa_intergen = None  # type: ignore[misc, assignment]
    _dataset_available = False

__all__ = [
    "align_follower_to_leader",
    "process_motion_interhuman",
    "rigid_transform",
    "salsa_to_interhuman",
    "get_salsa_caption",
    "PAIR2LEVEL",
    "SALSA_CAPTIONS",
    "SalsaInterGenDataset",
    "collate_salsa_intergen",
]
