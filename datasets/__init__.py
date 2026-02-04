import os
import lightning.pytorch as pl
import torch
from .interhuman import InterHumanDataset
from datasets.evaluator import (
    EvaluatorModelWrapper,
    EvaluationDataset,
    get_dataset_motion_loader,
    get_motion_loader)
# from .dataloader import build_dataloader

__all__ = [
    'InterHumanDataset', 'EvaluationDataset',
    'get_dataset_motion_loader', 'get_motion_loader']

def build_loader(cfg, data_cfg):
    # setup data
    if data_cfg.NAME == "interhuman":
        train_dataset = InterHumanDataset(data_cfg)
    elif data_cfg.NAME == "salsa_intergen":
        from Salsa_utils.salsa_intergen_dataset import SalsaInterGenDataset, collate_salsa_intergen_for_training
        lmdb_dir = data_cfg.DATA_ROOT if os.path.isabs(data_cfg.DATA_ROOT) else os.path.abspath(data_cfg.DATA_ROOT)
        train_dataset = SalsaInterGenDataset(
            lmdb_dir=lmdb_dir,
            max_gt_length=getattr(data_cfg, "MAX_GT_LENGTH", 300),
            min_gt_length=getattr(data_cfg, "MIN_GT_LENGTH", 15),
            swap_person=True,
            split=getattr(data_cfg, "MODE", "train"),
            split_dir=getattr(data_cfg, "SPLIT_DIR", None),
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.BATCH_SIZE,
            num_workers=1,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_salsa_intergen_for_training,
        )
    else:
        raise NotImplementedError

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=1,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
        )

    return loader

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg, batch_size, num_workers):
        """
        Initialize LightningDataModule for ProHMR training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
            dataset_cfg (CfgNode): Dataset configuration file
        """
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._collate_fn = None

    def setup(self, stage = None):
        """
        Create train and validation datasets
        """
        if self.cfg.NAME == "interhuman":
            self.train_dataset = InterHumanDataset(self.cfg)
            self._collate_fn = None
        elif self.cfg.NAME == "salsa_intergen":
            from Salsa_utils.salsa_intergen_dataset import SalsaInterGenDataset, collate_salsa_intergen_for_training
            lmdb_dir = self.cfg.DATA_ROOT if os.path.isabs(self.cfg.DATA_ROOT) else os.path.abspath(self.cfg.DATA_ROOT)
            self.train_dataset = SalsaInterGenDataset(
                lmdb_dir=lmdb_dir,
                max_gt_length=getattr(self.cfg, "MAX_GT_LENGTH", 300),
                min_gt_length=getattr(self.cfg, "MIN_GT_LENGTH", 15),
                swap_person=True,
                split=getattr(self.cfg, "MODE", "train"),
                split_dir=getattr(self.cfg, "SPLIT_DIR", None),
            )
            self._collate_fn = collate_salsa_intergen_for_training
        else:
            raise NotImplementedError(f"Dataset NAME={getattr(self.cfg, 'NAME', None)} not implemented")

    def train_dataloader(self):
        """
        Return train dataloader
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
            collate_fn=self._collate_fn,
            )
