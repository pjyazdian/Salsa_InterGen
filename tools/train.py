import sys
sys.path.append(sys.path[0] + r"/../")
import torch
import lightning.pytorch as pl
import torch.optim as optim
from collections import OrderedDict
from datasets import DataModule
from configs import get_config
import os
from os.path import join as pjoin
import wandb
from models import *

os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'nccl'
from lightning.pytorch.strategies import DDPStrategy
torch.set_float32_matmul_precision('medium')

class LitTrainModel(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        # cfg init
        self.cfg = cfg
        self.mode = cfg.TRAIN.MODE

        self.automatic_optimization = False

        self.save_root = pjoin(self.cfg.GENERAL.CHECKPOINT, self.cfg.GENERAL.EXP_NAME)
        self.model_dir = pjoin(self.save_root, 'model')
        self.meta_dir = pjoin(self.save_root, 'meta')
        self.log_dir = pjoin(self.save_root, 'log')

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.use_wandb = getattr(self.cfg.TRAIN, 'USE_WANDB', True)
        if not self.use_wandb:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = None

        self.model = model

    def _configure_optim(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=float(self.cfg.TRAIN.LR), weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=10, max_iters=self.cfg.TRAIN.EPOCH, verbose=True)
        return [optimizer], [scheduler]

    def configure_optimizers(self):
        return self._configure_optim()

    def forward(self, batch_data):
        name, text, motion1, motion2, motion_lens = batch_data
        motion1 = motion1.detach().float()  # .to(self.device)
        motion2 = motion2.detach().float()  # .to(self.device)
        motions = torch.cat([motion1, motion2], dim=-1)

        B, T = motion1.shape[:2]

        batch = OrderedDict({})
        batch["text"] = text
        batch["motions"] = motions.reshape(B, T, -1).type(torch.float32)
        batch["motion_lens"] = motion_lens.long()

        loss, loss_logs = self.model(batch)
        return loss, loss_logs

    def on_train_start(self):
        self.rank = 0
        self.world_size = 1
        self.start_time = time.time()
        self.it = self.cfg.TRAIN.LAST_ITER if self.cfg.TRAIN.LAST_ITER else 0
        self.epoch = self.cfg.TRAIN.LAST_EPOCH if self.cfg.TRAIN.LAST_EPOCH else 0
        self.logs = OrderedDict()
        self.epoch_logs = OrderedDict()
        self.epoch_logs_count = 0
        # Init wandb only on rank 0 so metrics are recorded (DDP-safe)
        if self.use_wandb and self.trainer.global_rank == 0:
            wandb.init(project="intergen", name=self.cfg.GENERAL.EXP_NAME, reinit=True)
            wandb.define_metric("train/*", step_metric="iter_step")
            wandb.define_metric("loss", step_metric="iter_step")
            wandb.define_metric("epoch/*", step_metric="epoch_step")


    def training_step(self, batch, batch_idx):
        loss, loss_logs = self.forward(batch)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        opt.step()

        return {"loss": loss,
            "loss_logs": loss_logs}


    def on_train_batch_end(self, outputs, batch, batch_idx):
        if outputs.get('skip_batch') or not outputs.get('loss_logs'):
            return
        for k, v in outputs['loss_logs'].items():
            if k not in self.logs:
                self.logs[k] = v.item()
            else:
                self.logs[k] += v.item()

        self.it += 1
        if self.it % self.cfg.TRAIN.LOG_STEPS == 0 and self.trainer.global_rank == 0:
            mean_loss = OrderedDict({})
            for tag, value in self.logs.items():
                mean_loss[tag] = value / self.cfg.TRAIN.LOG_STEPS
            if self.use_wandb:
                # Train section: per-iteration metrics, x-axis = iter_step
                wandb_dict = {f"train/{k}": v for k, v in mean_loss.items()}
                wandb_dict["loss"] = mean_loss.get("total", 0.0)
                wandb_dict["iter_step"] = self.it
                wandb.log(wandb_dict)
                # Accumulate for epoch-level log (once per epoch in on_train_epoch_end)
                for k, v in mean_loss.items():
                    self.epoch_logs[k] = self.epoch_logs.get(k, 0.0) + v
                self.epoch_logs_count += 1
            else:
                for tag, value in mean_loss.items():
                    self.writer.add_scalar(tag, value, self.it)
            self.logs = OrderedDict()
            print_current_loss(self.start_time, self.it, mean_loss,
                               self.trainer.current_epoch,
                               inner_iter=batch_idx,
                               lr=self.trainer.optimizers[0].param_groups[0]['lr'])



    def on_train_epoch_end(self):
        # Epoch section: log once per epoch (averages), x-axis = epoch_step
        if self.use_wandb and self.trainer.global_rank == 0 and self.epoch_logs_count > 0:
            epoch_mean = OrderedDict((k, v / self.epoch_logs_count) for k, v in self.epoch_logs.items())
            wandb_dict_epoch = {f"epoch/{k}": v for k, v in epoch_mean.items()}
            wandb_dict_epoch["epoch/loss"] = epoch_mean.get("total", 0.0)
            wandb_dict_epoch["epoch_step"] = self.trainer.current_epoch
            wandb.log(wandb_dict_epoch)
            self.epoch_logs = OrderedDict()
            self.epoch_logs_count = 0
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()


    def save(self, file_name):
        state = {}
        try:
            state['model'] = self.model.module.state_dict()
        except:
            state['model'] = self.model.state_dict()
        torch.save(state, file_name, _use_new_zipfile_serialization=False)
        return


def build_models(cfg):
    if cfg.NAME == "InterGen":
        model = InterGen(cfg)
    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="interhuman", choices=["interhuman", "salsa"],
                        help="Dataset to train on: interhuman (default) or salsa")
    args, _ = parser.parse_known_args()

    print(os.getcwd())
    model_cfg = get_config("configs/model.yaml")
    train_cfg = get_config("configs/train.yaml")
    if args.dataset == "salsa":
        data_cfg = get_config("configs/datasets_salsa.yaml").salsa_train
        cache_dir = data_cfg.DATA_ROOT if os.path.isabs(data_cfg.DATA_ROOT) else os.path.abspath(data_cfg.DATA_ROOT)
        data_dir = os.path.join(os.getcwd(), "data")
        from Salsa_utils.salsa_intergen_cache import ensure_salsa_global_stats
        ensure_salsa_global_stats(cache_dir, data_dir)
        os.environ["INTERGEN_GLOBAL_MEAN"] = os.path.join(data_dir, "global_mean_salsa.npy")
        os.environ["INTERGEN_GLOBAL_STD"] = os.path.join(data_dir, "global_std_salsa.npy")
    else:
        data_cfg = get_config("configs/datasets.yaml").interhuman

    datamodule = DataModule(data_cfg, train_cfg.TRAIN.BATCH_SIZE, train_cfg.TRAIN.NUM_WORKERS)
    model = build_models(model_cfg)


    if train_cfg.TRAIN.RESUME:
        ckpt = torch.load(train_cfg.TRAIN.RESUME, map_location="cpu")
        for k in list(ckpt["state_dict"].keys()):
            if "model" in k:
                ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print("checkpoint state loaded!")
    litmodel = LitTrainModel(model, train_cfg)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=litmodel.model_dir,
                                                       every_n_epochs=train_cfg.TRAIN.SAVE_EPOCH)
    trainer = pl.Trainer(
        default_root_dir=litmodel.model_dir,
        devices="auto", accelerator='gpu',
        max_epochs=train_cfg.TRAIN.EPOCH,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision=32,
        callbacks=[checkpoint_callback],

    )

    trainer.fit(model=litmodel, datamodule=datamodule)
