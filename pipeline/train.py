import os
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as L
import torch
import zarr
from CleanDiffuser.cleandiffuser.diffusion import ContinuousRectifiedFlow
from CleanDiffuser.cleandiffuser.nn_diffusion import DiT1d
from CleanDiffuser.cleandiffuser.utils import MinMaxNormalizer
from pytorch_lightning.callbacks import ModelCheckpoint
from termcolor import cprint
from torchvision.transforms.v2 import Normalize, Resize

from dataset.xarm_dataset import xArmDataset

from CleanDiffuser.cleandiffuser.nn_condition import IdentityCondition
import torch.nn as nn
from typing import Dict, Optional
from CleanDiffuser.cleandiffuser.nn_condition.resnets import ResNet18


def load_config() -> dict:
    parser = argparse.ArgumentParser(description="Train Diffusion Policy for xArm with config")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/xarm.yaml",
        help="Path to config file (.json or .yaml/.yml)",
    )
    args = parser.parse_args()
    config_path = args.config

    # Support YAML with comments (Hydra/OmegaConf or PyYAML), and JSON
    if config_path.endswith((".yaml", ".yml")):
        # try OmegaConf first (Hydra)
        try:
            from omegaconf import OmegaConf
            cfg_ = OmegaConf.load(config_path)
            return OmegaConf.to_container(cfg_, resolve=True)  # type: ignore
        except Exception:
            # fallback to PyYAML
            try:
                import yaml  # type: ignore
            except Exception as e:
                raise ImportError(
                    "YAML config requested but neither omegaconf (hydra-core) nor PyYAML is available.\n"
                    "Please install one of them, e.g.: pip install hydra-core or pip install pyyaml"
                ) from e
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)


class DatasetWrapper:
    def __init__(self, dataset, image_size: int = 224):
        NORM_PARAMS = (0.5, 0.5, 0.5)
        self.dataset = dataset
        self.normalize = Normalize(NORM_PARAMS, NORM_PARAMS)
        self.resize = Resize((image_size, image_size))

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # image process - xArmDataset has already normalized the image to [0,1], and resize to 224x224 and normalize to [-1,1] here
        image_arm = item["obs"]["rgb_arm"].float()  # already [0,1]
        image_fix = item["obs"]["rgb_fix"].float()  # already [0,1]
        image_arm = self.resize(image_arm)  # resize to 224x224
        image_fix = self.resize(image_fix)  # resize to 224x224
        image_arm = self.normalize(image_arm)  # normalize to [-1,1]
        image_fix = self.normalize(image_fix)  # normalize to [-1,1]

        # state process - use pose+gripper state and external force information
        pos_with_gripper = item["obs"]["pos"].float()  # (To, 7) pose + gripper state
        force = item["obs"]["force"].float()  # (T_force, 6) force history
        lowdim = pos_with_gripper  # (To, 7) pose + gripper state
        
        # action - now 13 dim (6 dim position + 1 dim gripper action + 6 dim delta_force)
        act = item["action"].float()  # (Ta, 13) 13 dim action (6 dim position + 1 dim gripper action + 6 dim delta_force)

        # return data in the format of MultiViewResnetWithLowdimObsCondition
        obs = {"image_arm": image_arm, "image_fix": image_fix, "lowdim": lowdim, "force": force}
        return {"x0": act, "condition_cfg": obs}


class MultiViewResnetWithLowdimObsSeqCondition(IdentityCondition):

    def __init__(
        self,
        image_sz: int = 224,
        in_channel: int = 3,
        lowdim: int = 7,
        force_dim: int = 6,
        T_force: int = 10,  # force history length
        image_emb_dim: int = 256,
        lowdim_emb_dim: int = 64,
        force_emb_dim: int = 128,  # force embedding
        dropout: float = 0.0,
    ):
        super().__init__(dropout)
        self.T_force = T_force
        self.force_dim = force_dim
        
        self.resnet18_arm = ResNet18(image_sz=image_sz, in_channel=in_channel, emb_dim=image_emb_dim)
        self.resnet18_fix = ResNet18(image_sz=image_sz, in_channel=in_channel, emb_dim=image_emb_dim)
        self.lowdim_mlp = nn.Sequential(
            nn.Linear(lowdim, lowdim_emb_dim), nn.SiLU(), nn.Linear(lowdim_emb_dim, lowdim_emb_dim)
        )
        # force history encoder: flattened (T_force * force_dim) input, aggregated output
        self.force_mlp = nn.Sequential(
            nn.Linear(T_force * force_dim, force_emb_dim), nn.SiLU(), nn.Linear(force_emb_dim, force_emb_dim)
        )

    def forward(self, condition: Dict[str, torch.Tensor], mask: Optional[torch.Tensor] = None):
        image_arm = condition["image_arm"]  # (b, To, C, H, W)
        image_fix = condition["image_fix"]  # (b, To, C, H, W)
        lowdim = condition["lowdim"]  # (b, To, lowdim)
        force = condition["force"]  # (b, T_force, force_dim)

        # image encode per frame
        b, To = image_arm.shape[0], image_arm.shape[1]
        image_arm = image_arm.reshape(b * To, *image_arm.shape[-3:])  # (b*To, C, H, W)
        image_fix = image_fix.reshape(b * To, *image_fix.shape[-3:])  # (b*To, C, H, W)
        image_feat_arm = self.resnet18_arm(image_arm)  # (b*To, 256)
        image_feat_fix = self.resnet18_fix(image_fix)  # (b*To, 256)
        image_feat_arm = image_feat_arm.reshape(b, To, -1)  # (b, To, 256)
        image_feat_fix = image_feat_fix.reshape(b, To, -1)  # (b, To, 256)

        force_flat = force.reshape(b, -1)  # (b, T_force * force_dim) = (b, 60)
        force_feat = self.force_mlp(force_flat)  # (b, force_emb_dim)

        # encode lowdim per frame then flatten
        lowdim_feat = self.lowdim_mlp(lowdim)  # (b, To, 64)
        lowdim_feat = lowdim_feat.reshape(b, -1)  # (b, To * 64) = (b, 128)

        vec_condition = torch.cat([lowdim_feat, force_feat], dim=-1)  # (b, 128 + force_emb_dim)
        seq_condition = torch.cat([image_feat_arm, image_feat_fix], dim=-1)  # (b, To, 512)
        
        cond_dict = {
            "vec_condition": vec_condition,
            "seq_condition": seq_condition,
            "seq_condition_mask": None,
        }
        return cond_dict
        
if __name__ == "__main__":

    # =============================================== load config ============================================================ #
    cfg = load_config()
    devices = cfg.get("devices", [0])
    seed = int(cfg.get("seed", 0))
    ckpt_path = cfg.get("ckpt_path", "") or None
    dataset_path = cfg.get("dataset_path")
    work_dir = Path(cfg.get("work_dir"))
    model = cfg.get("model", "dit")
    batch_size = int(cfg.get("batch_size", 64))
    num_workers = int(cfg.get("num_workers", 8))
    max_steps = int(cfg.get("max_steps", 30000))
    precision = cfg.get("precision", "bf16-mixed")
    accumulate_grad_batches = int(cfg.get("accumulate_grad_batches", 1))
    use_persistent_workers = bool(cfg.get("use_persistent_workers", True))
    shuffle = bool(cfg.get("shuffle", True))
    image_size = int(cfg.get("image_size", 224))
    horizon = int(cfg.get("horizon", cfg.get("Ta", 64)))
    To = int(cfg.get("To", 2))
    log_every_n_steps = int(cfg.get("log_every_n_steps", 50))
    checkpoint_every_n_steps = int(cfg.get("checkpoint_every_n_steps", 5000))
    save_top_k = int(cfg.get("save_top_k", -1))
    save_last = bool(cfg.get("save_last", True))
    wandb_cfg = cfg.get("wandb", {})
    wandb_logger = None
    if isinstance(wandb_cfg, dict) and bool(wandb_cfg.get("enable", False)):
        if bool(wandb_cfg.get("offline", True)):
            os.environ["WANDB_MODE"] = "offline"
        try:
            from pytorch_lightning.loggers import WandbLogger  # type: ignore
            wandb_logger = WandbLogger(
                project=str(wandb_cfg.get("project", "clean_diffuser")),
                entity=wandb_cfg.get("entity", None),
                name=str(wandb_cfg.get("name", "dp_xarm")),
                save_dir=str(work_dir),
                log_model=bool(wandb_cfg.get("log_model", False)),
            )
        except Exception as e:
            cprint(f"{e}", "yellow")
    csv_logger = None
    try:
        from pytorch_lightning.loggers import CSVLogger  # type: ignore
        csv_logger = CSVLogger(save_dir=str(work_dir), name="lightning")
    except Exception as e:
        cprint(f"{e}", "yellow")
    # ====================================================================================================================== #




    # =============================================== load model =========================================================== #

    L.seed_everything(seed)

    nn_condition = MultiViewResnetWithLowdimObsSeqCondition(
            image_sz=image_size,
            in_channel=3,
            lowdim=7,                   # 6 dim position + 1 dim gripper state
            force_dim=6,                # 6 dim force
            T_force=10,                 # 10 step force history
            image_emb_dim=256,
            lowdim_emb_dim=64,
            force_emb_dim=128,          # aggregated embedding dim for force history
            dropout=0.0,
        )

    nn_diffusion = DiT1d(
        x_dim=13,                   # 13 dim action (6 dim position + 1 dim gripper action + 6 dim delta_force)
        x_seq_len=horizon,          # action horizon
        vec_emb_dim=256,            # lowdim 64*To 2 + force 128 = 128 + 128 = 256
        seq_emb_dim=512,            # image_feat_arm 256 + image_feat_fix 256 = 512  (with To)
        d_model=384, 
        n_heads=6,
        depth=12, 
        head_type="mlp",            
        use_cross_attn=True,
        adaLN_on_cross_attn=True,
        timestep_emb_type="untrainable_fourier",
        timestep_emb_params={"scale": 0.2},
        )


    policy = ContinuousRectifiedFlow(
        nn_diffusion=nn_diffusion, nn_condition=nn_condition
    )

    # ====================================================================================================================== #




    # =============================================== training ============================================================= #

    dataset = xArmDataset(
        file_path=dataset_path,
        To=int(cfg.get("To", 2)),
        Ta=horizon,  # training always uses horizon; Ta in config only affects inference
        T_force=int(cfg.get("T_force", 10)),
        normalizer_path=cfg.get("normalizer_path", None),
    )
    dataloader = torch.utils.data.DataLoader(
        DatasetWrapper(dataset, image_size=image_size),
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=use_persistent_workers,
        shuffle=shuffle,
    )
    callback = ModelCheckpoint(
        dirpath=work_dir,
        every_n_train_steps=checkpoint_every_n_steps,
        save_top_k=save_top_k,
        save_last=save_last,
    )
    logger_list = [l for l in [wandb_logger, csv_logger] if l is not None]
    trainer = L.Trainer(
        devices=devices,
        max_steps=max_steps,
        callbacks=[callback],
        precision=precision,
        strategy="auto",
        accumulate_grad_batches=accumulate_grad_batches,
        default_root_dir=work_dir,
        log_every_n_steps=log_every_n_steps,
        logger=logger_list if len(logger_list) > 0 else True,
    )

    print(f"\nTask: {cfg.get('task')}")
    print(f"Model: {cfg.get('model')}")
    print(f"Horizon: {cfg.get('horizon')}")
    print(f"To: {cfg.get('To')}")
    print(f"T_force: {cfg.get('T_force')}")
    print(f"Work Dir: {cfg.get('work_dir')}")
    print(f"Dataset Path: {cfg.get('dataset_path')}")
    print(f"Normalizer Path: {cfg.get('normalizer_path')}\n")


    if ckpt_path is not None and len(str(ckpt_path)) > 0:
        trainer.fit(policy, dataloader, ckpt_path=ckpt_path)
    else:
        trainer.fit(policy, dataloader)
        
    # ==================================================================================================================== #