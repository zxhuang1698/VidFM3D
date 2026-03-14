import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision
import trimesh
from einops import rearrange
from lightning import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning_utilities.core.rank_zero import rank_zero_only
from PIL import Image
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics import MeanMetric
from torchmetrics.aggregation import BaseAggregator

from vidfm3d.utils import pylogger
from vidfm3d.utils.eval_utils import align_pmaps, build_mask, pmaps_to_pc
from vidfm3d.utils.loss import camera_loss, depth_loss, point_loss
from vidfm3d.utils.vis_utils import dump_viser_artifact, save_scene_glb, vfm_pca_images
from vidfm3d.vggt.utils.pose_enc import pose_encoding_to_extri_intri

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class AccumulatedSum(BaseAggregator):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            fn="sum",
            default_value=torch.tensor(0.0, dtype=torch.long),
            nan_strategy="warn",
            state_name="sum_value",
            **kwargs,
        )

    def update(self, value: int) -> None:
        self.sum_value += value

    def compute(self) -> torch.LongTensor:
        return self.sum_value


class VideoProbeLitModule(LightningModule):
    def __init__(
        self,
        probe: torch.nn.Module,
        loss_weights: Dict[str, float],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        probe_type: str = "camera",
        pretrained: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
        train_viz_freq: int = 2000,  # iteration
        train_eval_freq: int = 200,  # iteration
        val_viz_freq: int = 20,  # epoch
        soft_bg_weight: float = 0.0,  # if > 0, no hard masking, weigh the confidence map instead
        cmap_power: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["probe"])
        self.kwargs = kwargs
        self.output_path = kwargs.get("output_path", None)

        self.train_viz_freq = train_viz_freq
        self.val_viz_freq = val_viz_freq
        self.train_eval_freq = train_eval_freq
        self.soft_bg_weight = soft_bg_weight
        self.cmap_power = cmap_power
        self.hard_masking = soft_bg_weight <= 0
        self.probe = probe
        self.probe_type = probe_type
        self.pretrained = pretrained
        self.loss_weights = loss_weights
        self.resume_from_checkpoint = resume_from_checkpoint

        assert self.probe_type in [
            "camera",
            "image",
            "pixalign",
        ], f"Invalid probe type: {self.probe_type}"

        # use register_buffer to save these with checkpoints
        # so that when we resume training, these bookkeeping variables are preserved
        self.register_buffer(
            "epoch_fraction", torch.tensor(0.0, dtype=torch.float32, device=self.device)
        )
        self.register_buffer(
            "train_total_samples", torch.tensor(0, dtype=torch.long, device=self.device)
        )

        self.train_total_samples_per_step = (
            AccumulatedSum()
        )  # these need to be reduced across GPUs, so use Metric

        self.val_loss = MeanMetric()

    def forward(self, data: Dict[str, torch.Tensor]) -> Any:
        # vfm feature can come from either video model or per-frame image model
        video_tokens = data["vfm_feat"]

        # predict the pmaps from the video tokens
        if self.probe_type == "pixalign":
            output = self.probe(
                video_tokens.permute(0, 1, 4, 2, 3),  # [B, S, C, H, W]
                data["image"].shape,  # [B, S, C, Hi, Wi]
                fg_mask=data["masks"] if "masks" in data else None,
            )
        elif self.probe_type in ["camera", "image"]:
            query_input = (
                data["image"] if self.probe_type == "image" else data["plucker"]
            )
            if "masks" in data:
                query_input = torch.cat([query_input, data["masks"]], dim=2)
            output = self.probe(
                query_input,
                video_tokens,
                cam_intr=data["intrinsics"] if self.probe_type == "camera" else None,
                cam_extr=data["extrinsics"] if self.probe_type == "camera" else None,
            )

        output_dict = {}
        if "world_points" in output:
            output_dict["pmaps"] = output["world_points"].permute(0, 1, 4, 2, 3)
        if "depth" in output:
            output_dict["dmaps"] = output["depth"].permute(0, 1, 4, 2, 3)
        if "pose_enc_list" in output:
            output_dict["pose_list"] = output["pose_enc_list"]
            output_dict["pose"] = output["pose_enc"]

        return output_dict

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

        # the wandb logger lives in self.loggers
        # find the wandb logger and watch the model and gradients
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                self.wandb_logger = logger
                # log gradients, parameter histogram and model topology
                self.wandb_logger.watch(
                    self.probe, log="all", log_freq=500, log_graph=False
                )

    def on_train_epoch_start(self) -> None:
        # our custom dataset and sampler has to have epoch set by calling set_epoch
        if hasattr(self.trainer.train_dataloader, "dataset") and hasattr(
            self.trainer.train_dataloader.dataset, "set_epoch"
        ):
            self.trainer.train_dataloader.dataset.set_epoch(self.current_epoch)
        if hasattr(self.trainer.train_dataloader, "sampler") and hasattr(
            self.trainer.train_dataloader.sampler, "set_epoch"
        ):
            self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        # reset the validation metrics
        self.val_loss.reset()

        # our custom dataset and sampler has to have epoch set by calling set_epoch
        for loader in self.trainer.val_dataloaders:
            if hasattr(loader, "dataset") and hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(0)
            if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(0)

    def model_step(
        self, batch: Dict[str, torch.Tensor], return_metrics: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        device = self.device

        # Move data to device
        for key_name in batch:
            if isinstance(batch[key_name], torch.Tensor):
                batch[key_name] = batch[key_name].to(device, non_blocking=True)

        # Forward pass to get the pmaps
        preds = self.forward(batch)

        # Calculate the loss
        loss_details = {}
        loss = 0.0

        # Prepare the foreground mask and confidence map
        batch["cmaps"] = batch["cmaps"].pow(self.cmap_power)
        if "masks" in batch and self.hard_masking:
            fg_mask = batch["masks"].squeeze(2)
            conf_map = batch["cmaps"].squeeze(2)
        elif "masks" in batch and not self.hard_masking:
            fg_mask = None
            conf_map = batch["cmaps"].squeeze(2)
            conf_map[batch["masks"].squeeze(2) <= 0] *= self.soft_bg_weight
        else:
            fg_mask = None
            conf_map = batch["cmaps"].squeeze(2)

        if "pmaps" in preds and self.loss_weights["pmap"] > 0:
            pmap_loss_dict = point_loss(
                preds["pmaps"].permute(0, 1, 3, 4, 2),  # [B, S, H, W, 3]
                batch["cmaps"].squeeze(2),
                batch["pmaps"].permute(0, 1, 3, 4, 2),  # [B, S, H, W, 3]
                mask=fg_mask,
                normalize_pred=self.kwargs.get("normalize_pred", False),
                normalize_gt=self.kwargs.get("normalize_gt", False),
                gradient_loss="grad",
            )
            loss += self.loss_weights["pmap"] * pmap_loss_dict["loss_point"]
            if self.loss_weights["pmap_grad"] > 0:
                loss += (
                    self.loss_weights["pmap_grad"] * pmap_loss_dict["loss_grad_point"]
                )
            loss_details.update(pmap_loss_dict)
        if "dmaps" in preds and self.loss_weights["dmap"] > 0:
            dmap_loss_dict = depth_loss(
                preds["dmaps"].permute(0, 1, 3, 4, 2),  # [B, S, H, W, 1]
                batch["cmaps"].squeeze(2),
                batch["dmaps"].permute(0, 1, 3, 4, 2),  # [B, S, H, W, 1]
                mask=fg_mask,
                gradient_loss="grad",
            )
            loss += self.loss_weights["dmap"] * dmap_loss_dict["loss_depth"]
            if self.loss_weights["dmap_grad"] > 0:
                loss += (
                    self.loss_weights["dmap_grad"] * dmap_loss_dict["loss_grad_depth"]
                )
            loss_details.update(dmap_loss_dict)
        if "pose" in preds and self.loss_weights["camera"] > 0:
            cam_loss_dict = camera_loss(
                preds["pose_list"],
                batch["intrinsics"],
                batch["extrinsics"],
                batch["image"].shape[-2:],
                loss_type="huber",
                return_metrics=return_metrics,
            )
            loss += self.loss_weights["camera"] * cam_loss_dict["loss_camera"]
            loss_details.update(cam_loss_dict)

        return batch, preds, loss, loss_details

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        data, preds, loss, loss_details = self.model_step(
            batch, return_metrics=self.global_step % (self.train_viz_freq // 20) == 0
        )

        if not isinstance(
            loss, (torch.Tensor, dict, type(None))
        ):  # this will cause a lightning.fabric.utilities.exceptions.MisconfigurationException
            # log loss and the batch information to help debugging
            # use print instead of log because the logger only logs on rank 0, but this could happen on any rank
            print(f"Loss is not a tensor or dict but {type(loss)}, value: {loss}")
            print(f"Loss details: {loss_details}")
            print(f"Batch: {batch}")
            print(f"Batch index: {batch_idx}")
            print(f"Data: {data}")
            print(f"Preds: {preds}")
            loss = None  # set loss to None will still break the training loop in DDP, this is intended - we should fix the data to avoid nan loss in the first place
            return loss

        self.epoch_fraction = torch.tensor(
            self.trainer.current_epoch + batch_idx / self.trainer.num_training_batches,
            device=self.device,
        )

        self.log(
            "trainer/epoch",
            self.epoch_fraction,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log(
            "trainer/lr",
            self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        # log the details of the loss
        if loss_details is not None:
            for key, value in loss_details.items():
                self.log(
                    f"train/{key}",
                    value,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                )

        # Log the total number of samples seen so far
        batch_size = data["image"].shape[0]
        self.train_total_samples_per_step(batch_size)  # aggregate across all GPUs
        self.train_total_samples += (
            self.train_total_samples_per_step.compute()
        )  # accumulate across all steps
        self.train_total_samples_per_step.reset()
        self.log(
            "trainer/total_samples",
            self.train_total_samples,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

        # log the image and point cloud
        if (
            self.global_step % self.train_viz_freq == 0
            or self.global_step % self.train_eval_freq == 0
        ):
            aligned_mses = []
            for i in range(batch_size):
                save_path = os.path.join(
                    self.output_path,
                    "viz",
                    "train",
                    f"step_{self.global_step}",
                    f"sample_{i:02d}",
                )
                pca_maps = self._make_pca_tensor(
                    data["vfm_feat"][i],  # (T,H,W,C)
                    data["vfm_idx"][i],  # (S,)
                    data["vfm_name"][i],  # "wan" or "dino"
                    data["image"].shape[-2],  # H
                    data["image"].shape[-1],  # W
                )  # → (S, 3, H, W)
                if self.global_step % self.train_viz_freq == 0:
                    # visualize the four images for each batch sample
                    self.visualize_image_grid(
                        mv_images=data["image"][i],
                        conf_map=data["cmaps"][i],
                        masks=data["masks"][i] if "masks" in data else None,
                        pred_depths=preds["dmaps"][i] if "dmaps" in preds else None,
                        gt_depths=data["dmaps"][i] if "dmaps" in data else None,
                        pred_pmaps=preds["pmaps"][i] if "pmaps" in preds else None,
                        gt_pmaps=data["pmaps"][i] if "pmaps" in data else None,
                        pca_map=pca_maps,
                        save_path=save_path + "_depth.png",
                        name=f"train-viz/{i:02d}:depth",
                    )
                # visualize the point cloud for each batch sample
                if "pmaps" in preds:
                    mask_i = build_mask(
                        data["cmaps"][i],
                        data["masks"][i] if "masks" in data else None,
                        thresh=0.75,
                    ).to(self.device)
                    pred_aligned = align_pmaps(
                        preds["pmaps"][i].detach(),  # detach so grads untouched
                        data["pmaps"][i].detach(),
                        mask_i,
                    )
                    aligned_mse = torch.norm(
                        pred_aligned - data["pmaps"][i].detach(), dim=1, keepdim=True
                    )
                    weights = data["cmaps"][i] * (
                        data["masks"][i].float() if "masks" in data else 1.0
                    )
                    aligned_mse = (aligned_mse * weights).sum() / (weights.sum() + 1e-6)
                    aligned_mses.append(aligned_mse.item())

                    # visualize the point clouds
                    if self.global_step % self.train_viz_freq == 0:
                        self.visualize_pmaps(
                            pmaps=pred_aligned,
                            images=data["image"][i],
                            conf=data["cmaps"][i],
                            conf_thresh=0.75,
                            max_points=4096,
                            fg_masks=data["masks"][i] if "masks" in data else None,
                            save_path=save_path + "_pmaps_pred.ply",
                            name=f"train-viz/{i:02d}:pmaps-pred",
                        )
                        self.visualize_pmaps(
                            pmaps=data["pmaps"][i],
                            images=data["image"][i],
                            conf=data["cmaps"][i],
                            conf_thresh=0.75,
                            max_points=4096,
                            fg_masks=data["masks"][i] if "masks" in data else None,
                            save_path=save_path + "_pmaps_gt.ply",
                            name=f"train-viz/{i:02d}:pmaps-gt",
                        )
                        self.visualize_pmaps(
                            pmaps=pred_aligned,
                            images=data["image"][i],
                            conf=data["cmaps"][i],
                            pmaps_gt=data["pmaps"][i],
                            conf_thresh=0.75,
                            max_points=2048,
                            fg_masks=data["masks"][i] if "masks" in data else None,
                            save_path=save_path + "_pmaps_overlay.ply",
                            name=f"train-viz/{i:02d}:pmaps-overlay",
                        )
            aligned_mse_avg = np.mean(aligned_mses)
            self.log(
                "train/pmap_mse_aligned",
                aligned_mse_avg,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )

        return loss

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        data, preds, loss, loss_details = self.model_step(
            batch,
            return_metrics=True,
        )
        # Extract the dataset name and batch size
        batch_size = data["image"].shape[0]

        # Log the overall validation loss
        self.val_loss(loss)

        # Log the details of the loss with dataset name and view number in the key
        if loss_details is not None:
            for key, value in loss_details.items():
                self.log(
                    f"val/{key}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    reduce_fx="mean",
                    sync_dist=True,
                    add_dataloader_idx=False,
                    batch_size=batch_size,
                )

        loss_value = loss.detach().cpu().item()
        del loss, loss_details

        # Log the aligned pmap mse if available
        if "pmaps" in preds:
            aligned_mses = []
            aligned_preds = []
            for i in range(batch_size):
                mask_i = build_mask(
                    data["cmaps"][i],
                    data["masks"][i] if "masks" in data else None,
                    thresh=0.75,
                ).to(self.device)
                pred_aligned = align_pmaps(
                    preds["pmaps"][i].detach(),  # detach so grads untouched
                    data["pmaps"][i].detach(),
                    mask_i,
                )
                aligned_mse = torch.norm(
                    pred_aligned - data["pmaps"][i].detach(), dim=1, keepdim=True
                )
                weights = data["cmaps"][i] * (
                    data["masks"][i].float() if "masks" in data else 1.0
                )
                aligned_mse = (aligned_mse * weights).sum() / (weights.sum() + 1e-6)
                aligned_mses.append(aligned_mse.item())
                aligned_preds.append(pred_aligned)
            aligned_mse_avg = np.mean(aligned_mses)
            self.log(
                "val/pmap_mse_aligned",
                aligned_mse_avg,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                reduce_fx="mean",
                sync_dist=True,
                add_dataloader_idx=False,
                batch_size=batch_size,
            )

        # Visualize point clouds
        if (
            self.current_epoch % self.val_viz_freq == (self.val_viz_freq - 1)
            or self.current_epoch == 0
        ):
            for i in range(batch_size):
                # Only log/visualize the first 2 batches, each batch take 4 samples, rank 0 only
                if batch_idx < 2 and i < 2:
                    save_path = os.path.join(
                        self.output_path,
                        "viz",
                        "val",
                        f"epoch_{self.current_epoch}",
                        f"batch_{batch_idx}_sample_{i}",
                    )

                    pca_maps = self._make_pca_tensor(
                        data["vfm_feat"][i],  # (T,H,W,C)
                        data["vfm_idx"][i],  # (S,)
                        data["vfm_name"][i],  # "wan" or "dino"
                        data["image"].shape[-2],  # H
                        data["image"].shape[-1],  # W
                    )  # → (S, 3, H, W)

                    # Log / Save images and point clouds
                    self.visualize_image_grid(
                        mv_images=data["image"][i],
                        conf_map=data["cmaps"][i],
                        masks=data["masks"][i] if "masks" in data else None,
                        pred_depths=preds["dmaps"][i] if "dmaps" in preds else None,
                        gt_depths=data["dmaps"][i] if "dmaps" in data else None,
                        pred_pmaps=preds["pmaps"][i] if "pmaps" in preds else None,
                        gt_pmaps=data["pmaps"][i] if "pmaps" in data else None,
                        pca_map=pca_maps,
                        save_path=save_path + "_depth.png",
                        name=f"val-viz/{batch_idx}-{i}:depth",
                    )
                    if "pmaps" in preds:
                        self.visualize_pmaps(
                            pmaps=aligned_preds[i],
                            images=data["image"][i],
                            conf=data["cmaps"][i],
                            conf_thresh=0.75,
                            max_points=4096,
                            fg_masks=data["masks"][i] if "masks" in data else None,
                            save_path=save_path + "_pmaps_pred.ply",
                            name=f"val-viz/{batch_idx}-{i}:pmaps-pred",
                        )
                        self.visualize_pmaps(
                            pmaps=data["pmaps"][i],
                            images=data["image"][i],
                            conf=data["cmaps"][i],
                            conf_thresh=0.75,
                            max_points=4096,
                            fg_masks=data["masks"][i] if "masks" in data else None,
                            save_path=save_path + "_pmaps_gt.ply",
                            name=f"val-viz/{batch_idx}-{i}:pmaps-gt",
                        )
                        self.visualize_pmaps(
                            pmaps=aligned_preds[i],
                            images=data["image"][i],
                            conf=data["cmaps"][i],
                            pmaps_gt=data["pmaps"][i],
                            conf_thresh=0.75,
                            max_points=2048,
                            fg_masks=data["masks"][i] if "masks" in data else None,
                            save_path=save_path + "_pmaps_overlay.ply",
                            name=f"val-viz/{batch_idx}-{i}:pmaps-overlay",
                        )
                        save_scene_glb(
                            pmaps=preds["pmaps"][i],  # [S,3,H,W]  predicted points
                            images=data["image"][i],  # [S,3,H,W]
                            conf=data["cmaps"][i],  # [S,1,H,W]
                            extrinsics=data["extrinsics"][i],  # [S,3,4]
                            save_path=save_path + "_scene.glb",
                            conf_percentile=75.0,  # drop lowest 75 % conf
                            max_points=2048,
                        )

        del batch, preds
        torch.cuda.empty_cache()

        return loss_value

    def on_validation_epoch_end(self) -> None:
        self.log("val/loss", self.val_loss, prog_bar=True)

        # if self.current_epoch % 5 == 4 or self.current_epoch == 0:
        #     self.log("val/acc", self.val_acc, sync_dist=True)
        #     self.log("val/comp", self.val_comp, sync_dist=True)
        #     self.log("val/chamfer", self.val_chamfer, sync_dist=True, prog_bar=True)
        #     self.log("val/psnr", self.val_psnr, sync_dist=True, prog_bar=True)

        # if we dont do these, wandb for some reason cannot display the validation loss with them as the x-axis
        self.log("trainer/epoch", self.epoch_fraction, sync_dist=True)
        self.log(
            "trainer/total_samples",
            self.train_total_samples.cpu().item(),
            sync_dist=True,
        )

    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        data, preds, loss, loss_details = self.model_step(
            batch,
            return_metrics=True,
        )
        # Extract the dataset name and batch size
        batch_size = data["image"].shape[0]

        # Log the overall validation loss
        self.val_loss(loss)

        # Log the details of the loss with dataset name and view number in the key
        if loss_details is not None:
            for key, value in loss_details.items():
                self.log(
                    f"val/{key}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    reduce_fx="mean",
                    sync_dist=True,
                    add_dataloader_idx=False,
                    batch_size=batch_size,
                )

        loss_value = loss.detach().cpu().item()
        del loss, loss_details

        for i in range(batch_size):
            save_path = os.path.join(
                self.output_path,
                "viz",
                "val",
                f"epoch_{self.current_epoch}",
                f"batch_{batch_idx}_sample_{i}",
            )
            pca_maps = self._make_pca_tensor(
                data["vfm_feat"][i],  # (T,H,W,C)
                data["vfm_idx"][i],  # (S,)
                data["vfm_name"][i],
                data["image"].shape[-2],  # H
                data["image"].shape[-1],  # W
            )  # → (S, 3, H, W)
            self.visualize_image_grid(
                mv_images=data["image"][i],
                conf_map=data["cmaps"][i],
                masks=data["masks"][i] if "masks" in data else None,
                pred_depths=preds["dmaps"][i] if "dmaps" in preds else None,
                gt_depths=data["dmaps"][i] if "dmaps" in data else None,
                pred_pmaps=preds["pmaps"][i] if "pmaps" in preds else None,
                gt_pmaps=data["pmaps"][i] if "pmaps" in data else None,
                pca_map=pca_maps,
                save_path=save_path + "_grid.png",
                name=f"val-viz/{batch_idx}-{i}:grid",
                save_individual=True,
                save_only=True,  # Don't log to wandb, just save the images
            )

        # Log the aligned pmap mse if available
        if "pmaps" in preds:
            aligned_mses = []
            aligned_preds = []
            for i in range(batch_size):
                mask_i = build_mask(
                    data["cmaps"][i],
                    data["masks"][i] if "masks" in data else None,
                    thresh=0.75,
                ).to(self.device)
                pred_aligned = align_pmaps(
                    preds["pmaps"][i].detach(),  # detach so grads untouched
                    data["pmaps"][i].detach(),
                    mask_i,
                )
                aligned_mse = torch.norm(
                    pred_aligned - data["pmaps"][i].detach(), dim=1, keepdim=True
                )
                weights = data["cmaps"][i] * (
                    data["masks"][i].float() if "masks" in data else 1.0
                )
                aligned_mse = (aligned_mse * weights).sum() / (weights.sum() + 1e-6)
                aligned_mses.append(aligned_mse.item())
                aligned_preds.append(pred_aligned)
            aligned_mse_avg = np.mean(aligned_mses)
            self.log(
                "val/pmap_mse_aligned",
                aligned_mse_avg,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                reduce_fx="mean",
                sync_dist=True,
                add_dataloader_idx=False,
                batch_size=batch_size,
            )
            if "pose" in preds:
                pred_extrinsics = pose_encoding_to_extri_intri(
                    preds["pose_list"][-1].detach(),
                    batch["image"].shape[-2:],
                    pose_encoding_type="absT_quaR_FoV",
                    build_intrinsics=False,
                )[
                    0
                ]  # (B, S, 3, 4)
            else:
                pred_extrinsics = data["extrinsics"]

            for i in range(batch_size):
                scene_dir_pred = os.path.join(
                    self.output_path, "viser_viz", f"batch_{batch_idx}_sample_{i}_pred"
                )
                dump_viser_artifact(
                    aligned_preds[i],
                    data["image"][i],
                    data["cmaps"][i],
                    pred_extrinsics[i],
                    scene_dir_pred,
                    max_pts=100_000,  # Keep more points since we can filter dynamically
                )
                scene_dir_gt = os.path.join(
                    self.output_path, "viser_viz", f"batch_{batch_idx}_sample_{i}_gt"
                )
                dump_viser_artifact(
                    data["pmaps"][i],
                    data["image"][i],
                    data["cmaps"][i],
                    data["extrinsics"][i],
                    scene_dir_gt,
                    max_pts=100_000,  # Keep more points since we can filter dynamically
                )

        # Visualize point clouds
        if (
            self.current_epoch % self.val_viz_freq == (self.val_viz_freq - 1)
            or self.current_epoch == 0
        ):
            for i in range(batch_size):
                # Only log/visualize the first 2 batches, each batch take 4 samples, rank 0 only
                if batch_idx < 2 and i < 2:
                    save_path = os.path.join(
                        self.output_path,
                        "viz",
                        "val",
                        f"epoch_{self.current_epoch}",
                        f"batch_{batch_idx}_sample_{i}",
                    )

                    pca_maps = self._make_pca_tensor(
                        data["vfm_feat"][i],  # (T,H,W,C)
                        data["vfm_idx"][i],  # (S,)
                        data["vfm_name"][i],  # "wan" or "dino"
                        data["image"].shape[-2],  # H
                        data["image"].shape[-1],  # W
                    )  # → (S, 3, H, W)

                    # Log / Save images and point clouds
                    self.visualize_image_grid(
                        mv_images=data["image"][i],
                        conf_map=data["cmaps"][i],
                        masks=data["masks"][i] if "masks" in data else None,
                        pred_depths=preds["dmaps"][i] if "dmaps" in preds else None,
                        gt_depths=data["dmaps"][i] if "dmaps" in data else None,
                        pred_pmaps=preds["pmaps"][i] if "pmaps" in preds else None,
                        gt_pmaps=data["pmaps"][i] if "pmaps" in data else None,
                        pca_map=pca_maps,
                        save_path=save_path + "_depth.png",
                        name=f"val-viz/{batch_idx}-{i}:depth",
                    )
                    if "pmaps" in preds:
                        self.visualize_pmaps(
                            pmaps=aligned_preds[i],
                            images=data["image"][i],
                            conf=data["cmaps"][i],
                            conf_thresh=0.75,
                            max_points=4096,
                            fg_masks=data["masks"][i] if "masks" in data else None,
                            save_path=save_path + "_pmaps_pred.ply",
                            name=f"val-viz/{batch_idx}-{i}:pmaps-pred",
                        )
                        self.visualize_pmaps(
                            pmaps=data["pmaps"][i],
                            images=data["image"][i],
                            conf=data["cmaps"][i],
                            conf_thresh=0.75,
                            max_points=4096,
                            fg_masks=data["masks"][i] if "masks" in data else None,
                            save_path=save_path + "_pmaps_gt.ply",
                            name=f"val-viz/{batch_idx}-{i}:pmaps-gt",
                        )
                        self.visualize_pmaps(
                            pmaps=aligned_preds[i],
                            images=data["image"][i],
                            conf=data["cmaps"][i],
                            pmaps_gt=data["pmaps"][i],
                            conf_thresh=0.75,
                            max_points=2048,
                            fg_masks=data["masks"][i] if "masks" in data else None,
                            save_path=save_path + "_pmaps_overlay.ply",
                            name=f"val-viz/{batch_idx}-{i}:pmaps-overlay",
                        )
                        save_scene_glb(
                            pmaps=preds["pmaps"][i],  # [S,3,H,W]  predicted points
                            images=data["image"][i],  # [S,3,H,W]
                            conf=data["cmaps"][i],  # [S,1,H,W]
                            extrinsics=data["extrinsics"][i],  # [S,3,4]
                            save_path=save_path + "_scene.glb",
                            conf_percentile=75.0,  # drop lowest 75 % conf
                            max_points=16384,
                        )

        del batch, preds
        torch.cuda.empty_cache()

        return loss_value

    @staticmethod
    def _make_pca_tensor(
        vfm_feat: torch.Tensor,  # (T, N, C)
        vfm_idx: torch.Tensor,  # (S,)
        vfm_name: str,  # "wan" or "dino"
        H: int,
        W: int,
    ) -> torch.Tensor:  # → (S, 3, H, W)
        """
        Build ONE global PCA basis from the whole clip (`vfm_feat`) and
        return the PCA-RGB map **only for the requested frames** in `vfm_idx`.
        Output is float in [0,1] on the current device.
        """
        T, hp, wp, C = vfm_feat.shape

        # run PCA → list[ np.uint8 ]  (length = T)
        pca_imgs = vfm_pca_images(
            vfm_feat.reshape(-1, C).cpu(),  # (T*hp*wp, C)
            tp=T,
            hp=hp,
            wp=wp,
            hi=H,
            wi=W,
            return_pil=False,
        )

        # pick the frames that correspond to the rendered views
        sel = [pca_imgs[int(t)] for t in vfm_idx.cpu()]
        # (H,W,3) uint8  →  (3,H,W) float32 in [0,1]
        tensors = [
            torch.from_numpy(im).permute(2, 0, 1).to(torch.float32) / 255.0
            for im in sel
        ]
        return torch.stack(tensors, dim=0)  # (S,3,H,W)

    @torch.no_grad()
    @rank_zero_only
    def visualize_image_grid(
        self,
        mv_images: torch.Tensor,
        conf_map: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        pred_depths: Optional[torch.Tensor] = None,
        gt_depths: Optional[torch.Tensor] = None,
        pred_pmaps: Optional[torch.Tensor] = None,
        gt_pmaps: Optional[torch.Tensor] = None,
        pca_map: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None,
        name: Optional[str] = None,
        save_individual: bool = False,
        save_only: bool = False,
    ):
        """
        Creates a grid visualization of up to 8 frames from the same sample.
        For each frame, we produce multiple columns:
          [RGB | Conf | PredPmap | GTPmap | ErrPmap | PredDepth | GTDepth | ErrDepth | PCA]

        We'll unify min/max for depth across pred & gt, unify min/max for pmap across pred & gt,
        measure MSE for error maps, color them with JET, conf is grayscale, depths are Viridis,
        pmaps are X/Y/Z as RGB with same scale for pred & gt.

        Args:
            save_individual: If True, saves individual maps separately in addition to the grid
        """
        assert mv_images.dim() == 4
        S, C, H, W = mv_images.shape
        if masks is not None:
            masks = masks > 0

        # Subsample to 8 frames if more
        if S > 8:
            indices = torch.arange(0, S, S // 8)
            mv_images = mv_images[indices]
            conf_map = conf_map[indices]
            masks = masks[indices] if masks is not None else None
            pred_depths = pred_depths[indices] if pred_depths is not None else None
            gt_depths = gt_depths[indices] if gt_depths is not None else None
            pred_pmaps = pred_pmaps[indices] if pred_pmaps is not None else None
            gt_pmaps = gt_pmaps[indices] if gt_pmaps is not None else None
            pca_map = pca_map[indices] if pca_map is not None else None
            S = 8

        # Convert everything to CPU for min/max
        if pred_depths is not None and gt_depths is not None:
            pred_depths = pred_depths.detach().clone()
            gt_depths = gt_depths.detach().clone()
            # unify min/max for depth
            if masks is None:
                all_depth_values = (
                    torch.cat([pred_depths.view(-1), gt_depths.view(-1)]).cpu().numpy()
                )
            else:
                all_depth_values = (
                    torch.cat(
                        [
                            pred_depths[masks.expand_as(pred_depths)].view(-1),
                            gt_depths[masks.expand_as(gt_depths)].view(-1),
                        ]
                    )
                    .cpu()
                    .numpy()
                )
            d_min = float(np.percentile(all_depth_values, 5, axis=0))
            d_max = float(np.percentile(all_depth_values, 95, axis=0))
            if abs(d_max - d_min) < 1e-9:
                d_max = d_min + 1e-9
            if masks is not None:
                # if we have masks, apply them to the depth maps
                pred_depths[~masks.expand_as(pred_depths)] = d_max
                gt_depths[~masks.expand_as(gt_depths)] = d_max

            # Depth error => MSE
            err_dmap = (pred_depths[:, 0] - gt_depths[:, 0]).abs()  # [S, H, W]
            dm_err_min = 0.0
            dm_err_max = float(err_dmap.max().item())

        # unify min/max for pmaps
        if pred_pmaps is not None and gt_pmaps is not None:
            pred_pmaps = pred_pmaps.detach().clone()
            gt_pmaps = gt_pmaps.detach().clone()
            if masks is None:
                all_pmaps = torch.cat([pred_pmaps, gt_pmaps], dim=0)  # [2*S, 3, H, W]
                p_max = torch.max(all_pmaps.permute(0, 2, 3, 1).reshape(-1, 3), dim=0)[
                    0
                ]
                p_min = torch.min(all_pmaps.permute(0, 2, 3, 1).reshape(-1, 3), dim=0)[
                    0
                ]
            else:
                pred_pmaps = pred_pmaps.permute(0, 2, 3, 1)  # [S, H, W, 3]
                gt_pmaps = gt_pmaps.permute(0, 2, 3, 1)
                all_pmaps = torch.cat(
                    [pred_pmaps[masks.squeeze(1)], gt_pmaps[masks.squeeze(1)]], dim=0
                )  # (-1, 3)
                p_max = torch.max(all_pmaps, dim=0)[0]
                p_min = torch.min(all_pmaps, dim=0)[0]

                pred_pmaps[~masks.squeeze(1)] = p_max
                gt_pmaps[~masks.squeeze(1)] = p_max

                pred_pmaps = pred_pmaps.permute(0, 3, 1, 2)  # [S, 3, H, W]
                gt_pmaps = gt_pmaps.permute(0, 3, 1, 2)

            # PMAP error => MSE across channels
            err_pmap = torch.norm(pred_pmaps - gt_pmaps, dim=1)  # [S, H, W]
            pm_err_min = 0.0
            pm_err_max = float(err_pmap.max().item())

        # unify min/max for conf
        conf_all = conf_map.detach().cpu()
        c_min = 0
        c_max = float(conf_all.max())
        if abs(c_max - c_min) < 1e-9:
            c_max = c_min + 1e-9
        if masks is not None:
            conf_map = conf_map.detach().clone()
            # if we have masks, apply them to the confidence map
            conf_map[~masks.expand_as(conf_map)] = c_min

        # Prepare individual save directory if needed
        individual_save_dir = None
        if save_individual and save_path is not None:
            individual_save_dir = save_path.replace(".png", "_individual")
            os.makedirs(individual_save_dir, exist_ok=True)

        rows = []
        for i in range(S):
            # Convert RGB [3,H,W] -> BGR
            rgb = mv_images[i].detach().cpu()
            rgb_np = (rgb.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

            # Conf grayscale
            conf_1ch = conf_map[i, 0].detach().cpu()
            conf_bgr = self._apply_viridis_cv2(
                conf_1ch, c_min, c_max, colormap=None, grayscale=True
            )

            pmap_pred_bgr, pmap_gt_bgr, pmap_err_bgr = None, None, None
            if pred_pmaps is not None and gt_pmaps is not None:
                # PMAP pred normalize with [p_min, p_max]
                pmap_pred_bgr = self._pmap_to_color(pred_pmaps[i], p_min, p_max)
                pmap_gt_bgr = self._pmap_to_color(gt_pmaps[i], p_min, p_max)
                pmap_err_bgr = self._apply_viridis_cv2(
                    err_pmap[i], pm_err_min, pm_err_max, colormap=cv2.COLORMAP_JET
                )

            d_pred, d_gt, d_err_bgr = None, None, None
            if pred_depths is not None and gt_depths is not None:
                # Depth pred & gt => unify min/max => apply Viridis
                d_pred = self._apply_viridis_cv2(
                    pred_depths[i, 0], d_min, d_max, colormap=cv2.COLORMAP_VIRIDIS
                )
                d_gt = self._apply_viridis_cv2(
                    gt_depths[i, 0], d_min, d_max, colormap=cv2.COLORMAP_VIRIDIS
                )
                d_err_bgr = self._apply_viridis_cv2(
                    err_dmap[i], dm_err_min, dm_err_max, colormap=cv2.COLORMAP_JET
                )

            # PCA
            pca_bgr = None
            if pca_map is not None:
                assert pca_map.shape[:2] == (
                    S,
                    3,
                ), f"expected (S,3,H,W) got {pca_map.shape}"
                pca_rgb = (pca_map[i].detach().cpu() * 255).byte()
                pca_bgr = cv2.cvtColor(
                    pca_rgb.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR
                )

            # Save individual maps if requested
            if individual_save_dir is not None:
                view_dir = os.path.join(individual_save_dir, f"view_{i:02d}")
                os.makedirs(view_dir, exist_ok=True)

                # Save RGB image
                cv2.imwrite(os.path.join(view_dir, "rgb.png"), rgb_bgr)

                # Save confidence map
                cv2.imwrite(os.path.join(view_dir, "confidence.png"), conf_bgr)

                # Save depth maps if available
                if d_pred is not None:
                    cv2.imwrite(os.path.join(view_dir, "depth_pred.png"), d_pred)
                if d_gt is not None:
                    cv2.imwrite(os.path.join(view_dir, "depth_gt.png"), d_gt)
                if d_err_bgr is not None:
                    cv2.imwrite(os.path.join(view_dir, "depth_error.png"), d_err_bgr)

                # Save pmap visualizations if available
                if pmap_pred_bgr is not None:
                    cv2.imwrite(os.path.join(view_dir, "pmap_pred.png"), pmap_pred_bgr)
                if pmap_gt_bgr is not None:
                    cv2.imwrite(os.path.join(view_dir, "pmap_gt.png"), pmap_gt_bgr)
                if pmap_err_bgr is not None:
                    cv2.imwrite(os.path.join(view_dir, "pmap_error.png"), pmap_err_bgr)

                # Save PCA map if available
                if pca_bgr is not None:
                    cv2.imwrite(os.path.join(view_dir, "pca.png"), pca_bgr)

            # concatenate the row
            cols = [
                rgb_bgr,
                conf_bgr,
            ]
            if pred_pmaps is not None and gt_pmaps is not None:
                cols.append(pmap_pred_bgr)
                cols.append(pmap_gt_bgr)
                cols.append(pmap_err_bgr)
            if pred_depths is not None and gt_depths is not None:
                cols.append(d_pred)
                cols.append(d_gt)
                cols.append(d_err_bgr)
            if pca_bgr is not None:
                cols.append(pca_bgr)  # 9-th column

            row_img = np.concatenate(cols, axis=1)
            rows.append(row_img)

        final_grid = np.concatenate(rows, axis=0)

        if not save_only:
            # Log to W&B if we have it
            for logger in self.loggers:
                if isinstance(logger, WandbLogger):
                    import wandb

                    final_rgb = cv2.cvtColor(final_grid, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(final_rgb)
                    # resize the height to 384
                    pil_img = pil_img.resize(
                        (int(pil_img.width * 384 / pil_img.height), 384)
                    )
                    logger.experiment.log({name or "image-grid": wandb.Image(pil_img)})
                    break

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, final_grid)

        return final_grid

    def _apply_viridis_cv2(
        self,
        map_1ch: torch.Tensor,
        min_val: float,
        max_val: float,
        colormap: Optional[int] = cv2.COLORMAP_VIRIDIS,
        grayscale: bool = False,
    ) -> np.ndarray:
        """
        Normalizes map_1ch to [0..255] with the specified (min_val, max_val).
        Then applies either an OpenCV colormap or grayscale.
        Returns HxWx3 BGR uint8 array.
        """
        arr = map_1ch.detach().cpu().numpy()
        denom = max_val - min_val
        if denom < 1e-9:
            denom = 1e-9
        norm = (arr - min_val) / denom
        norm = np.clip(norm, 0, 1)
        norm_255 = (norm * 255).astype(np.uint8)

        if grayscale:
            # produce HxWx3
            gray_bgr = cv2.cvtColor(norm_255, cv2.COLOR_GRAY2BGR)
            return gray_bgr
        else:
            colored_bgr = cv2.applyColorMap(norm_255, colormap)
            return colored_bgr

    def _pmap_to_color(
        self,
        pmap: torch.Tensor,
        p_min: torch.Tensor,  # [3] min values for X, Y, Z
        p_max: torch.Tensor,  # [3] max values for X, Y, Z
    ) -> np.ndarray:
        """
        pmap is shape [3,H,W]. We interpret them as XYZ => treat as RGB.
        We unify (p_min, p_max) across all channels, scale to [0..1],
        then convert to BGR for display.
        """
        arr = pmap.detach()  # shape [3,H,W]
        p_min = p_min.view(3, 1, 1)  # [3,1,1]
        p_max = p_max.view(3, 1, 1)  # [3,1,1]
        denom = p_max - p_min
        denom[denom < 1e-9] = 1e-9
        scaled = (arr - p_min) / denom
        scaled = scaled.cpu().numpy()
        scaled = np.clip(scaled, 0.0, 1.0)
        scaled_255 = (scaled * 255).astype(np.uint8)  # shape [3,H,W]
        scaled_255 = np.transpose(scaled_255, (1, 2, 0))  # [H,W,3]
        bgr = cv2.cvtColor(scaled_255, cv2.COLOR_RGB2BGR)
        return bgr

    @rank_zero_only
    def visualize_pmaps(
        self,
        pmaps: torch.Tensor,  # [S,3,H,W] (already aligned if you wish)
        images: torch.Tensor,  # [S,3,H,W]  RGB in [0..1]
        conf: torch.Tensor,  # [S,1,H,W]
        pmaps_gt: Optional[torch.Tensor] = None,  # [S,3,H,W] (if provided)
        fg_masks: Optional[torch.Tensor] = None,
        conf_thresh: float = 0.75,  # threshold for confidence
        max_points: int = 4096,
        save_path: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """
        Builds one coloured point-cloud:

        • GT points - GREEN
        • Pred points - RED    (if pmaps_gt provided)
        • otherwise only pred points (true colour from images).

        Then delegates to visualize_point_cloud.
        """
        mask = build_mask(conf, fg_masks, thresh=conf_thresh).to(
            pmaps.device
        )  # [S,1,H,W]

        # ---------- predicted
        pc_pred = pmaps_to_pc(pmaps, images, mask, max_points)  # [N,6]
        if pmaps_gt is None:
            self.visualize_point_cloud(pc_pred, save_path, name)
            return

        # ---------- ground-truth
        pc_gt = pmaps_to_pc(pmaps_gt, images, mask, max_points)

        # re-colour: pred = red / gt = green
        pc_pred[:, 3:] = torch.tensor([1.0, 0.0, 0.0], device=pc_pred.device)  # R
        pc_gt[:, 3:] = torch.tensor([0.0, 1.0, 0.0], device=pc_gt.device)  # G
        pc_combined = torch.cat([pc_pred, pc_gt], dim=0)  # [N1+N2,6]

        self.visualize_point_cloud(pc_combined, save_path, name)

    @rank_zero_only
    def visualize_image(
        self,
        image: torch.Tensor,
        save_path: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """
        image: shape [C, H, W], assumed in [0,1].
        Logs to W&B as an image or grid, and optionally saves to disk.
        """
        assert image.dim() == 3, f"Image has shape {tuple(image.shape)}; expected CxHxW"
        assert image.shape[0] in [
            1,
            3,
        ], f"Image has shape {tuple(image.shape)}; expected C=1 or C=3"
        assert (
            image.max() <= 1 and image.min() >= 0
        ), f"Image has values {image.min()}..{image.max()}; expected [0,1]"

        # Move to CPU
        image = image.detach().cpu()
        image_pil = torchvision.transforms.ToPILImage()(image)

        # Log to wandb if we have it
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                import wandb

                image_wandb = [wandb.Image(image_pil)]
                logger.experiment.log({name or "image": image_wandb})
                break

        # Save to disk if needed
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image_pil.save(save_path)

    @rank_zero_only
    def visualize_point_cloud(
        self,
        point_cloud: torch.Tensor,
        save_path: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Visualize or log a colored point cloud. Assumes shape [N,6] for (x,y,z,r,g,b) in [0..1].
        Args:
            point_cloud: A [N,6] tensor with XYZ and RGB in [0..1].
            save_path: If provided, exports a .ply file to this path.
            name: A name for logging (if using wandb or another logger).
        """
        if point_cloud.dim() != 2 or point_cloud.shape[1] < 6:
            log.warning(
                f"Point cloud has shape {tuple(point_cloud.shape)}; expected Nx6. Skipping visualization."
            )
            return

        # Move to CPU and convert to numpy
        pc_np = point_cloud.detach().cpu().numpy()

        # Split into xyz and color
        xyz = pc_np[:, :3]
        color = pc_np[:, 3:6]  # assume [0..1]; scale for ply or wandb

        # Convert color to [0..255] uint8
        color_255 = (color * 255).clip(0, 255).astype(np.uint8)

        # 1) Log to wandb if we have a WandbLogger
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                import wandb

                # wandb.Object3D expects Nx6 or Nx7 with color
                # We'll pack xyz + color_255 together
                # Right now +y is up, -z is forward, wandb use +z up, +x forward
                # so new xyz = [-z, -x ,-y]
                xyz_wandb = np.stack([-xyz[:, 2], -xyz[:, 0], -xyz[:, 1]], axis=-1)
                pc_wandb = np.concatenate([xyz_wandb, color_255], axis=1)
                pc_wandb = [wandb.Object3D(pc_wandb)]
                logger.experiment.log({name or "pointcloud": pc_wandb})
                break  # only need to log once

        # 2) Export to .ply if save_path is given
        if save_path is not None:
            # Create a point cloud via trimesh
            pc_trimesh = trimesh.PointCloud(vertices=xyz, colors=color_255)
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            pc_trimesh.export(save_path)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        if self.hparams.scheduler is not None:
            scheduler_config = self.hparams.scheduler

            # HACK: if the class is pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR,
            # both warmup_epochs and max_epochs should be scaled.
            # more specifically, max_epochs should be scaled to total number of steps that we will have during training,
            # and warmup_epochs should be scaled up proportionally.
            if scheduler_config.func is LinearWarmupCosineAnnealingLR:
                # Extract the keyword arguments from the partial object
                scheduler_kwargs = {k: v for k, v in scheduler_config.keywords.items()}
                original_warmup_epochs = scheduler_kwargs["warmup_epochs"]
                original_max_epochs = scheduler_kwargs["max_epochs"]

                total_steps = (
                    self.trainer.estimated_stepping_batches
                )  # total number of total steps in all training epochs

                # Scale warmup_epochs and max_epochs
                scaled_warmup_epochs = int(
                    original_warmup_epochs * total_steps / original_max_epochs
                )
                scaled_max_epochs = total_steps

                # Update the kwargs with scaled values
                scheduler_kwargs.update(
                    {
                        "warmup_epochs": scaled_warmup_epochs,
                        "max_epochs": scaled_max_epochs,
                    }
                )

                # Re-initialize the scheduler with updated parameters
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer=optimizer, **scheduler_kwargs
                )
            else:
                scheduler = scheduler_config(optimizer=optimizer)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "name": "train/lr",  # put lr inside train group in loggers
                    "scheduler": scheduler,
                    "interval": "step"
                    if scheduler_config.func is LinearWarmupCosineAnnealingLR
                    else "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.probe = torch.compile(self.probe)

        # Load pretrained weights if available and not resuming
        # note that if resume_from_checkpoint is set, the Trainer is responsible for actually loading the checkpoint
        # so we are only using resume_from_checkpoint as a check of whether we should load the pretrained weights
        if self.pretrained and not self.resume_from_checkpoint:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self) -> None:
        log.info(f"Loading pretrained: {self.pretrained}")
        log.info(f"Loading pretrained weights from {self.pretrained}")
        checkpoint = torch.load(self.pretrained)

        # Load the probe weights
        filtered_state_dict = {
            k: v for k, v in checkpoint["state_dict"].items() if k.startswith("probe.")
        }
        # Remove the 'probe.' prefix from the keys
        filtered_state_dict = {
            k[len("probe.") :]: v for k, v in filtered_state_dict.items()
        }
        # Load the filtered state_dict into the model
        self.probe.load_state_dict(filtered_state_dict, strict=True)

    @staticmethod
    def _update_ckpt_keys(
        ckpt,
        new_head_name="downstream_head",
        head_to_keep="downstream_head1",
        head_to_discard="downstream_head2",
    ):
        """Helper function to use the weights of a model with multiple heads in a model with a single head.
        specifically, keep only the weights of the first head and delete the weights of the second head.
        """
        new_ckpt = {"model": {}}

        for key, value in ckpt["model"].items():
            if key.startswith(head_to_keep):
                new_key = key.replace(head_to_keep, new_head_name)
                new_ckpt["model"][new_key] = value
            elif key.startswith(head_to_discard):
                continue
            else:
                new_ckpt["model"][key] = value

        return new_ckpt
