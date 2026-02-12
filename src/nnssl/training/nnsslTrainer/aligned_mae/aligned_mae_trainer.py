import copy
from typing import Union, Tuple, List
from typing_extensions import override

import torch
import numpy as np

from torch import autocast

from nnssl.architectures.consis_arch import ConsisMAE, FeatureContrastiveDecoderAligned, ConsisMAEMaxPool
from nnssl.ssl_data.configure_basic_dummyDA import (
    configure_rotation_dummyDA_mirroring_and_inital_patch_size,
)
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import (
    BaseMAETrainer,
)
from nnssl.utilities.helpers import dummy_context
from nnssl.ssl_data.dataloading.aligned_transform import * #OverlapTransform, OverlapTransformNoRician, OverlapTransformNoInvert, OverlapTransformNoRicianNoInvert, OverlapTransformNoRicianNoInvertWindow, OverlapTransformNoRicianNoInvertClip, OverlapTransformNoRicianNoInvertRandClipRandWindow

import torch.nn as nn
from nnssl.ssl_data.dataloading.data_loader_3d import (
    nnsslDataLoader3D,
    nnsslDataLoader3DCenter,
    nnsslAnatDataLoader3D,
)

from batchgenerators.utilities.file_and_folder_operations import (
    join,
    isfile,
    save_json,
    maybe_mkdir_p,
    load_json,
)
from torch.nn.parallel import DistributedDataParallel as DDP

class FP32SyncBatchNorm(nn.SyncBatchNorm):
    #self.device = torch.device("cuda")
    def forward(self, x):
        with autocast(torch.device("cuda").type, enabled=False):
            y = super().forward(x.float())
        return y.to(x.dtype)

def _device_of_module(m: nn.Module):
    for p in m.parameters(recurse=True):
        return p.device
    # fall back to buffer device if no params
    for b in m.buffers(recurse=True):
        return b.device
    return torch.device("cpu")

def _clone_syncbn_to_fp32(child: nn.SyncBatchNorm) -> FP32SyncBatchNorm:
    new = FP32SyncBatchNorm(
        child.num_features,
        eps=child.eps,
        momentum=child.momentum,
        affine=child.affine,
        track_running_stats=child.track_running_stats,
        process_group=child.process_group,
    ).to(_device_of_module(child))

    with torch.no_grad():
        if child.affine:
            # weights/bias can be None if affine=False; we already guard on affine
            new.weight.copy_(child.weight.data.float())
            new.bias.copy_(child.bias.data.float())
        if child.track_running_stats:
            # These buffers can be None depending on init/flags
            if child.running_mean is not None:
                new.running_mean.copy_(child.running_mean.data.float())
            if child.running_var is not None:
                new.running_var.copy_(child.running_var.data.float())
            if hasattr(child, "num_batches_tracked") and child.num_batches_tracked is not None:
                new.num_batches_tracked.copy_(child.num_batches_tracked.data)
    return new

def convert_syncbn_to_fp32(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.SyncBatchNorm):
            setattr(module, name, _clone_syncbn_to_fp32(child))
        else:
            convert_syncbn_to_fp32(child)

class BaseAlignedResMAETrainer(BaseMAETrainer):
    """
    Base class for Key-Value Consistency EVA Trainer.
    This class inherits from EvaMAETrainer and is designed to handle
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the BaseKVConsisEvaTrainer with the given arguments.
        """
        super().__init__(*args, **kwargs)

        # Default initial patch size, can be overridden in get_dataloaders
        self.initial_patch_size = (256, 256, 256)
        self.total_batch_size = 4 #2
        self.initial_lr = 1e-2
        self.num_epochs = 1000 #250
        #self.num_iterations_per_epoch = 2
        self.teacher = None
        self.teacher_mom = 0.995  # Momentum for the teacher model update
        self.config_plan.patch_size = (
            160,
            160,
            160,
        )  # Default patch size for KV Consis Eva

    def initialize(self):
        if not self.was_initialized:
            self._set_batch_size()
            self.network: nn.Module
            self.adaptation_plan: AdaptationPlan
            self.network, self.adaptation_plan = (
                self.build_architecture_and_adaptation_plan(
                    self.config_plan, self.num_input_channels, self.num_output_channels
                )
            )
            save_json(self.adaptation_plan.serialize(), self.adaptation_json_plan)
            self.network.to(self.device)

            self.verify_adaptation_plans(
                self.adaptation_plan.serialize(),
                self.configuration_name,
                self.network.state_dict(),
            )
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file("Using torch.compile...")
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.network
                )
                convert_syncbn_to_fp32(self.network)
                self.network = DDP(
                    self.network,
                    device_ids=[self.local_rank],
                    find_unused_parameters=True,
                )

            self.loss = self.build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was already initialized. "
                "That should not happen."
            )

    def build_loss(self):
        """
        Builds the loss function for the model.
        This method is overridden to provide specific loss logic.
        """
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        # Create the loss function
        return AlignedMAELoss(device=self.device)

    def on_validation_epoch_start(self):
        # self.network.eval()

        # the predictor part of the model requires network to be in training mode...
        pass

    @override
    def build_architecture_and_adaptation_plan(
            self,
            config_plan,
            num_input_channels: int,
            num_output_channels: int,
            *args,
            **kwargs,
    ):
        # ---------------------------- Create architecture --------------------------- #
        architecture = ConsisMAE(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=False,
            only_last_stage_as_latent=False,
            use_projector=False,
        )
        # --------------------- Build associated adaptation plan --------------------- #
        # no changes to original mae since projector can be thrown away
        adapt_plan = self.save_adaption_plan(num_input_channels)
        return architecture, adapt_plan

    def get_validation_transforms(self):
        """
        Returns the validation transforms for the model.
        This method is overridden to provide specific validation transforms.
        """
        return OverlapTransform(
            train="none",
            data_key="data",
            initial_patch_size=self.initial_patch_size,
            patch_size=self.config_plan.patch_size,
        )

    def get_training_transforms(
            self,
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: dict,
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            order_resampling_data: int = 3,
            order_resampling_seg: int = 1,
            border_val_seg: int = -1,
            use_mask_for_norm: List[bool] = None,
    ):
        """
        Returns the training transforms for the model.
        This method is overridden to provide specific training transforms.
        """
        return OverlapTransform(
            train="train",
            data_key="data",
            initial_patch_size=self.initial_patch_size,
            patch_size=patch_size,
            do_dummy_2d_data_aug=do_dummy_2d_data_aug,
            order_resampling_data=order_resampling_data,
            order_resampling_seg=order_resampling_seg,
            border_val_seg=border_val_seg,
            use_mask_for_norm=use_mask_for_norm,
        )

    def get_dataloaders(self):
        """
        Dataloader creation is very different depending on the use-case of training.
        This method has to be implemneted for other use-cases aside from MAE more specifically.
        """
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.config_plan.patch_size
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = configure_rotation_dummyDA_mirroring_and_inital_patch_size(patch_size)
        if do_dummy_2d_data_aug:
            self.print_to_log_file("Using dummy 2D data augmentation")

        self.initial_patch_size = initial_patch_size
        self.print_to_log_file("Initial patch size: {}".format(initial_patch_size))

        # ------------------------ Training data augmentations ----------------------- #
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            mirror_axes,
            do_dummy_2d_data_aug,
            order_resampling_data=3,
            order_resampling_seg=1,
            use_mask_for_norm=self.config_plan.use_mask_for_norm,
        )

        # ----------------------- Validation data augmentations ---------------------- #
        val_transforms = self.get_validation_transforms()

        return self.make_generators(initial_patch_size, tr_transforms, val_transforms)

    def on_train_start(self):
        super().on_train_start()
        if self.teacher is None:
            # create a deep copy of the network to use as a teacher
            self.teacher = copy.deepcopy(self.network)
            self.teacher = self.teacher.to(self.device)
            self.teacher = (
                self.teacher.eval()
            )  # set the teacher to eval mode and not training
            for param in self.teacher.parameters():
                param.requires_grad = False

    def ema(self, teacher_model, student_model, update_bn=False):
        mom = self.teacher_mom

        if mom == 0.0:
            # if the momentum is 0, we just copy the student model to the teacher model
            teacher_model.load_state_dict(student_model.state_dict())
            return

        for p_s, p_t in zip(student_model.parameters(), teacher_model.parameters()):
            p_t.data = mom * p_t.data + (1 - mom) * p_s.data

        if not update_bn:
            return  # update BN stat buffers if required
        for (n_s, m_s), (n_t, m_t) in zip(
                student_model.named_modules(), teacher_model.named_modules()
        ):
            if isinstance(m_s, torch.nn.modules.batchnorm._NormBase) and n_s == n_t:
                m_t.running_mean.data = (
                        mom * m_t.running_mean.data + (1 - mom) * m_s.running_mean.data
                )
                m_t.running_var.data = (
                        mom * m_t.running_var.data + (1 - mom) * m_s.running_var.data
                )

    def shared_step(self, batch: dict, is_train: bool = True) -> dict:
        """
        Shared step for both training and validation.
        This method is overridden to provide specific shared step logic.
        """
        data, bboxes = batch["all_crops"], batch["rel_bboxes"]

        data = data.to(self.device, non_blocking=True)
        bboxes = bboxes.to(self.device, non_blocking=True)
        #import pdb; pdb.set_trace()

        with torch.no_grad():
            self.teacher.eval()
            teacher_output = self.teacher(data)
            teacher_output = {
                k: v
                for k, v in teacher_output.items()
                if k == "proj" or k == "image_latent"
            }
            self.network.train()  # set the network to training mode

        # We use the self.batch_size as it is not identical with the plan batch_size in ddp cases.
        mask = self.mask_creation(
            2 * self.batch_size, self.config_plan.patch_size, self.mask_percentage
        ).to(self.device, non_blocking=True)
        # Make the mask the same size as the data
        rep_D, rep_H, rep_W = (
            data.shape[2] // mask.shape[2],
            data.shape[3] // mask.shape[3],
            data.shape[4] // mask.shape[4],
        )
        mask = (
            mask.repeat_interleave(rep_D, dim=2)
            .repeat_interleave(rep_H, dim=3)
            .repeat_interleave(rep_W, dim=4)
        )

        masked_data = data * mask

        
        if is_train:
            self.optimizer.zero_grad(set_to_none=True)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            with torch.no_grad() if not is_train else dummy_context():
                output = self.network(masked_data)
                # del data
                loss_dict = self.loss(
                    model_output=output,
                    target=teacher_output,
                    gt_recon=data,
                    rel_bboxes=bboxes,
                    mask=mask,
                )
                l = loss_dict["loss"]

        if is_train:
            if self.grad_scaler is not None:
                self.grad_scaler.scale(l).backward()
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

            # update the teacher network with momentum of 0.95
            with torch.no_grad():
                self.ema(self.teacher, self.network, update_bn=False)

        return {k: v.detach().cpu().numpy() for k, v in loss_dict.items()}

    def train_step(self, batch: dict) -> dict:
        return self.shared_step(batch, is_train=True)

    def validation_step(self, batch: dict) -> dict:
        return self.shared_step(batch, is_train=False)


class HuberMAETrainer_BS24(BaseAlignedResMAETrainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the BaseKVConsisEvaTrainer with the given arguments.
        """
        super().__init__(*args, **kwargs)

        # Default initial patch size, can be overridden in get_dataloaders
        self.initial_patch_size = (256, 256, 256)
        self.total_batch_size = 24 #16 #32 # 4 #2
        self.initial_lr = 1e-2
        self.num_epochs = 1000 #1000 #250
        self.teacher = None
        self.teacher_mom = 0.995  # Momentum for the teacher model update
        self.config_plan.patch_size = (
            160,
            160,
            160,
        )  # Default patch size for KV Consis Eva
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 25 #50
        #self.num_iterations_per_epoch = 2

    def build_loss(self):
        """
        Builds the loss function for the model.
        This method is overridden to provide specific loss logic.
        """
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        # Create the loss function
        return AlignedMAELoss(device=self.device,
                recon_weight=5.0,
                fg_cos_weight=0.0,
                ntxent_weight=0.0,
                )

    def get_training_transforms(
            self,
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: dict,
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            order_resampling_data: int = 3,
            order_resampling_seg: int = 1,
            border_val_seg: int = -1,
            use_mask_for_norm: List[bool] = None,
    ):
        """
        Returns the training transforms for the model.
        This method is overridden to provide specific training transforms.
        """
        return OverlapTransformNoRicianNoInvertWindow(
            train="train",
            data_key="data",
            initial_patch_size=self.initial_patch_size,
            patch_size=patch_size,
            do_dummy_2d_data_aug=do_dummy_2d_data_aug,
            order_resampling_data=order_resampling_data,
            order_resampling_seg=order_resampling_seg,
            border_val_seg=border_val_seg,
            use_mask_for_norm=use_mask_for_norm,
        )



class AlignedHuberFTTrainer_BS24(HuberMAETrainer_BS24):

    def __init__(self, *args, **kwargs):
        """
        Initialize the ConMAETrainer with the given arguments. for 2nd stage
        """
        super().__init__(*args, **kwargs)
        self.total_batch_size = 24
        self.teacher_mom = 0.995
        self.initial_lr = 1e-2
        self.num_epochs = 250
        self.mask_percentage = 0.75  # Default mask percentage for ConMAE
        self.config_plan.patch_size = (160, 160, 160)  # Patch size for ConMAE

    def build_loss(self):
        """
        Builds the loss function for the model.
        This method is overridden to provide specific loss logic.
        """
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        # Create the loss function
        return AlignedMAELoss(
            device=self.device, recon_weight=5.0, fg_cos_weight=1.0, ntxent_weight=0.1
        )

    def get_training_transforms(
            self,
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: dict,
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            order_resampling_data: int = 3,
            order_resampling_seg: int = 1,
            border_val_seg: int = -1,
            use_mask_for_norm: List[bool] = None,
    ):
        """
        Returns the training transforms for the model.
        This method is overridden to provide specific training transforms.
        """
        return OverlapTransformNoRicianNoInvertClip(
            train="train",
            data_key="data",
            initial_patch_size=self.initial_patch_size,
            patch_size=patch_size,
            do_dummy_2d_data_aug=do_dummy_2d_data_aug,
            order_resampling_data=order_resampling_data,
            order_resampling_seg=order_resampling_seg,
            border_val_seg=border_val_seg,
            use_mask_for_norm=use_mask_for_norm,
        )



class AlignedHuberFTTrainer_MaxPool_BS20(BaseAlignedResMAETrainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the ConMAETrainer with the given arguments. for 2nd stage
        """
        super().__init__(*args, **kwargs)
        self.total_batch_size = 20
        self.teacher_mom = 0.995
        self.initial_lr = 1e-2
        self.num_epochs = 250
        self.mask_percentage = 0.75  # Default mask percentage for ConMAE
        self.config_plan.patch_size = (160, 160, 160)  # Patch size for ConMAE

    @override
    def build_architecture_and_adaptation_plan(
            self,
            config_plan,
            num_input_channels: int,
            num_output_channels: int,
            *args,
            **kwargs,
    ):
        # ---------------------------- Create architecture --------------------------- #
        architecture = ConsisMAEMaxPool(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=False,
            only_last_stage_as_latent=False,
            use_projector=False,
        )
        # --------------------- Build associated adaptation plan --------------------- #
        # no changes to original mae since projector can be thrown away
        adapt_plan = self.save_adaption_plan(num_input_channels)
        return architecture, adapt_plan

    def build_loss(self):
        """
        Builds the loss function for the model.
        This method is overridden to provide specific loss logic.
        """
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        # Create the loss function
        return AlignedMAELoss(
            device=self.device, recon_weight=5.0, fg_cos_weight=1.0, ntxent_weight=0.1
        )

    def get_training_transforms(
            self,
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: dict,
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            order_resampling_data: int = 3,
            order_resampling_seg: int = 1,
            border_val_seg: int = -1,
            use_mask_for_norm: List[bool] = None,
    ):
        """
        Returns the training transforms for the model.
        This method is overridden to provide specific training transforms.
        """
        return OverlapTransformNoRicianNoInvertClip(
            train="train",
            data_key="data",
            initial_patch_size=self.initial_patch_size,
            patch_size=patch_size,
            do_dummy_2d_data_aug=do_dummy_2d_data_aug,
            order_resampling_data=order_resampling_data,
            order_resampling_seg=order_resampling_seg,
            border_val_seg=border_val_seg,
            use_mask_for_norm=use_mask_for_norm,
        )

