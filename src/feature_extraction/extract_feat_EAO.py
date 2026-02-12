import multiprocessing as mp
mp.set_start_method("spawn", force=True)
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.transforms import (
    Compose,
    CenterSpatialCropd,
    ToTensord,
    MapTransform,
    ResizeWithPadOrCropd,
    CropForegroundd,
    LoadImaged
)
from monai.transforms import DeleteItemsd
import numpy as np
import SimpleITK as sitk
from functools import partial
import json
import argparse
import h5py
from transforms.monai_transforms import MaskCenterCropd

# Import nnssl components (these imports trigger paths.py to read the environment variables)
from nnssl.experiment_planning.experiment_planners.plan import Plan
# from nnssl.preprocessing.normalization.normalization_schemes import z_score_normalization
from nnssl.training.nnsslTrainer.aligned_mae.aligned_mae_trainer import AlignedHuberFTTrainer_MaxPool_BS20 as Trainer
from nnssl.imageio.simpleitk_reader_writer import SimpleITKIO
# Using torch-based GPU-accelerated resampling instead of CPU-based default_resampling
# from nnssl.preprocessing.resampling.default_resampling import resample_data_or_seg_to_spacing
from nnssl.preprocessing.cropping.cropping import crop_to_nonzero
from typing import Type, Protocol
from numbers import Number
import torch.distributed as dist
from models.consis_arch import ConsisMAE

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_first_valid_key(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"None of the specified keys found: {keys}")

def load_pretrained_weights(
    resenc_model,
    pretrained_weights_file,
):
    if dist.is_initialized():
        saved_model = torch.load(
            pretrained_weights_file,
            map_location=torch.device("cuda", dist.get_rank()),
            weights_only=False,
        )
    else:
        saved_model = torch.load(pretrained_weights_file, weights_only=False)
    if 'network_weights' in saved_model:
        # print("Loaded weights from 'network_weights'")
        pretrained_dict = saved_model['network_weights']
    elif 'state_dict' in saved_model:
        # print("Loaded weights from 'state_dict'")
        pretrained_dict = saved_model['state_dict']
    else:
        raise KeyError("No compatible weight dictionary ('network_weights' or 'state_dict') found in checkpoint")

    mod = resenc_model

    model_dict = mod.state_dict()

    in_conv_weights_model = get_first_valid_key(model_dict, [
        "encoder.stem.convs.0.all_modules.0.weight",
        "encoder.res_unet.encoder.stem.convs.0.all_modules.0.weight"
    ])

    in_conv_weights_pretrained = get_first_valid_key(pretrained_dict, [
        "encoder.stem.convs.0.all_modules.0.weight",
        "encoder.res_unet.encoder.stem.convs.0.all_modules.0.weight"
    ])


    in_channels_model = in_conv_weights_model.shape[1]
    in_channels_pretrained = in_conv_weights_pretrained.shape[1]

    if in_channels_model != in_channels_pretrained:
        assert in_channels_pretrained == 1, (
            f"The input channels do not match. Pretrained model: {in_channels_pretrained}; your network: "
            f"your network: {in_channels_model}"
        )

        repeated_weight_tensor = in_conv_weights_pretrained.repeat(
            1, in_channels_model, 1, 1, 1) / in_channels_model
        target_data_ptr = in_conv_weights_pretrained.data_ptr()
        for key, weights in pretrained_dict.items():
            if weights.data_ptr() == target_data_ptr:
                # print(key)
                pretrained_dict[key] = repeated_weight_tensor

        # SPECIAL CASE HARDCODE INCOMING
        # Normally, these keys have the same data_ptr that points to the weights that are to be replicated:
        # - encoder.stem.convs.0.conv.weight
        # - encoder.stem.convs.0.all_modules.0.weight
        # - decoder.encoder.stem.convs.0.conv.weight
        # - decoder.encoder.stem.convs.0.all_modules.0.weight
        # But this is not the case for 'VariableSparkMAETrainer_BS8', where we replace modules from the original
        # encoder architecture, so that the following two point to a different tensor:
        # - encoder.stem.convs.0.conv.weight
        # - decoder.encoder.stem.convs.0.conv.weight
        # resulting in a shape mismatch for the two missing keys in the check below.
        # It is important to note, that the weights being trained are located at 'all_modules.0.weight', so we
        # have to use those as the source of replication

    skip_strings_in_pretrained = [".seg_layers."]
    skip_strings_in_pretrained.extend(["decoder.stages", "decoder.transpconvs"])

    final_pretrained_dict = {}
    for key, v in pretrained_dict.items():
        if key in model_dict and all(
            [i not in key for i in skip_strings_in_pretrained]
        ):
            assert model_dict[key].shape == pretrained_dict[key].shape, (
                f"The shape of the parameters of key {key} is not the same. Pretrained model: "
                f"{pretrained_dict[key].shape}; your network: {model_dict[key].shape}. The pretrained model "
                f"does not seem to be compatible with your network."
            )
            final_pretrained_dict[key] = v

    model_dict.update(final_pretrained_dict)

    # print("################### Loading pretrained weights from file ", fname, '###################')
    # print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
    # for key, value in final_pretrained_dict.items():
    #     print(key, 'shape', value.shape)
    # print("################### Done ###################")
    # exit()
    mod.load_state_dict(model_dict)

    return mod



class ResEncoderPatchLatent(nn.Module):
    def __init__(
        self,
        **hypparams,
    ):
        super(ResEncoderPatchLatent, self).__init__()

        self.res_unet = ConsisMAE(
            1,
            n_stages=6,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=torch.nn.modules.conv.Conv3d,
            kernel_sizes=[
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
            ],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=torch.nn.modules.instancenorm.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-05, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            num_classes=hypparams["num_classes"],
            only_last_stage_as_latent=False,
            use_projector=False,
            use_projector_global=True,

        )
        self.res_unet.encoder.return_skips = True
        self.res_unet = load_pretrained_weights(
            self.res_unet,
            hypparams["chpt_path"],
        )
        # remove self.res_unet.decoder and self.res_unet.seg_layers to save memory since we won't use them for feature extraction
        self.res_unet.decoder = None
        self.res_unet.seg_layers = None

    def forward(self, x):
        skips = self.res_unet.encoder(x)

        patch_latent = torch.concat([self.res_unet.v_adaptive_pool(s) for s in skips], dim=1)
        return patch_latent # torch.Size([4, 1120, 16, 16, 16])




def convert_dtype(image: np.ndarray, target_dtype: Type[Number]) -> np.ndarray:
    return image.astype(target_dtype)


def z_score_normalization(
    image: np.ndarray,
    use_mask_for_norm: bool,
    non_zero_mask: np.ndarray,
    target_dtype: Type[Number],
) -> np.ndarray:
    image = convert_dtype(image, target_dtype)
    if use_mask_for_norm:
        mask = non_zero_mask >= 0
        mean = image[mask].mean()
        std = image[mask].std()
        image[mask] = (image[mask] - mean) / max(std, 1e-8)
    else:
        mean = image.mean()
        std = image.std()
        image = (image - mean) / max(std, 1e-8)
    return image


def torch_resample_to_spacing(
    data: np.ndarray,
    current_spacing: tuple,
    new_spacing: tuple,
    is_seg: bool = False,
    order: int = 3,
    device: str = 'cuda',
) -> np.ndarray:
    """
    GPU-accelerated resampling using PyTorch.

    Args:
        data: Input array of shape (C, X, Y, Z)
        current_spacing: Current voxel spacing (tuple of 3 floats)
        new_spacing: Target voxel spacing (tuple of 3 floats)
        is_seg: Whether this is a segmentation mask
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)
        device: Device to use for computation ('cuda' or 'cpu')

    Returns:
        Resampled array of shape (C, X', Y', Z')
    """
    assert data.ndim == 4, "data must be (c, x, y, z)"

    # Compute new shape based on spacing
    old_shape = np.array(data.shape[1:])  # (X, Y, Z)
    current_spacing = np.array(current_spacing)
    new_spacing = np.array(new_spacing)

    new_shape = np.round(old_shape * current_spacing / new_spacing).astype(int)

    # Convert to torch tensor and move to device
    data_torch = torch.from_numpy(data).float().to(device)

    # Add batch dimension for F.interpolate: (1, C, X, Y, Z)
    data_torch = data_torch.unsqueeze(0)

    # Choose interpolation mode
    if is_seg or order == 0:
        mode = 'nearest'
        align_corners = None
    elif order == 1:
        mode = 'trilinear'
        align_corners = False
    else:  # order == 3 or higher, use trilinear as best approximation
        mode = 'trilinear'
        align_corners = False

    # Perform resampling
    resampled = F.interpolate(
        data_torch,
        size=tuple(new_shape),
        mode=mode,
        align_corners=align_corners
    )

    # Remove batch dimension and convert back to numpy
    resampled = resampled.squeeze(0).cpu().numpy()

    return resampled


# Custom transform for loading images with SimpleITK (matching nnssl pipeline)
class SimpleITKLoadImaged(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.reader = SimpleITKIO()

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img_path = d[key]
            # Use SimpleITKIO to read image (matches nnssl preprocessing)
            image_data, properties = self.reader.read_images([img_path])
            d[key] = image_data
            d[f'{key}_properties'] = properties
        return d

# make mask_original that just copies the mask without any processing
class CopyMaskd(MapTransform):
    def __init__(self, keys, mask_key, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key

    def __call__(self, data):
        d = dict(data)
        for key in self.mask_key:
            mask = d[key]
            d[f'{key}_original'] = mask.copy()
        return d

# Custom transform for resampling (GPU-accelerated with torch)
class ResampleToSpacingd(MapTransform):
    """
    GPU-accelerated resampling in torch. Doesn't include the complex anisotropic separate_z resampling logic used in nnssl & nnUNet
    """
    def __init__(self, keys, target_spacing=(1.0, 1.0, 1.0), order=3, allow_missing_keys=False, is_seg_list=None, device='cuda'):
        super().__init__(keys, allow_missing_keys)
        self.target_spacing = target_spacing
        self.order = order
        self.is_seg_list = is_seg_list
        self.device = device if torch.cuda.is_available() else 'cpu'
        assert len(keys) == len(is_seg_list), "Length of keys and is_seg_list must match"

    def __call__(self, data):
        d = dict(data)
        for key, is_seg in zip(self.key_iterator(d), self.is_seg_list):
            image = d[key]
            properties = d.get(f'{key}_properties', None)
            if properties is None:
                raise ValueError(f"Properties not found for {key}. Make sure SimpleITKLoadImaged was run first.")

            current_spacing = properties['spacing']

            # GPU-accelerated resampling using torch
            resampled = torch_resample_to_spacing(
                data=image,
                current_spacing=current_spacing,
                new_spacing=self.target_spacing,
                is_seg=is_seg,
                order=self.order,
                device=self.device
            )
            d[key] = resampled
        return d


# Custom transform for Z-score normalization (matching nnssl pipeline)
class ZScoreNormalizationd(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[key]
            # print(image.shape) # (1, 390, 400, 400)
            # Ensure image is in correct shape (C, X, Y, Z)
            if image.ndim == 3:
                image = image[None]  # Add channel dimension

            # Create a simple foreground mask (non-zero regions)
            # For each channel, create mask and normalize
            normalized = np.zeros_like(image, dtype=np.float32)
            image, non_zero_mask, bbox = crop_to_nonzero(image, None)
            
            # Apply z-score normalization
            normalized = z_score_normalization(
                image=image,
                use_mask_for_norm=True,
                non_zero_mask=non_zero_mask[0],
                target_dtype=np.float32
            )
            d[key] = normalized
        return d  
  
  
if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    # Docker-compatible arguments (short form for convenience)
    ap.add_argument("-i", "--input", "--imgs_path", dest="imgs_path", type=str,
                    default='/workspace/inputs',
                    help='Path to input images directory')
    ap.add_argument("-o", "--output", "--dest", dest="dest", type=str,
                    default='/workspace/outputs',
                    help='Destination folder to save features')

    # Optional paths with Docker-friendly defaults
    ap.add_argument("--masks_path", type=str, default=None,
                    help='Path to foreground masks for roi-disease (set to None for non-roi diseases)')
    ap.add_argument("--checkpoint", type=str, default='./work_dir/CT-NEXUS/fold_all/checkpoint_final.pth',
                    help='Path to model checkpoint')

    # Processing parameters
    ap.add_argument("--batch_size", type=int, default=1, help='Batch size for feature extraction') #8
    ap.add_argument("--dump_dir", type=str, default=None, help='Directory to save debug images and masks')
    ap.add_argument("--num_workers", type=int, default=0, help='Number of workers for data loading') #8
    ap.add_argument("--num_classes", type=int, default=2, help='Number of classes for classification')

    args = ap.parse_args()

    # Create dump directory for debugging
    if args.dump_dir is not None:
        os.makedirs(args.dump_dir, exist_ok=True)

    # Load the trained model
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Extract plan from checkpoint to get configuration
    plan_path = os.path.join(os.path.dirname(os.path.dirname(args.checkpoint)), 'plans.json')
    plan = Plan.load_from_file(plan_path)
    # Initialize the trainer to get the network architecture
    configuration_name = 'onemmiso'
    model = ResEncoderPatchLatent(
        num_classes=args.num_classes,
        chpt_path=args.checkpoint,
        pool_size=16 if args.masks_path is not None else 32,  # Use smaller pool size for roi-disease to get smaller latent feature maps
    )
    model.eval()

    model.to(device)

    # get only encoder of the model and mean pool (bs, c, d, h, w) acrodd d, h, w

    imgs_path = args.imgs_path
    os.makedirs(args.dest, exist_ok=True)

    # Process all image files in the input directory
    datalist = []
    imgs_files = sorted([f for f in os.listdir(imgs_path) if f.endswith('.nii.gz')])
    if args.masks_path:
        imgs_files = [f for f in imgs_files if os.path.exists(os.path.join(args.masks_path, f))]

    for img_file in imgs_files:
        img_id = img_file.split('.nii.gz')[0]
        img_full_path = os.path.join(imgs_path, img_file)
        # Construct path to foreground mask
        mask_full_path = os.path.join(args.masks_path, img_file) if args.masks_path is not None else None
        assert os.path.exists(img_full_path), f'Image file not found: {img_full_path}'
        if mask_full_path is not None:
            assert os.path.exists(mask_full_path), f'Mask file not found: {mask_full_path}'
            datalist.append({
                "image": img_full_path,
                "mask": mask_full_path,
                'filename': img_id
            })
        else:
            datalist.append({
                "image": img_full_path,
                'filename': img_id
            })
    #datalist = datalist[:1] # for debugging

    # Preprocessing pipeline matching default_preprocessor.py with ZScore normalization
    # Uses SimpleITK IO and nnssl resampler (matching the training preprocessing)
    if args.masks_path is None:
        ImageTransforms = Compose([
            # Load images using SimpleITK (matching nnssl pipeline)
            SimpleITKLoadImaged(keys=["image"]),
            #CopyMaskd(keys=["mask"], mask_key=["mask"]),
            # Z-score normalization (matching default_preprocessor with ZScoreNormalization)
            ZScoreNormalizationd(keys=["image"]),
            # Resample to isotropic 1mm spacing using nnssl resampler
            ResampleToSpacingd(keys=["image"], target_spacing=(1.0, 1.0, 1.0), order=3, is_seg_list=[False]),
            # Center crop to 320x320x320
            #CenterSpatialCropd(keys=["image"], roi_size=[320, 320, 320]),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=[320, 320, 320]),
            ToTensord(keys=["image"]),
        ])
    else:
        ImageTransforms = Compose([
            # Load images using SimpleITK (matching nnssl pipeline)
            SimpleITKLoadImaged(keys=["image", "mask"]),
            CopyMaskd(keys=["mask"], mask_key=["mask"]),
            # Load binary foreground mask using standard MONAI loader
            #LoadImaged(keys=["mask"], image_only=True),
            # Z-score normalization (matching default_preprocessor with ZScoreNormalization)
            ZScoreNormalizationd(keys=["image"]),
            # Resample to isotropic 1mm spacing using nnssl resampler
            ResampleToSpacingd(keys=["image", "mask"], target_spacing=(1.0, 1.0, 1.0), order=3, is_seg_list=[False, True]),
            # Crop image based on foreground mask (before normalization)
            #CropForegroundd(keys=["image"], source_key="mask", margin=10),
            MaskCenterCropd(keys=["image", "mask"], mask_key="mask", roi_size=(160, 160, 160), fg_labels=[1]),
            #MaskCenterCropd(keys=["mask"], mask_key="mask", roi_size=(160, 160, 160)),
            # Resize/pad to 160x160x160
            ResizeWithPadOrCropd(keys=["image", "mask"], spatial_size=[160, 160, 160]),
            DeleteItemsd(keys=["mask_original"]),  # Remove original mask to save memory
            ToTensord(keys=["image", "mask"]),
        ])


    # Create dataloader
    from monai.data import DataLoader as MonaiDataLoader, Dataset
    from monai.data import ThreadDataLoader
    dataset = Dataset(data=datalist, transform=ImageTransforms)
    dataloader = ThreadDataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Extract embeddings and save immediately after each batch
    processed_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Handle both dict and list outputs from DataLoader
            if isinstance(batch, dict):
                images = batch["image"]
                masks = batch.get("mask", None)
                filenames = batch.get('filename', [f'batch_{batch_idx}_sample_{i}' for i in range(images.shape[0])])
            else:
                raise ValueError(f"Expected batch to be a dict, but got {type(batch)}. Please ensure that the DataLoader returns a dictionary with keys 'image', 'mask', and 'filename'.")
                images = batch[0]
                masks = None
                filenames = [f'batch_{batch_idx}_sample_{i}' for i in range(images.shape[0])]

            # Ensure filenames is a list
            if not isinstance(filenames, (list, tuple)):
                filenames = [filenames]
                raise ValueError(f"Expected 'filename' to be a list, but got {type(filenames)}. Please ensure that the DataLoader returns a list of filenames for each batch.")
            assert len(filenames) == images.shape[0], f'Number of filenames {len(filenames)} does not match batch size {images.shape[0]}'

            #print(f"Processing batch {batch_idx + 1}/{len(dataloader)} with {len(filenames)} samples")
            #print(f"Input shape: {images.shape}")

            # Save debug images and masks as nii.gz files
            if args.dump_dir:
                for i, filename in enumerate(filenames):
                    # Convert tensors to numpy arrays (remove batch dimension)
                    image_np = images[i].cpu().numpy()

                    # Remove channel dimension if it's 1 (SimpleITK expects 3D array for 3D images)
                    if image_np.shape[0] == 1:
                        image_np = image_np[0]

                    # Convert to SimpleITK image and save
                    image_sitk = sitk.GetImageFromArray(image_np)  # SimpleITK expects Z, Y, X order
                    image_sitk.SetSpacing((1.0, 1.0, 1.0))  # Set spacing to 1mm isotropic
                    image_output_path = os.path.join(args.dump_dir, f'{filename}_image.nii.gz')
                    sitk.WriteImage(image_sitk, image_output_path)

                    # Save mask if available
                    if masks is not None:
                        mask_np = masks[i].cpu().numpy() if torch.is_tensor(masks[i]) else masks[i]

                        # Remove channel dimension if it's 1
                        if mask_np.shape[0] == 1:
                            mask_np = mask_np[0]

                        # Convert to SimpleITK image and save
                        mask_sitk = sitk.GetImageFromArray(mask_np)
                        mask_sitk.SetSpacing((1.0, 1.0, 1.0))
                        mask_output_path = os.path.join(args.dump_dir, f'{filename}_mask.nii.gz')
                        sitk.WriteImage(mask_sitk, mask_output_path)

            # Move to device
            images = images.to(device, non_blocking=True)

            # Forward pass through the model to get image latent features
            image_embeddings = model(images)
            image_embeddings = image_embeddings.detach().cpu()

            # Save h5 files immediately for this batch
            for i, filename in enumerate(filenames):
                single_out_path = os.path.join(args.dest, f'{filename}.h5')

                # Save in h5 format
                with h5py.File(single_out_path, 'w') as hf:
                    #hf.create_dataset('y', data=np.array(labels[i]))
                    hf.create_dataset('y_hat', data=image_embeddings[i].numpy())

                processed_count += 1

            # Clean up memory immediately after saving
            del images, image_embeddings
            torch.cuda.empty_cache()