from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
    CenterSpatialCropd,
    MapTransform,
)
import os
import glob
import numpy as np
import SimpleITK as sitk
from multiprocessing import Pool
from functools import partial
import torch

ImageTransforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        # Orientationd(keys=["image"], axcodes="RAS"),
        # Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
        # ScaleIntensityRanged(
        #     keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        # ),
        # SpatialPadd(keys=["image"], spatial_size=[224, 224, 160]),
        # CenterSpatialCropd(
        #     roi_size=[224, 224, 160],
        #     keys=["image"],
        # ),
        # ToTensord(keys=["image"]),
    ]
)




def load_image_np(path):
    """Load NIfTI image and return NumPy array and reference image for metadata."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    return arr, img


def get_mask_center(mask_arr):
    """Return (z, y, x) center of non-zero mask voxels."""
    coords = np.argwhere(mask_arr == 1)
    if coords.size == 0:
        raise ValueError("Mask is empty (no voxels with value 1).")
    return tuple(coords.mean(axis=0).astype(int))


def crop_center_with_padding_np(arr, center, size=(160, 160, 160)):
    """Crop a 3D NumPy array with zero padding around a center (z, y, x)."""
    zc, yc, xc = center
    dz, dy, dx = size[0] // 2, size[1] // 2, size[2] // 2

    z_start, z_end = zc - dz, zc + dz
    y_start, y_end = yc - dy, yc + dy
    x_start, x_end = xc - dx, xc + dx

    cropped = np.zeros(size, dtype=arr.dtype)

    z_start_valid = max(z_start, 0)
    y_start_valid = max(y_start, 0)
    x_start_valid = max(x_start, 0)

    z_end_valid = min(z_end, arr.shape[0])
    y_end_valid = min(y_end, arr.shape[1])
    x_end_valid = min(x_end, arr.shape[2])

    z_off = z_start_valid - z_start
    y_off = y_start_valid - y_start
    x_off = x_start_valid - x_start

    cropped[
    z_off:z_off + (z_end_valid - z_start_valid),
    y_off:y_off + (y_end_valid - y_start_valid),
    x_off:x_off + (x_end_valid - x_start_valid)
    ] = arr[
        z_start_valid:z_end_valid,
        y_start_valid:y_end_valid,
        x_start_valid:x_end_valid
        ]

    return cropped

import numpy as np  
import SimpleITK as sitk  
from typing import Dict, Hashable, Mapping  
  

class MaskCenterTorchCropd(MapTransform):
    """GPU-accelerated MONAI transform to crop around mask center with padding using torch."""

    def __init__(self, keys, mask_key="mask", roi_size=(224, 224, 160), fg_labels=None, device='cuda'):
        super().__init__(keys)
        self.mask_key = mask_key
        self.roi_size = roi_size
        self.fg_labels = fg_labels
        self.img_key = 'image'
        self.device = device if torch.cuda.is_available() else 'cpu'

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)

        # Convert mask to torch tensor if not already
        mask_arr = d[self.mask_key]
        assert torch.is_tensor(mask_arr), f"Mask must be a torch tensor for MaskCenterTorchCropd, but got {type(mask_arr)}. Please ensure that the mask is converted to a torch tensor before this transform."
        # if not torch.is_tensor(mask_arr):
        #     mask_arr = torch.from_numpy(mask_arr).to(self.device)
        # else:
        #     mask_arr = mask_arr.to(self.device)

        if len(mask_arr.shape) == 4:  # Remove channel dimension if present
            mask_arr = mask_arr[0]

        # Make binary mask for specified foreground labels
        if self.fg_labels is not None:
            # Use torch operations for mask creation
            fg_mask = torch.zeros_like(mask_arr, dtype=torch.uint8)
            for label in self.fg_labels:
                fg_mask = fg_mask | (mask_arr == label)
            mask_arr = fg_mask

            coords = torch.nonzero(mask_arr == 1, as_tuple=False)

            if coords.size(0) == 0:
                print(f"Mask is empty, using fallback to original mask")
                # Fall back to original mask
                mask_arr_orig = d['mask_original']
                if not torch.is_tensor(mask_arr_orig):
                    mask_arr_orig = torch.from_numpy(mask_arr_orig).to(self.device)
                else:
                    mask_arr_orig = mask_arr_orig.to(self.device)

                if len(mask_arr_orig.shape) == 4:
                    mask_arr_orig = mask_arr_orig[0]

                fg_mask_orig = torch.zeros_like(mask_arr_orig, dtype=torch.uint8)
                for label in self.fg_labels:
                    fg_mask_orig = fg_mask_orig | (mask_arr_orig == label)

                shape_ori = mask_arr_orig.shape
                shape_resampled = mask_arr.shape
                coords = torch.nonzero(fg_mask_orig == 1, as_tuple=False)

                if coords.size(0) == 0:
                    print(f"================== No coordinates found in original mask, using image center.")
                    center = (shape_resampled[0]//2, shape_resampled[1]//2, shape_resampled[2]//2)
                else:
                    # Scale coordinates to resampled shape
                    scale_z = shape_resampled[0] / shape_ori[0]
                    scale_y = shape_resampled[1] / shape_ori[1]
                    scale_x = shape_resampled[2] / shape_ori[2]
                    coords = coords.float()
                    coords[:, 0] *= scale_z
                    coords[:, 1] *= scale_y
                    coords[:, 2] *= scale_x
                    coords = coords.long()
                    center = tuple(coords.float().mean(dim=0).long().cpu().tolist())
            else:
                center = tuple(coords.float().mean(dim=0).long().cpu().tolist())
        else:
            # Center cropping if no fg_labels provided
            img_arr = d[self.img_key]
            if not torch.is_tensor(img_arr):
                img_arr = torch.from_numpy(img_arr)
            shape_img = img_arr.shape[1:] if len(img_arr.shape) == 4 else img_arr.shape
            center = (shape_img[0]//2, shape_img[1]//2, shape_img[2]//2)

        # Crop each key around the mask center
        for key in self.keys:
            arr = d[key]

            # Convert to torch tensor if not already
            if not torch.is_tensor(arr):
                arr = torch.from_numpy(arr).to(self.device)
            else:
                arr = arr.to(self.device)

            has_channel = len(arr.shape) == 4
            if has_channel:
                arr_data = arr[0]  # Remove channel for processing
            else:
                arr_data = arr

            cropped = self._crop_with_padding_torch(arr_data, center, self.roi_size)

            if has_channel:
                d[key] = cropped.unsqueeze(0)  # Add channel back
            else:
                d[key] = cropped

        return d

    def _crop_with_padding_torch(self, arr: torch.Tensor, center: tuple, size: tuple) -> torch.Tensor:
        """Crop 3D tensor with zero padding around center (z, y, x) - GPU accelerated."""
        zc, yc, xc = center
        dz, dy, dx = size[0] // 2, size[1] // 2, size[2] // 2

        z_start, z_end = zc - dz, zc + dz
        y_start, y_end = yc - dy, yc + dy
        x_start, x_end = xc - dx, xc + dx

        cropped = torch.zeros(size, dtype=arr.dtype, device=arr.device)

        z_start_valid = max(z_start, 0)
        y_start_valid = max(y_start, 0)
        x_start_valid = max(x_start, 0)

        z_end_valid = min(z_end, arr.shape[0])
        y_end_valid = min(y_end, arr.shape[1])
        x_end_valid = min(x_end, arr.shape[2])

        z_off = z_start_valid - z_start
        y_off = y_start_valid - y_start
        x_off = x_start_valid - x_start

        cropped[
            z_off:z_off + (z_end_valid - z_start_valid),
            y_off:y_off + (y_end_valid - y_start_valid),
            x_off:x_off + (x_end_valid - x_start_valid)
        ] = arr[
            z_start_valid:z_end_valid,
            y_start_valid:y_end_valid,
            x_start_valid:x_end_valid
        ]

        return cropped


class MaskCenterCropd(MapTransform):
    """Custom MONAI transform to crop around mask center with padding."""

    def __init__(self, keys, mask_key="mask", roi_size=(224, 224, 160), fg_labels=None):
        super().__init__(keys)
        self.mask_key = mask_key
        self.roi_size = roi_size
        self.fg_labels = fg_labels
        self.img_key = 'image'
      
    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:  
        d = dict(data)  
          
        # Get mask center  
        mask_arr = d[self.mask_key]  
        if len(mask_arr.shape) == 4:  # Remove channel dimension if present  
            mask_arr = mask_arr[0]  
        
        # make binary mask for specified foreground labels
        if self.fg_labels is not None:
            #print(f"Using foreground labels: {self.fg_labels}")
            mask_arr = np.isin(mask_arr, self.fg_labels).astype(np.uint8)

            coords = np.argwhere(mask_arr == 1)  
            if coords.size == 0:  
                print(f"Mask unique values: {np.unique(mask_arr)}")
                print(f'Original unique values in mask: {np.unique(d[self.mask_key])}')
                print(f'shapes of image and mask: image {d["image"].shape=}, mask {d[self.mask_key].shape=}')
                #raise ValueError(f"Mask is empty (no voxels with value 1) for data {d['filename']}.")  
                # fall back to original mask to get coordinates
                mask_arr_orig = d['mask_original']
                if len(mask_arr_orig.shape) == 4:  # Remove channel dimension if present  
                    mask_arr_orig = mask_arr_orig[0]
                mask_arr_orig = np.isin(mask_arr_orig, self.fg_labels).astype(np.uint8)
                shape_ori = mask_arr_orig.shape
                shape_resampled = mask_arr.shape
                coords = np.argwhere(mask_arr_orig == 1)
                #assert coords.size > 0, f"Original mask is also empty for data {d['filename']}."
                # if no coordinates found, just use center of image
                if coords.size == 0:
                    #center = (shape_resampled[0]//2, shape_resampled[1]//2, shape_resampled[2]//2)
                    print(f"================== No coordinates found in original mask, using image center.")
                    coords = np.array([[shape_resampled[0]//2, shape_resampled[1]//2, shape_resampled[2]//2]])
                else:
                    # scale the coordinates to resampled shape
                    scale_z = shape_resampled[0] / shape_ori[0]
                    scale_y = shape_resampled[1] / shape_ori[1]
                    scale_x = shape_resampled[2] / shape_ori[2]
                    coords = np.array([[int(c[0]*scale_z), int(c[1]*scale_y), int(c[2]*scale_x)] for c in coords])
            center = tuple(coords.mean(axis=0).astype(int))  

        else:
            # center cropping if no fg_labels provided
            img_arr = d[self.img_key]
            shape_img = img_arr.shape[1:] if len(img_arr.shape) == 4 else img_arr.shape
            center = (shape_img[0]//2, shape_img[1]//2, shape_img[2]//2)
            
        # Crop each key around the mask center  
        for key in self.keys:  
            arr = d[key]  
            has_channel = len(arr.shape) == 4  
            if has_channel:  
                arr_data = arr[0]  # Remove channel for processing  
            else:  
                arr_data = arr  
            #print(f"Cropping key '{key}' with shape {arr_data.shape} around center {center} to roi_size {self.roi_size} and filename {d.get('filename', 'N/A')}")
            #import pdb; pdb.set_trace()
            cropped = self._crop_with_padding(arr_data, center, self.roi_size)  
              
            if has_channel:  
                d[key] = cropped[np.newaxis, ...]  # Add channel back  
            else:  
                d[key] = cropped  

        # if self.fg_labels and coords.size > 0:
        #     # make sure cropped mask has non-zero values
        #     cropped_mask = d[self.mask_key]
        #     assert cropped_mask.max() > 0, f"Cropped mask is empty for data {d['filename']} after cropping around center {center} with roi_size {self.roi_size}."
          
        return d  
    # def __call__(self, data):
    #     d = dict(data)
    #     mask = d[self.mask_key]

    #     if torch.is_tensor(mask):
    #         mask_np = mask.cpu().numpy()
    #     else:
    #         mask_np = mask

    #     center = self._get_center_from_mask(mask_np)

    #     for key in self.keys:
    #         arr = d[key]
    #         is_tensor = torch.is_tensor(arr)
    #         if is_tensor:
    #             arr_np = arr.cpu().numpy()
    #         else:
    #             arr_np = arr

    #         cropped_np = self._crop_with_padding(arr_np, center, self.roi_size)

    #         if is_tensor:
    #             d[key] = torch.from_numpy(cropped_np).to(arr.dtype).to(arr.device)
    #         else:
    #             d[key] = cropped_np

    #     return d
    # def _get_center_from_mask(self, mask_arr):
    #     """Return (z, y, x) center of non-zero mask voxels."""
    #     coords = np.argwhere(mask_arr == 1)
    #     if coords.size == 0:
    #         raise ValueError("Mask is empty (no voxels with value 1).")
    #     return tuple(coords.mean(axis=0).astype(int))
      
    def _crop_with_padding(self, arr, center, size):  
        """Crop 3D array with zero padding around center (z, y, x)."""  
        zc, yc, xc = center  
        dz, dy, dx = size[0] // 2, size[1] // 2, size[2] // 2  
          
        z_start, z_end = zc - dz, zc + dz  
        y_start, y_end = yc - dy, yc + dy  
        x_start, x_end = xc - dx, xc + dx  
          
        if torch.is_tensor(arr):
            cropped = torch.zeros(size, dtype=arr.dtype, device=arr.device)
        else:
            cropped = np.zeros(size, dtype=arr.dtype)  
          
        z_start_valid = max(z_start, 0)  
        y_start_valid = max(y_start, 0)  
        x_start_valid = max(x_start, 0)  
          
        z_end_valid = min(z_end, arr.shape[0])  
        y_end_valid = min(y_end, arr.shape[1])  
        x_end_valid = min(x_end, arr.shape[2])  
          
        z_off = z_start_valid - z_start  
        y_off = y_start_valid - y_start  
        x_off = x_start_valid - x_start  
          
        cropped[  
            z_off:z_off + (z_end_valid - z_start_valid),  
            y_off:y_off + (y_end_valid - y_start_valid),  
            x_off:x_off + (x_end_valid - x_start_valid)  
        ] = arr[  
            z_start_valid:z_end_valid,  
            y_start_valid:y_end_valid,  
            x_start_valid:x_end_valid  
        ]  
          
        return cropped  
  
  
# ImageTransformsCropMaskCenter = Compose(  
#     [  
#         LoadImaged(keys=["image", "mask"]),  
#         EnsureChannelFirstd(keys=["image", "mask"]),  
#         Orientationd(keys=["image", "mask"], axcodes="RAS"),  
#         Spacingd(keys=["image", "mask"], pixdim=(1.5, 1.5, 3), mode=("bilinear", "nearest")),  
#         ScaleIntensityRanged(  
#             keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True  
#         ),  
#         MaskCenterCropd(keys=["image"], mask_key="mask", roi_size=[224, 224, 160], fg_labels=[1]),  
#         ToTensord(keys=["image"]),  
#     ]  
# )

# Create dummy data  
def create_dummy_data(output_dir="./dummy_data"):  
    """Create dummy CT image and mask for testing."""  
    os.makedirs(output_dir, exist_ok=True)  
      
    # Create dummy CT image (300x300x200 voxels)  
    ct_array = np.random.randint(-1000, 1000, size=(200, 300, 300), dtype=np.int16)  
    ct_image = sitk.GetImageFromArray(ct_array)  
    ct_image.SetSpacing((1.0, 1.0, 2.0))  # 1mm x 1mm x 2mm spacing  
    ct_path = os.path.join(output_dir, "dummy_ct.nii.gz")  
    sitk.WriteImage(ct_image, ct_path)  
      
    # Create dummy mask with a sphere in the center  
    mask_array = np.zeros((200, 300, 300), dtype=np.uint8)  
    center_z, center_y, center_x = 100, 150, 150  
    radius = 30  
      
    for z in range(200):  
        for y in range(300):  
            for x in range(300):  
                if (z - center_z)**2 + (y - center_y)**2 + (x - center_x)**2 < radius**2:  
                    mask_array[z, y, x] = 1  
      
    mask_image = sitk.GetImageFromArray(mask_array)  
    mask_image.SetSpacing((1.0, 1.0, 2.0))  
    mask_path = os.path.join(output_dir, "dummy_mask.nii.gz")  
    sitk.WriteImage(mask_image, mask_path)  
      
    return ct_path, mask_path  
  
  
  
# Main execution  
if __name__ == "__main__":  
    print("Creating dummy data...")  
    #ct_path, mask_path = create_dummy_data()  
      
    print(f"Created dummy CT: {ct_path}")  
    print(f"Created dummy mask: {mask_path}")  
      
    # Create data dictionary  
    data_dict = {  
        "image": ct_path,  
        "mask": mask_path,  
    }  
      
    print("\nApplying transforms...")  
    transformed = ImageTransforms(data_dict)  
      
    print("\n=== Output Information ===")  
    print(f"Output image shape: {transformed['image'].shape}")  
    print(f"Output image dtype: {transformed['image'].dtype}")  
    print(f"Output image min/max: {transformed['image'].min():.3f} / {transformed['image'].max():.3f}")  
      
    # Verify it's a PyTorch tensor  
    print(f"\nIs PyTorch tensor: {isinstance(transformed['image'], torch.Tensor)}")  
      
    print("\n✓ Pipeline executed successfully!")