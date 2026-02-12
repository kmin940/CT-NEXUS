import os
import SimpleITK as sitk
import monai.transforms as mt
import numpy as np
import blosc2
from glob import glob
import random
random.seed(42)

# b2r = blosc2.open(urlpath=src_path, mode="r", dparams={"nthreads": 1}, mmap_mode=None)
# /cluster/projects/bwanggroup/sumin/nnssl_data/nnssl_preprocessed/Dataset001_MerlinTrain/nnsslPlans_onemmiso/Dataset001_MerlinTrain/Dataset001_MerlinTrain/AC423bdbd/ses-DEFAULT

root = '/cluster/projects/bwanggroup/sumin/nnssl_data/nnssl_preprocessed/Dataset001_MerlinTrain/nnsslPlans_onemmiso/Dataset001_MerlinTrain/Dataset001_MerlinTrain'
out_dir = '/cluster/projects/bwanggroup/sumin/visualize_random'
os.makedirs(out_dir, exist_ok=True)

nii_files = sorted(glob(os.path.join(root, '**', '*.b2nd'), recursive=True))
# sample 10
nii_files = random.sample(nii_files, 10)

center_crop = mt.CenterSpatialCrop(roi_size=(-1, 320, 320))

for nii_file in nii_files:
    print(f"Processing {nii_file}...")
    img_path = os.path.join(root, nii_file)

    # --- load ---
    #img = sitk.ReadImage(img_path)
    #print("Original spacing:", img.GetSpacing(), "size:", img.GetSize())

    # --- resample to 1mm ---
    #img_1mm = resample_sitk_to_spacing(img, new_spacing=(1.0, 1.0, 1.0))
    #print("Resampled spacing:", img_1mm.GetSpacing(), "size:", img_1mm.GetSize())

    # --- to numpy ---
    #arr = sitk.GetArrayFromImage(img)     # (Z, Y, X)
    arr = blosc2.open(urlpath=img_path, mode="r", dparams={"nthreads": 1}, mmap_mode=None)
    arr = arr[:]                    # (1, Z, Y, X)
    #arr = arr[None, ...]                      # (1, Z, Y, X)

    # --- center crop ---
    arr_cropped = center_crop(arr)
    arr_cropped = arr_cropped[0]                              # (Z, Y, X)

    print("Cropped:", arr.shape)

    # --- back to sitk ---
    file_name = os.path.basename(nii_file)
    out_img = sitk.GetImageFromArray(arr_cropped)
    sitk.WriteImage(out_img, os.path.join(out_dir, file_name.replace('.b2nd', '.nii.gz')))
    print("Saved to:", os.path.join(out_dir, file_name.replace('.b2nd', '.nii.gz')))

    # save original image for comparison
    orig_out_path = os.path.join(out_dir, file_name.replace('.b2nd', '_orig.nii.gz'))
    arr = arr[0]  # (Z, Y, X)
    img = sitk.GetImageFromArray(arr)
    sitk.WriteImage(img, orig_out_path)
    print("Saved original to:", orig_out_path)
