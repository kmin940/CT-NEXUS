from nnssl.preprocessing.cropping.cropping import crop_to_1024
import SimpleITK as sitk
import os
import multiprocessing as mp

src_file = '/scratch/kmin940/FLARE_Task4_CT_FM/train_all'
dst = '/scratch/kmin940/FLARE_Task4_CT_FM/train_1024'
os.makedirs(dst, exist_ok=True)
files = os.listdir(src_file)

files = [f for f in files if f.endswith('.nii.gz') and 'autoPET' in f]
def process(file):
    print(file)
    img = sitk.ReadImage(os.path.join(src_file, file))
    img_array = sitk.GetArrayFromImage(img)
    img_array = img_array[None]  # add channel axis
    cropped, _, _ = crop_to_1024(img_array, None)
    # assert no dim is 0
    assert all([s > 0 for s in cropped.shape]), f"cropped shape has 0 in it: {cropped.shape}"
    cropped_img = sitk.GetImageFromArray(cropped[0])
    cropped_img.SetOrigin(img.GetOrigin())
    cropped_img.SetSpacing(img.GetSpacing())
    cropped_img.SetDirection(img.GetDirection())
    sitk.WriteImage(cropped_img, os.path.join(dst, file))

if __name__ == '__main__':
    with mp.Pool(35) as p:
        p.map(process, files)