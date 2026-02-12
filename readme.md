# Install

### uv
[uv](https://docs.astral.sh/uv/) is a fast Python package installer and resolver.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12
source .venv/bin/activate  
uv pip install -e .
```

## Docker Image Building
```bash
# From CT-NEXUS root
cd CT-NEXUS
docker build -f Dockerfile -t ctnexus .
docker save ctnexus:latest | gzip > ctnexus.tar.gz
```

### Run Container
Place [test_demo](https://huggingface.co/datasets/kmin06/CVPR26-3DCTFMCompetition/tree/main/AMOS-clf-tr-val/test_demo) in the current directory.

```
docker load -i ctnexus.tar.gz
mkdir test_demo_outputs

## for Non-ROI disease
docker container run --gpus "device=0" -m 32G --name ctnexus --rm -v $PWD/test_demo/images/:/workspace/inputs/ -v $PWD/test_demo_outputs/:/workspace/outputs/ ctnexus:latest /bin/bash -c "sh extract_feat_LP.sh"

## for ROI disease
docker container run --gpus "device=0" -m 32G --name ctnexus --rm -e MASKS_DIR=/workspace/inputs/fg_masks/adrenal_hyperplasia -v $PWD/test_demo/:/workspace/inputs/ -v $PWD/test_demo_outputs/:/workspace/outputs/ ctnexus:latest /bin/bash -c "sh extract_feat_LP.sh"
``` 

## Usage

### Feature Extraction

Extract features from CT images using pretrained CT-NEXUS model:

**Basic usage (Non-ROI disease):**
```bash
python src/feature_extraction/extract_feat_LP.py \
  -i /path/to/input/images \
  -o /path/to/output/features \
  --checkpoint ./work_dir/CT-NEXUS/fold_all/checkpoint_final.pth
```

**ROI-based disease (with masks):**
```bash
python src/feature_extraction/extract_feat_LP.py \
  -i /path/to/input/images \
  -o /path/to/output/features \
  --masks_path /path/to/foreground/masks \
  --checkpoint ./work_dir/CT-NEXUS/fold_all/checkpoint_final.pth
```

**Parameters:**
- `-i, --input`: Input directory containing `.nii.gz` CT images (required)
- `-o, --output`: Output directory for extracted features (`.h5` files) (required)
- `--masks_path`: Path to foreground masks for ROI-based diseases (optional)
- `--checkpoint`: Path to pretrained model checkpoint (default: `./work_dir/CT-NEXUS/fold_all/checkpoint_final.pth`)
- `--num_classes`: Number of classification classes (default: 2)
- `--batch_size`: Batch size for inference (default: 1)
- `--dump_dir`: Directory to save debug images and masks (optional)
- `--num_workers`: Number of workers for data loading (default: 0)

**Output:**
- Features are saved as `.h5` files in the output directory
- Each image produces one `.h5` file containing the `y_hat` dataset with extracted features

**Preprocessing pipeline:**
- Images are normalized using Z-score normalization
- Resampled to isotropic 1mm spacing
- Non-ROI: Center cropped/padded to 320×320×320
- ROI: Center cropped to 160×160×160 based on mask, then padded to 160×160×160

### Preprocessing

```
export nnssl_raw="/path/to/nnssl_raw"
export nnssl_preprocessed="/path/to/nnssl_preprocessed"
export nnssl_results="/path/to/nnssl_results"

nnssl_plan_and_preprocess -d 001 -c onemmiso -np 60
```

### Pretraining CT-NEXUS:
- Stage 1:
```
nnUNet_n_proc_DA=16 nnssl_train 001 onemmiso -tr HuberMAETrainer_BS24 -p nnsslPlans -num_gpus 4
```

- Stage 2:
```
nnUNet_n_proc_DA=16 nnssl_train 001 onemmiso -tr AlignedHuberFTTrainer_MaxPool_BS20 -p nnsslPlans -num_gpus 4
```


## Pretrained Weights and Docker

Pretrained weights and Docker image are available [here](https://drive.google.com/drive/folders/1VR0u8gvpuYSXSbZEwoq1z169SWztWH0a?usp=drive_link)
