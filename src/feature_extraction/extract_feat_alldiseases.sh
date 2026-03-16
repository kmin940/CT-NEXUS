#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

INPUT_DIR="${INPUT_DIR:-/path/to/AMOS-clf-tr-val/images}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/features_LP_MultiStage}"

disease_list=(
  splenomegaly
  adrenal_hyperplasia
  fatty_liver
  cholecystitis
  liver_calcifications
  hydronephrosis
  gallstone
  liver_lesion
  kidney_stone
  liver_cyst
  renal_cyst
  atherosclerosis
  colorectal_cancer
  ascites
  lymphadenopathy
)
non_roi_disease_list=(
  atherosclerosis
  colorectal_cancer
  ascites
  lymphadenopathy
)
for disease in "${disease_list[@]}"; do
    if [[ " ${non_roi_disease_list[@]} " =~ " ${disease} " ]]; then
      echo "Running feature extraction for non-roi ${disease} ..."
      MASKS_DIR=""
    else
      echo "Running feature extraction for roi ${disease} ..."
      MASKS_DIR="/home/jma/Documents/cryoSumin/CT_FM/data/raw_data_classify/amos-clf-tr-val/fg_masks/${disease}"
    fi

    # Build command with optional masks_path
    CMD="python3 extract_feat_LP.py -i \"$INPUT_DIR\" -o \"$OUTPUT_DIR/${disease}\""

    # Add masks_path argument if MASKS_DIR is set and not empty
    if [ -n "$MASKS_DIR" ]; then
        CMD="$CMD --batch_size 8 --num_workers 8 --masks_path \"$MASKS_DIR\""
    else
        CMD="$CMD --batch_size 1 --num_workers 8"
    fi

    # Run feature extraction
    eval $CMD
done
