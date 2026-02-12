import argparse
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import save_json
from tqdm import tqdm

from nnssl.data.raw_dataset import Dataset, Image, Session, Subject, AssociatedMasks, Collection
from nnssl.paths import nnssl_raw

sep = os.path.sep


def _create_pretrain_json(openmind_root_dir: Path, dat_name: str):
    # path to OpenMind metadata file
    openmind_dir = openmind_root_dir #/ "imagesTr"
    print(openmind_dir)
    #data_csv_path = openmind_root_dir / "openneuro_metadata.csv"

    #data_csv = pd.read_csv(data_csv_path)
    #data = data_csv.to_dict(orient="records")

    # "openmind" environment variable holds the path to the OpenMind dataset download directory
    #dat_name = "Dataset002_SmartBRAINMRI"
    collection = Collection(collection_name=dat_name, collection_index=1)

    #subject_info_keys = ["age", "sex", "handedness", "race", "weight", "bmi", "health_status"]
    # image_info_keys = [
    #     "derived_from",
    #     "is_brain_extract",
    #     "manufacturer",
    #     "model_name",
    #     "phase_encoding_direction",
    #     "magnetic_field_strength",
    #     "repetition_time",
    #     "echo_time",
    # ]

    # Iterate over every image and one by one extend the Collection
    #print(data)
    data = os.listdir(openmind_dir)
    data = [f for f in data if f.endswith('.nii.gz')]
    for file in tqdm(data):
        relative_scan_path = os.path.join(openmind_dir, file)
        modality = "MRI"
        subject_info = {} # {k: dic[k] for k in subject_info_keys if k in dic if not pd.isna(dic[k])}
        image_info = {} #{k: dic[k] for k in image_info_keys if k in dic if not pd.isna(dic[k])}

        #parts = relative_scan_path.split(sep)
        #dataset_id, subject_id = parts[0], parts[1]
        dataset_id = dat_name
        subject_id =file.split('.nii.gz')[0] #'_'.join(file.split('_')[:3])
        session_id = "ses-DEFAULT"
        #if parts[2].startswith("ses-"):
        #    session_id = parts[2]
        # pre_image_part = parts[-2]
        # if "derived_from" in image_info:
        #     image_name = pre_image_part[:-5] + parts[-1]
        # else:
        #     image_name = parts[-1]
        image_name = file

        image_path = relative_scan_path # str(openmind_dir / relative_scan_path)
        print(image_path)

        relative_anon_mask_path = None #dic.get("anon_mask_path")
        relative_anat_mask_path = None #dic.get("anat_mask_path")

        anonymization_mask_path = anatomy_mask_path = None
        if relative_anon_mask_path:
            anonymization_mask_path = str(openmind_dir / relative_anon_mask_path)
        if relative_anat_mask_path:
            anatomy_mask_path = str(openmind_dir / relative_anat_mask_path)

        associated_masks = AssociatedMasks(anonymization_mask=anonymization_mask_path, anatomy_mask=anatomy_mask_path)

        if dataset_id not in collection.datasets:
            collection.datasets[dataset_id] = Dataset(dataset_index=dataset_id, name=None, dataset_info={})

        if subject_id not in collection.datasets[dataset_id].subjects:
            collection.datasets[dataset_id].subjects[subject_id] = Subject(
                subject_id=subject_id, subject_info=subject_info
            )
        if session_id not in collection.datasets[dataset_id].subjects[subject_id].sessions:
            collection.datasets[dataset_id].subjects[subject_id].sessions[session_id] = Session(
                session_id=session_id, images=[]
            )
        collection.datasets[dataset_id].subjects[subject_id].sessions[session_id].images.append(
            Image(
                name=image_name,
                image_path=image_path,
                modality=modality,
                image_info=image_info,
                associated_masks=associated_masks,
            )
        )

    # data_csv["dataset_id"] = data_csv["unique_id"].apply(lambda uid: uid[:8])
    # grouped_iqs_csv = (
    #     data_csv.groupby(["dataset_id", "modality", "derived_from"], dropna=False)
    #     .first()["image_quality_score"]
    #     .reset_index()
    # )
    # dataset_id_tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    # for _, row in grouped_iqs_csv.iterrows():
    #     dataset_id = row["dataset_id"]
    #     modality = row["modality"]
    #     derived_from = row["derived_from"] if not pd.isna(row["derived_from"]) else None
    #     iqs_value = row["image_quality_score"]

    #     dataset_id_tree[dataset_id][modality][derived_from] = iqs_value

    # # insert image quality score info into each dataset_info field,
    # # later on this info is used for the entire IQS filtering
    # for dataset_id, dataset in collection.datasets.items():
    #     dicts = []
    #     modality_tree = dataset_id_tree[dataset_id]
    #     for modality, derived_from_tree in modality_tree.items():
    #         for derived_from, iqs_value in derived_from_tree.items():
    #             dicts.append({"modality": modality, "derived_from": derived_from, "image_quality_score": iqs_value})
    #     dataset.dataset_info["image_quality_score"] = dicts

    pretrain_json = collection.to_dict(relative_paths=True)
    #nnssl_raw = '/home/sumin/Documents/cryoSumin/nnUNet_data/nnUNet_raw'
    pretrain_json_path = Path(nnssl_raw, dat_name, "pretrain_data.json")
    pretrain_json_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(pretrain_json, pretrain_json_path, indent=4, sort_keys=True)
    print(f"Successfully saved the OpenMind pretrain_data.json at {pretrain_json_path}")

# python -m nnssl.dataset_conversion.Dataset001_CryoET --openmind_root_dir /home/sumin/Documents/cryoSumin/nnUNet_data/nnUNet_raw/Dataset126_CAIIRadius4Class7

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--openmind_root_dir",
        type=Path,
        required=True,
        help="Path to the root directory of the OpenMind dataset download directory. "
             "If you downloaded the dataset from hugginface, you should point to the parent directory of the `openneuro_metadata.csv` file.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Dataset002_SmartBRAIN",
        help="Name of the dataset to be created inside nnUNet_raw",
    )
    args = parser.parse_args()
    _create_pretrain_json(args.openmind_root_dir, args.dataset_name)


if __name__ == "__main__":
    main()