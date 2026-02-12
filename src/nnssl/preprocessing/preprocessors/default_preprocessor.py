#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from copy import deepcopy
from dataclasses import asdict
from functools import partial
import multiprocessing
from pathlib import Path
from typing import Callable, Literal, Union

from loguru import logger

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *


from nnssl.data.raw_dataset import Collection, IndependentImage
from nnssl.experiment_planning.experiment_planners.plan import (
    ConfigurationPlan,
    Plan,
    PREPROCESS_SPACING_STYLES,
)
from nnssl.paths import nnssl_preprocessed, nnssl_raw
from nnssl.preprocessing.cropping.cropping import crop_to_nonzero

from nnssl.preprocessing.preprocessors.normalize import normalize_arr
from nnssl.preprocessing.preprocessors.no_resampling_preprocessor import (
    no_resample_preprocess_case,
)
from nnssl.preprocessing.resampling.default_resampling import (
    compute_new_shape,
    get_resampling_scheme,
)
from nnssl.data.dataloading.dataset import nnSSLDatasetBlosc2
from nnssl.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnssl.data.utils import get_train_collection
from batchgenerators.utilities.file_and_folder_operations import write_pickle, load_pickle
from copy import deepcopy

def preprocess_case(
    data: np.ndarray,
    masks: list[np.ndarray] | None,
    properties: dict,
    plan: "Plan",
    config_plan: "ConfigurationPlan",
    verbose: bool,
    resample_save=True
):
    # let's not mess up the inputs!
    data = np.copy(data)
    if masks is not None:
        for mask in masks:
            assert (
                data.shape[1:] == mask.shape[1:]
            ), "Shape mismatch between image and associated masks. Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
        masks = [np.copy(mask) for mask in masks]

    has_masks = masks is not None

    # apply transpose_forward, this also needs to be applied to the spacing!
    data = data.transpose([0, *[i + 1 for i in plan.transpose_forward]])
    if has_masks:
        for cnt, mask in enumerate(masks):
            masks[cnt] = mask.transpose([0, *[i + 1 for i in plan.transpose_forward]])
    original_spacing = [properties["spacing"][i] for i in plan.transpose_forward]

    # crop, remember to store size before cropping!
    shape_before_cropping = data.shape[1:]
    properties["shape_before_cropping"] = shape_before_cropping
    # this command will generate a segmentation. This is important because of the nonzero mask which we may need
    data, masks, bbox = crop_to_nonzero(data, masks)
    properties["bbox_used_for_cropping"] = bbox
    properties["shape_after_cropping_and_before_resampling"] = data.shape[1:]

    # resample
    target_spacing = config_plan.spacing  # this should already be transposed

    if len(target_spacing) < len(data.shape[1:]):
        # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
        # in 2d configuration we do not change the spacing between slices
        target_spacing = [original_spacing[0]] + target_spacing
    new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

    # normalize
    # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
    # longer fitting the images perfectly!
    norm_mask = masks[0]
    if config_plan.normalization_schemes == ["ct_multiwindow"]:
        from .normalize import normalize_ct_arr

        data = normalize_ct_arr(
            data,
            norm_mask,
            config_plan.normalization_schemes,
            config_plan.use_mask_for_norm,
        )
        # assert ndim == 4
        assert data.ndim == 4, "data must be 4D after ct_multiwindow normalization"
        assert data.shape[0] == 11
    else:
        # works for 1 channel input, saves mean, std
        if config_plan.normalization_schemes == ["ZScoreNormalization"]:
            data, mean, std = normalize_arr(
                data,
                norm_mask,
                config_plan.normalization_schemes,
                config_plan.use_mask_for_norm,
            )
            properties['means'] = mean
            properties['stds'] = std
        else:
            data = normalize_arr(
                data,
                norm_mask,
                config_plan.normalization_schemes,
                config_plan.use_mask_for_norm,
            )

    if not resample_save:
        return data, masks
    print(f'should not reach')
    old_shape = data.shape[1:]
    resampling_fn = partial(
        get_resampling_scheme(config_plan.resampling_fn_data),
        **config_plan.resampling_fn_data_kwargs,
    )
    data = resampling_fn(data, new_shape, original_spacing, target_spacing)

    if has_masks:
        resampling_mask_fn = partial(
            get_resampling_scheme(config_plan.resampling_fn_mask),
            **config_plan.resampling_fn_mask_kwargs,
        )
        for cnt, mask in enumerate(masks):
            masks[cnt] = resampling_mask_fn(
                mask, new_shape, original_spacing, target_spacing
            )
    if verbose:
        print(
            f"old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, "
            f"new_spacing: {target_spacing}, fn_data: {config_plan.resampling_fn_data}"
        )
    if not has_masks:
        masks = None
    return data, masks


def preprocess_and_save(
    image: IndependentImage,
    output_directory: str,
    plan: Plan,
    config_plan: ConfigurationPlan,
    verbose: bool = True,
    pp_case_func: Callable[
        [np.ndarray, list[np.ndarray] | None, dict, Plan, ConfigurationPlan, bool],
        tuple[np.ndarray, list[np.ndarray] | None],
    ] = preprocess_case,
):
    """Reads the images and their properties, preprocesses them and saves them to disk. (in a compressed npz)"""
    output_image_filename = Path(join(output_directory, image.get_output_path("image")))
    output_anon_filename = Path(
        join(output_directory, image.get_output_path("anon_mask"))
    )
    output_anat_filename = Path(
        join(output_directory, image.get_output_path("anat_mask"))
    )
    output_image_filename.parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(str(output_image_filename) + ".b2nd") and os.path.exists(str(output_image_filename) + ".pkl"):
        resample_save = False
        # if verbose:
        #     print(f"Image {output_image_filename} already exists. Skipping...")
        # return True
    else:
        resample_save = True
    #try:
    #exclude = ['autoPET_fdg_4776e75543_04-01-2007-NA-PET-CT_Ganzkoerper__primaer_mit_KM-74749_', 'autoPET_fdg_1253499c80_10-07-2005-NA-PET-CT_Ganzkoerper__primaer_mit_KM-50242_', 'autoPET_fdg_d626611daf_11-29-2002-NA-PET-CT_Ganzkoerper__primaer_mit_KM-88747_', 'autoPET_fdg_f24f3ce1da_09-14-2001-NA-PET-CT_Ganzkoerper__primaer_mit_KM-27130_' , '/scratch/kmin940/FLARE_Task4_CT_FM/train_all/psma_d67e027e14323000_2019-09-02_']
    #if any([ex in image.image_path for ex in exclude]):
    #    return True
    print(f"Processing {image.image_path}")
    rw = plan.image_reader_writer_class()()
    image_path = image.image_path
    data, data_properties = rw.read_images([image_path])
    # Verify data is not None -- If it is, we discard the image.
    if np.any(np.isnan(data)):
        raise RuntimeError("Found NaNs in the image")
    if np.any(np.isinf(data)):
        raise RuntimeError("Found infs in the image")

    if image.associated_masks is not None:
        masks = [
            rw.read_seg(v)[0]
            for v in asdict(image.associated_masks).values()
            if v is not None
        ]
    else:
        masks = None
    data, masks = pp_case_func(
        data, masks, data_properties, plan, config_plan, verbose, resample_save
    )
    if not resample_save:
        print(data_properties)
        original_properties = load_pickle(str(output_image_filename) + ".pkl")
        comparison_props = deepcopy(data_properties)

        # 3. Verify the new keys exist
        assert 'means' in comparison_props, "Updated properties missing 'mean' key"
        assert 'stds' in comparison_props, "Updated properties missing 'std' key"

        # 4. Remove the new keys so the dictionaries should be identical
        comparison_props.pop('means')
        comparison_props.pop('stds')

        # 5. Assert equality of the remaining content
        assert comparison_props == original_properties, \
            f"Content mismatch! Difference: {set(comparison_props.items()) ^ set(original_properties.items())}"
        write_pickle(data_properties, str(output_image_filename) + ".pkl")
        print('==================== Wrote ', str(output_image_filename) + ".pkl")
        return True
    # print('dtypes', data.dtype, seg.dtype)
    block_size_data, chunk_size_data = nnSSLDatasetBlosc2.comp_blosc2_params(
        data.shape, tuple([160, 160, 160]), data.itemsize
    )
    if masks is not None:
        block_size_seg, chunk_size_seg = nnSSLDatasetBlosc2.comp_blosc2_params(
            data.shape, tuple([160, 160, 160]), data.itemsize
        )
        if image.associated_masks.anatomy_mask is not None:
            anat_mask = masks[0]
        else:
            anat_mask = None
        if image.associated_masks.anonymization_mask is not None:
            anon_mask = masks[-1]
        else:
            anon_mask = None
    else:
        block_size_seg, chunk_size_seg = None, None
        anat_mask, anon_mask = None, None

    nnSSLDatasetBlosc2.save_case(
        data,
        anon_mask,
        anat_mask,
        data_properties,
        str(output_image_filename),
        str(output_anon_filename),
        str(output_anat_filename),
        chunks=chunk_size_data,
        blocks=block_size_data,
        chunks_seg=chunk_size_seg,
        blocks_seg=block_size_seg,
    )
    # except Exception as e:
    #     print(f"Error processing {image_path}: {str(e)}")
    #     return False
    return True


def default_preprocess(
    dataset_name_or_id: Union[int, str],
    configuration_name: str,
    plans_identifier: str,
    part: int,
    total_parts: int,
    num_processes: int,
    verbose: bool = True,
):
    """
    Main function that is called externally.
    Does the preprocessing of the cases found in the dataset_name.
    This is the nnssl version, where we neglect any labels that may be present and create a new dataset.json
    that does not contain label information.
    """
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    assert isdir(
        join(nnssl_raw, dataset_name)
    ), "The requested dataset could not be found in nnssl_raw"

    plans_file = join(nnssl_preprocessed, dataset_name, plans_identifier + ".json")
    assert isfile(plans_file), (
        "Expected plans file (%s) not found. Run corresponding nnUNet_plan_experiment "
        "first." % plans_file
    )
    plan: Plan = Plan.load_from_file(plans_file)
    config_plan: ConfigurationPlan = plan.configurations[configuration_name]

    if verbose:
        print(f"Preprocessing the following configuration: {configuration_name}")
        print(config_plan)

    output_directory = join(
        nnssl_preprocessed, dataset_name, config_plan.data_identifier
    )

    maybe_mkdir_p(output_directory)

    collection: Collection = get_train_collection(join(nnssl_raw, dataset_name))
    pp_collection = deepcopy(collection)
    pp_collection.update_extension(new_extension=".b2nd")
    pp_collection.raw_to_pp_path(data_identifier=config_plan.data_identifier)
    save_json(
        pp_collection.to_dict(relative_paths=True),
        join(
            nnssl_preprocessed,
            dataset_name,
            f"pretrain_data__{configuration_name}.json",
        ),
    )
    # multiprocessing magic.
    spst: PREPROCESS_SPACING_STYLES
    spst = config_plan.spacing_style
    if spst == "noresample":
        pp_func = no_resample_preprocess_case
    elif spst in ["onemmiso", "median"]:
        pp_func = preprocess_case
    else:
        raise NotImplementedError("Unknown")

    preprocess_and_save_partial = partial(
        preprocess_and_save,
        output_directory=output_directory,
        plan=plan,
        config_plan=config_plan,
        verbose=verbose,
        pp_case_func=pp_func,
    )
    all_independent_images: list[IndependentImage] = collection.to_independent_images()
    # ------------------- Optional new splitting into sub-parts ------------------ #
    if total_parts > 1:
        total_images = len(all_independent_images)
        images_per_part = total_images // total_parts
        if part == total_parts - 1:
            all_independent_images = all_independent_images[part * images_per_part :]
        else:
            all_independent_images = all_independent_images[
                part * images_per_part : (part + 1) * images_per_part
            ]

    if num_processes > 1:
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            r = p.map(preprocess_and_save_partial, all_independent_images)
    else:
        r = [preprocess_and_save_partial(image=img) for img in all_independent_images]

    valid_imgs = [img for img, r in zip(all_independent_images, r) if r]

    if total_parts > 1:
        out_filename = join(
            nnssl_preprocessed,
            dataset_name,
            f"valid_imgs__{part}_of_{total_parts}.json",
        )
    else:
        out_filename = join(nnssl_preprocessed, dataset_name, "valid_imgs.json")
    save_json([img.to_dict() for img in valid_imgs], out_filename)

    # ------------------------- Merge problematic images ------------------------- #
    if total_parts > 1:
        all_valid_files = []
        content = os.listdir(join(nnssl_preprocessed, dataset_name))
        for c in content:
            if c.startswith("valid_imgs__") and f"_of_{total_parts}" in c:
                all_valid_files.append(c)
        if len(all_valid_files) == total_parts:
            logger.info("All images have been processed. Merging the results.")
            # all parts have been processed
            valid_images = []
            for f in all_valid_files:
                valid_images += load_json(join(nnssl_preprocessed, dataset_name, f))
            save_json(
                valid_images, join(nnssl_preprocessed, dataset_name, "valid_imgs.json")
            )
            for f in all_valid_files:
                os.remove(join(nnssl_preprocessed, dataset_name, f))

    return


if __name__ == "__main__":
    print("Not intended to be called here!")
