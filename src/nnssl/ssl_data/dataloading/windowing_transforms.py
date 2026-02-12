from typing import Literal, Tuple, List, Union

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform
import os

# ANATOMICAL_WINDOWS = {
#     "CT": {
#         "lung": {"center": -600, "width": 1500},
#         "mediastinum": {"center": 50, "width": 400},
#         "abdomen": {"center": 40, "width": 400},
#         "liver": {"center": 80, "width": 150},
#         "bone": {"center": 400, "width": 1800},
#         "brain": {"center": 40, "width": 80},
#         "subdural": {"center": 75, "width": 215},
#         "stroke": {"center": 40, "width": 40},
#         "temporal_bone": {"center": 600, "width": 2800},
#         "soft_tissue": {"center": 50, "width": 350},
#     }
# }
# def apply_anatomical_window(
#     volume: Union[torch.Tensor, np.ndarray], center: float, width: float
# ) -> Union[torch.Tensor, np.ndarray]:
#     """
#     Apply traditional center/width windowing to medical images.

#     Args:
#         volume: Input volume
#         center: Window center (level)
#         width: Window width

#     Returns:
#         Windowed volume with values in [0, 1]
#     """
#     is_numpy = isinstance(volume, np.ndarray)
#     if is_numpy:
#         volume = torch.from_numpy(volume).float()

#     # Calculate window bounds
#     min_val = center - width / 2
#     max_val = center + width / 2

#     # Apply windowing
#     windowed = (volume - min_val) / (max_val - min_val + 1e-8)
#     windowed = torch.clamp(windowed, 0, 1)

#     if is_numpy:
#         windowed = windowed.numpy()

#     return windowed
# def z_score_normalization(image: np.ndarray) -> np.ndarray:
#     mean = image.mean()
#     std = image.std()
#     image = (image - mean) / max(std, 1e-8)
#     return image


# class ScaleIntensityPercentileTransform(AbstractTransform):
#     def __init__(self, lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, 
#                  clip=False, per_channel=True, data_key='data', p_per_sample=1):
#         self.data_key = data_key
#         self.lower = lower
#         self.upper = upper
#         self.per_channel = per_channel
#         self.p_per_sample = p_per_sample
#         self.property_key = 'properties'

#     def __call__(self, **data_dict):
#         data = data_dict[self.data_key]
#         properties = data_dict['properties']
        
#         for b in range(data.shape[0]):
#             if np.random.uniform() < self.p_per_sample:
#                 mean = properties[b]['means'][0]
#                 std = properties[b]['stds']
                
#                 # Compute percentiles on subsample (8x faster)
#                 #flat = data[b].ravel()[::8] * std + mean
#                 start = np.random.randint(0, 8)
#                 flat = data[b].ravel()[start::8] * std + mean
#                 a_min = np.percentile(flat, self.lower)
#                 a_max = np.percentile(flat, self.upper)
                
#                 # Transform bounds to z-space and clip there (avoids undo/redo)
#                 z_min = (a_min - mean) / std
#                 z_max = (a_max - mean) / std
#                 clipped = np.clip(data[b], z_min, z_max)
                
#                 data[b] = z_score_normalization(clipped)
        
#         data_dict[self.data_key] = data
#         return data_dict


# class ScaleIntensityWindowTransform(AbstractTransform):
#     def __init__(self, data_key='data', p_per_sample=1):
#         self.data_key = data_key
#         self.p_per_sample = p_per_sample
#         self.windows = {
#             "mediastinum": {"center": 50, "width": 400},
#             "abdomen": {"center": 40, "width": 400},
#             "liver": {"center": 80, "width": 150},
#             "bone": {"center": 400, "width": 1800},
#             "soft_tissue": {"center": 50, "width": 350},
#         }

#     def __call__(self, **data_dict):
#         data = data_dict[self.data_key]
#         properties = data_dict['properties']
        
#         for b in range(data.shape[0]):
#             if np.random.uniform() < self.p_per_sample:
#                 mean = properties[b]['means'][0]
#                 std = properties[b]['stds']
                
#                 # Randomly select a window
#                 window_name = np.random.choice(list(self.windows.keys()))
#                 window = self.windows[window_name]
                
#                 w_min = window["center"] - window["width"] / 2
#                 w_max = window["center"] + window["width"] / 2
                
#                 # Transform bounds to z-space (avoids undo/redo of normalization)
#                 z_min = (w_min - mean) / std
#                 z_max = (w_max - mean) / std
#                 clipped = np.clip(data[b], z_min, z_max)
                
#                 data[b] = z_score_normalization(clipped)
        
#         data_dict[self.data_key] = data
#         return data_dict

def z_score_normalization(
    image: np.ndarray,
    #use_mask_for_norm: bool,
    #non_zero_mask: np.ndarray,
    #target_dtype: Type[Number],
) -> np.ndarray:
    #image = convert_dtype(image, target_dtype)
    # if use_mask_for_norm:
    #     mask = non_zero_mask >= 0
    #     mean = image[mask].mean()
    #     std = image[mask].std()
    #     image[mask] = (image[mask] - mean) / max(std, 1e-8)
    # else:
    mean = image.mean()
    std = image.std()
    image = (image - mean) / max(std, 1e-8)
    return image

class ScaleIntensity1000Transform(AbstractTransform):
    def __init__(self, lower=0.5, upper=99.5, a_min=-1000, a_max=1000, 
                 clip=False, per_channel=True, data_key='data', p_per_sample=1):
        """
        Scale intensity based on percentiles.
        
        Args:
            lower: lower percentile (default: 0.5)
            upper: upper percentile (default: 99.5)
            b_min: minimum value after scaling (default: 0.0)
            b_max: maximum value after scaling (default: 1.0)
            clip: whether to clip values outside [b_min, b_max] after scaling (default: False)
            per_channel: whether to compute percentiles per channel (default: True)
            data_key: key to access data in the dictionary (default: 'data')
            p_per_sample: probability of applying transform to each sample (default: 1)
        """
        self.data_key = data_key
        self.lower = lower
        self.upper = upper
        self.a_min = a_min
        self.a_max = a_max
        #self.clip = clip
        self.per_channel = per_channel
        self.p_per_sample = p_per_sample
        self.property_key = 'properties'
        

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        properties = data_dict['properties']

        
        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                mean = properties[b]['means'][0]
                std = properties[b]['stds']
                #import pdb; pdb.set_trace()
                # Compute percentiles across all channels for this sample
                sample_data = data[b]

                # undo z-score norm
                sample_data = sample_data * std + mean
                #a_min = np.percentile(sample_data, self.lower)
                #a_max = np.percentile(sample_data, self.upper)
                sample_data = np.clip(sample_data, self.a_min, self.a_max)

                data[b] = z_score_normalization(sample_data)
    
        data_dict[self.data_key] = data
        return data_dict

class ClipIntensityPercentileTransform(AbstractTransform):
    def __init__(self, lower=0.5, upper=99.5, 
                 per_channel=True, data_key='data', p_per_sample=0.1):
        self.data_key = data_key
        self.lower = lower
        self.upper = upper
        #self.clip = clip
        self.per_channel = per_channel
        self.p_per_sample = p_per_sample
        self.property_key = 'properties'

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        #properties = data_dict['properties']
        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                # clip based on percentiles
                lower_p = np.percentile(data[b], self.lower)
                upper_p = np.percentile(data[b], self.upper)
                data[b] = np.clip(data[b], lower_p, upper_p)
    
        data_dict[self.data_key] = data
        return data_dict
    
class ClipIntensityRandPercentileTransform(AbstractTransform):
    def __init__(self, lower_range=(0.0, 1.0), upper_range=(99.0, 100.0),
                 per_channel=True, data_key='data', p_per_sample=0.1):
        self.data_key = data_key
        self.lower_range = lower_range
        self.upper_range = upper_range
        #self.clip = clip
        self.per_channel = per_channel
        self.p_per_sample = p_per_sample
        self.property_key = 'properties'

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        #properties = data_dict['properties']
        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                lower = np.random.uniform(self.lower_range[0], self.lower_range[1])
                upper = np.random.uniform(self.upper_range[0], self.upper_range[1])
                # clip based on percentiles
                lower_p = np.percentile(data[b], lower)
                upper_p = np.percentile(data[b], upper)
                data[b] = np.clip(data[b], lower_p, upper_p)
    
        data_dict[self.data_key] = data
        return data_dict
    
class ScaleIntensityRandLowerUpperTransform(AbstractTransform):
    def __init__(
        self,
        a_min=-1000,
        a_max=1000,
        lower_scale_range=(0.9, 1.0),   # tighter for air
        upper_scale_range=(0.9, 1.2),   # looser for high HU
        data_key="data",
        p_per_sample=1.0,
    ):
        self.data_key = data_key
        self.a_min = a_min
        self.a_max = a_max
        self.lower_scale_range = lower_scale_range
        self.upper_scale_range = upper_scale_range
        self.p_per_sample = p_per_sample
        self.property_key = "properties"

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        properties = data_dict["properties"]

        base_min, base_max = self.a_min, self.a_max
        center = (base_min + base_max) / 2.0

        for b in range(data.shape[0]):
            if np.random.rand() < self.p_per_sample:
                mean = properties[b]["means"][0]
                std = properties[b]["stds"]

                sample_data = data[b] * std + mean

                lower_scale = np.random.uniform(*self.lower_scale_range)
                upper_scale = np.random.uniform(*self.upper_scale_range)

                new_min = center - (center - base_min) * lower_scale
                new_max = center + (base_max - center) * upper_scale

                sample_data = np.clip(sample_data, new_min, new_max)

                data[b] = z_score_normalization(sample_data)

        data_dict[self.data_key] = data
        return data_dict


class ScaleIntensityRand1000Transform(AbstractTransform):
    def __init__(
        self,
        a_min=-1000,
        a_max=1000,
        scale_range=(0.9, 1.1),
        data_key="data",
        p_per_sample=1.0,
    ):
        self.data_key = data_key
        self.a_min = a_min
        self.a_max = a_max
        self.scale_range = scale_range
        self.p_per_sample = p_per_sample
        self.property_key = "properties"

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        properties = data_dict["properties"]

        base_min, base_max = self.a_min, self.a_max
        center = (base_min + base_max) / 2.0

        for b in range(data.shape[0]):
            if np.random.rand() < self.p_per_sample:
                mean = properties[b]["means"][0]
                std = properties[b]["stds"]

                sample_data = data[b] * std + mean

                scale = np.random.uniform(*self.scale_range)
                new_min = center - (center - base_min) * scale
                new_max = center + (base_max - center) * scale

                sample_data = np.clip(sample_data, new_min, new_max)

                data[b] = z_score_normalization(sample_data)

        data_dict[self.data_key] = data
        return data_dict

    
class ScaleIntensityRand1000TransformNotUsed(AbstractTransform):
    def __init__(self, lower=0.5, upper=99.5, a_min=-1000, a_max=1000, 
                 clip=False, per_channel=True, data_key='data', p_per_sample=1):
        """
        Scale intensity based on percentiles.
        
        Args:
            lower: lower percentile (default: 0.5)
            upper: upper percentile (default: 99.5)
            b_min: minimum value after scaling (default: 0.0)
            b_max: maximum value after scaling (default: 1.0)
            clip: whether to clip values outside [b_min, b_max] after scaling (default: False)
            per_channel: whether to compute percentiles per channel (default: True)
            data_key: key to access data in the dictionary (default: 'data')
            p_per_sample: probability of applying transform to each sample (default: 1)
        """
        self.data_key = data_key
        self.lower = lower
        self.upper = upper
        self.a_min = a_min
        self.a_max = a_max
        #self.clip = clip
        self.per_channel = per_channel
        self.p_per_sample = p_per_sample
        self.property_key = 'properties'
        

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        properties = data_dict['properties']

        
        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                mean = properties[b]['means'][0]
                std = properties[b]['stds']
                #import pdb; pdb.set_trace()
                # Compute percentiles across all channels for this sample
                sample_data = data[b]

                # undo z-score norm
                sample_data = sample_data * std + mean
                #a_min = np.percentile(sample_data, self.lower)
                #a_max = np.percentile(sample_data, self.upper)
                #sample_data = np.clip(sample_data, self.a_min, self.a_max)
                shift = np.random.uniform(-50, 50)
                sample_data = np.clip(sample_data, self.a_min + shift, self.a_max + shift)

                data[b] = z_score_normalization(sample_data)
    
        data_dict[self.data_key] = data
        return data_dict

class ScaleIntensityPercentileTransform(AbstractTransform):
    def __init__(self, lower=0.5, upper=99.5, 
                 per_channel=True, data_key='data', p_per_sample=1):
        """
        Scale intensity based on percentiles.
        
        Args:
            lower: lower percentile (default: 0.5)
            upper: upper percentile (default: 99.5)
            b_min: minimum value after scaling (default: 0.0)
            b_max: maximum value after scaling (default: 1.0)
            clip: whether to clip values outside [b_min, b_max] after scaling (default: False)
            per_channel: whether to compute percentiles per channel (default: True)
            data_key: key to access data in the dictionary (default: 'data')
            p_per_sample: probability of applying transform to each sample (default: 1)
        """
        self.data_key = data_key
        self.lower = lower
        self.upper = upper
        #self.b_min = b_min
        #self.b_max = b_max
        #self.clip = clip
        self.per_channel = per_channel
        self.p_per_sample = p_per_sample
        self.property_key = 'properties'

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        properties = data_dict['properties']
        #means = properties['means']
        #stds = properties['stds']
        #import pdb; pdb.set_trace()
        
        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                mean = properties[b]['means'][0]
                std = properties[b]['stds']
                #import pdb; pdb.set_trace()
                # Compute percentiles across all channels for this sample
                sample_data = data[b]

                # undo z-score norm
                sample_data = sample_data * std + mean
                a_min = np.percentile(sample_data, self.lower)
                a_max = np.percentile(sample_data, self.upper)
                sample_data = np.clip(sample_data, a_min, a_max)

                data[b] = z_score_normalization(sample_data)
                # if a_max - a_min > 1e-8:
                #     # Scale the data
                #     data[b] = (sample_data - a_min) / (a_max - a_min) * (self.b_max - self.b_min) + self.b_min
                    
                #     # Clip outliers if requested
                #     if self.clip:
                #         data[b] = np.clip(data[b], self.b_min, self.b_max)
        
        data_dict[self.data_key] = data
        return data_dict

class ScaleIntensityRandWindowTransform(AbstractTransform):
    def __init__(self, data_key='data', p_per_sample=1):
        """
        Apply CT windowing (randomly select one window) then z-score normalize.
        
        Args:
            data_key: key to access data in the dictionary (default: 'data')
            p_per_sample: probability of applying transform to each sample (default: 1)
            p_per_window: probability of selecting each specific window (uniform random if not applied)
        """
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        #self.p_per_window = p_per_window
        self.windows = {
            "lung": {"center": -600, "width": 1500},
            "mediastinum": {"center": 50, "width": 400},
            "abdomen": {"center": 40, "width": 400},
            "liver": {"center": 80, "width": 150},
            "bone": {"center": 400, "width": 1800},
            #"brain": {"center": 40, "width": 80},
            #"subdural": {"center": 75, "width": 215},
            #"stroke": {"center": 40, "width": 40},
            #"temporal_bone": {"center": 600, "width": 2800},
            "soft_tissue": {"center": 50, "width": 350},
            }
        self.variation_factor = 0.1

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        properties = data_dict['properties']

        
        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                mean = properties[b]['means'][0]
                std = properties[b]['stds']
                # Randomly select a window
                window_name = np.random.choice(list(self.windows.keys()))
                window = self.windows[window_name]
                
                center = window["center"]
                width = window["width"]

                # --- ADD VARIATION HERE ---
                # Apply a jitter of +/- 10% to the width and a shift to the center
                #variation_factor = 0.1  # 10% variation
                width = width * np.random.uniform(1 - self.variation_factor, 1 + self.variation_factor)
                center = center + (width * np.random.uniform(-self.variation_factor, self.variation_factor))

                # Calculate window bounds
                w_min = center - width / 2
                w_max = center + width / 2
                
                # Undo z-score normalization
                sample_data = data[b] * std + mean
                
                # Apply windowing (clip to window range)
                sample_data = np.clip(sample_data, w_min, w_max)
                
                # Re-normalize with new statistics
                data[b] = z_score_normalization(sample_data)
        #         import SimpleITK as sitk
        #         dest = '/cluster/home/t129616uhn/CT_FM/scripts/train_transformed'
        #         sitk.WriteImage(sitk.GetImageFromArray(data[b][0]), os.path.join(dest, f'data_{str(b)}.nii.gz'))
        # exit()
        data_dict[self.data_key] = data
        return data_dict


class ScaleIntensityWindowTransform(AbstractTransform):
    def __init__(self, data_key='data', p_per_sample=1):
        """
        Apply CT windowing (randomly select one window) then z-score normalize.
        
        Args:
            data_key: key to access data in the dictionary (default: 'data')
            p_per_sample: probability of applying transform to each sample (default: 1)
            p_per_window: probability of selecting each specific window (uniform random if not applied)
        """
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        #self.p_per_window = p_per_window
        self.windows = {
            "lung": {"center": -600, "width": 1500},
            "mediastinum": {"center": 50, "width": 400},
            "abdomen": {"center": 40, "width": 400},
            "liver": {"center": 80, "width": 150},
            "bone": {"center": 400, "width": 1800},
            #"brain": {"center": 40, "width": 80},
            #"subdural": {"center": 75, "width": 215},
            #"stroke": {"center": 40, "width": 40},
            #"temporal_bone": {"center": 600, "width": 2800},
            "soft_tissue": {"center": 50, "width": 350},
            }
        self.variation_factor = 0.1

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        properties = data_dict['properties']

        
        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                mean = properties[b]['means'][0]
                std = properties[b]['stds']
                # Randomly select a window
                window_name = np.random.choice(list(self.windows.keys()))
                window = self.windows[window_name]
                
                center = window["center"]
                width = window["width"]

                # Calculate window bounds
                w_min = center - width / 2
                w_max = center + width / 2
                
                # Undo z-score normalization
                sample_data = data[b] * std + mean
                
                # Apply windowing (clip to window range)
                sample_data = np.clip(sample_data, w_min, w_max)
                
                # Re-normalize with new statistics
                data[b] = z_score_normalization(sample_data)
        #         import SimpleITK as sitk
        #         dest = '/cluster/home/t129616uhn/CT_FM/scripts/train_transformed'
        #         sitk.WriteImage(sitk.GetImageFromArray(data[b][0]), os.path.join(dest, f'data_{str(b)}.nii.gz'))
        # exit()
        data_dict[self.data_key] = data
        return data_dict
# class ScaleIntensityRangePercentilesTransform(AbstractTransform):
#     def __init__(self, lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, 
#                  clip=False, per_channel=True, data_key='data', p_per_sample=1):
#         """
#         Scale intensity based on percentiles.
        
#         Args:
#             lower: lower percentile (default: 0.5)
#             upper: upper percentile (default: 99.5)
#             b_min: minimum value after scaling (default: 0.0)
#             b_max: maximum value after scaling (default: 1.0)
#             clip: whether to clip values outside [b_min, b_max] after scaling (default: False)
#             per_channel: whether to compute percentiles per channel (default: True)
#             data_key: key to access data in the dictionary (default: 'data')
#             p_per_sample: probability of applying transform to each sample (default: 1)
#         """
#         self.data_key = data_key
#         self.lower = lower
#         self.upper = upper
#         self.b_min = b_min
#         self.b_max = b_max
#         self.clip = clip
#         self.per_channel = per_channel
#         self.p_per_sample = p_per_sample

#     def __call__(self, **data_dict):
#         data = data_dict[self.data_key]
        
#         for b in range(data.shape[0]):
#             if np.random.uniform() < self.p_per_sample:
#                 if self.per_channel:
#                     # Process each channel separately
#                     for c in range(data.shape[1]):
#                         channel_data = data[b, c]
#                         a_min = np.percentile(channel_data, self.lower)
#                         a_max = np.percentile(channel_data, self.upper)
                        
#                         if a_max - a_min > 1e-8:
#                             # Scale the data
#                             data[b, c] = (channel_data - a_min) / (a_max - a_min) * (self.b_max - self.b_min) + self.b_min
                            
#                             # Clip outliers if requested
#                             if self.clip:
#                                 data[b, c] = np.clip(data[b, c], self.b_min, self.b_max)
#                 else:
#                     # Compute percentiles across all channels for this sample
#                     sample_data = data[b]
#                     a_min = np.percentile(sample_data, self.lower)
#                     a_max = np.percentile(sample_data, self.upper)
                    
#                     if a_max - a_min > 1e-8:
#                         # Scale the data
#                         data[b] = (sample_data - a_min) / (a_max - a_min) * (self.b_max - self.b_min) + self.b_min
                        
#                         # Clip outliers if requested
#                         if self.clip:
#                             data[b] = np.clip(data[b], self.b_min, self.b_max)
        
#         data_dict[self.data_key] = data
#         return data_dict

if __name__ == "__main__":

    import blosc2
    from batchgenerators.utilities.file_and_folder_operations import write_pickle, load_pickle
    import SimpleITK as sitk
    from batchgenerators.transforms.noise_transforms import (
        GaussianNoiseTransform,
        GaussianBlurTransform,
        RicianNoiseTransform,
    )
    from batchgenerators.transforms.color_transforms import (
        BrightnessMultiplicativeTransform,
        ContrastAugmentationTransform,
        GammaTransform,
    )
    import time
    dparams = {"nthreads": 1}
    path = '/cluster/projects/bwanggroup/sumin/nnssl_data/nnssl_preprocessed/Dataset001_MerlinTrain/nnsslPlans_onemmiso/Dataset001_MerlinTrain/Dataset001_MerlinTrain/AC4213649/ses-DEFAULT/AC4213649.b2nd'
    path_pkl = '/cluster/projects/bwanggroup/sumin/nnssl_data/nnssl_preprocessed/Dataset001_MerlinTrain/nnsslPlans_onemmiso/Dataset001_MerlinTrain/Dataset001_MerlinTrain/AC4213649/ses-DEFAULT/AC4213649.pkl'
    dest = '/cluster/home/t129616uhn/CT_FM/scripts/transformed'
    os.makedirs(dest, exist_ok=True)

    b2r = blosc2.open(urlpath=path, mode="r", dparams=dparams, mmap_mode=None)
    sitk.WriteImage(sitk.GetImageFromArray(b2r[0]), os.path.join(dest, 'ori.nii.gz'))
    #exit()
    test_volume = b2r[None]
    pkl_data = load_pickle(path_pkl)
    
    # import pdb; pdb.set_trace()

    inp_dict = {"data": test_volume, 'properties': [pkl_data]}
    #trafo = ScaleIntensityPercentileTransform(p_per_sample=1) # 0.90
    #trafo = ScaleIntensityWindowTransform(p_per_sample=1) #0.30
    #trafo = ClipIntensityPercentileTransform(p_per_sample=1) #0.74
    #trafo = ClipIntensityRandPercentileTransform(p_per_sample=1, lower_range=(6, 10), upper_range=(100, 100)) #0.70
    trafo = ScaleIntensityRand1000Transform(p_per_sample=1, scale_range=(0.98, 1.1)) #0.67
    trafo = ScaleIntensity1000Transform(p_per_sample=1)
    trafo = ScaleIntensityRandLowerUpperTransform(p_per_sample=1, lower_scale_range=(1.0, 1.1), upper_scale_range=(2.9, 3.0))

    start = time.time()
    res = trafo(**inp_dict)
    end = time.time()
    print(f'time ', end - start)
    res_data = res['data'][0,0]
    save_path = os.path.join(dest, f'transformed_1000lu3000.nii.gz')
    sitk.WriteImage(sitk.GetImageFromArray(res_data), save_path)
    print(save_path)



    # import blosc2
    # from batchgenerators.utilities.file_and_folder_operations import write_pickle, load_pickle
    # import SimpleITK as sitk
    # from batchgenerators.transforms.noise_transforms import (
    #     GaussianNoiseTransform,
    #     GaussianBlurTransform,
    #     RicianNoiseTransform,
    # )
    # from batchgenerators.transforms.color_transforms import (
    #     BrightnessMultiplicativeTransform,
    #     ContrastAugmentationTransform,
    #     GammaTransform,
    # )
    # import time
    # dparams = {"nthreads": 1}
    # path = '/cluster/projects/bwanggroup/sumin/nnssl_data/nnssl_preprocessed/Dataset001_MerlinTrain/nnsslPlans_onemmiso/Dataset001_MerlinTrain/Dataset001_MerlinTrain/AC4213649/ses-DEFAULT/AC4213649.b2nd'
    # path_pkl = '/cluster/projects/bwanggroup/sumin/nnssl_data/nnssl_preprocessed/Dataset001_MerlinTrain/nnsslPlans_onemmiso/Dataset001_MerlinTrain/Dataset001_MerlinTrain/AC4213649/ses-DEFAULT/AC4213649.pkl'
    # dest = '/cluster/home/t129616uhn/CT_FM/scripts/transformed'

    # b2r = blosc2.open(urlpath=path, mode="r", dparams=dparams, mmap_mode=None)
    # #sitk.WriteImage(sitk.GetImageFromArray(b2r[0]), os.path.join(dest, 'ori.nii.gz'))
    # #exit()
    # test_volume = b2r[None]
    # pkl_data = load_pickle(path_pkl)
    
    # # import pdb; pdb.set_trace()

    # inp_dict = {"data": test_volume, 'properties': [pkl_data]}
    # #trafo = ScaleIntensityPercentileTransform(p_per_sample=1) # 0.90
    # trafo = ScaleIntensityWindowTransform(p_per_sample=1) #0.30
    # keep_organ = 'soft_tissue'
    # windows = {
    #         "lung": {"center": -600, "width": 1500},
    #         "mediastinum": {"center": 50, "width": 400},
    #         "abdomen": {"center": 40, "width": 400},
    #         "liver": {"center": 80, "width": 150},
    #         "bone": {"center": 400, "width": 1800},
    #         #"brain": {"center": 40, "width": 80},
    #         #"subdural": {"center": 75, "width": 215},
    #         #"stroke": {"center": 40, "width": 40},
    #         #"temporal_bone": {"center": 600, "width": 2800},
    #         "soft_tissue": {"center": 50, "width": 350},
    #         }
    # windows = {k:v for k,v in windows.items() if k == keep_organ}
    # print(windows)
    # trafo.windows = windows
    # #trafo = ScaleIntensity1000Transform(p_per_sample=1)
    # #trafo = GaussianNoiseTransform(p_per_sample=1) #1.11
    #     #train="train",
    # #     data_key="data",
    # #     initial_patch_size=(256, 256, 256),
    # #     patch_size=(160, 160, 160),
    # #     rotation_for_DA={"x": (0, 30), "y": (0, 30), "z": (0, 30)},
    # #     mirror_axes=(0, 1, 2),
    # #     do_dummy_2d_data_aug=False,
    # #     order_resampling_data=3,
    # #     order_resampling_seg=1,
    # #     border_val_seg=-1,
    # #     use_mask_for_norm=False,
    # # )
    # #trafo = ScaleIntensityWindowTransform(p_per_sample=0.1)
    # #trafo = GaussianNoiseTransform(p_per_sample=1) # 1.12
    # # trafo = GaussianBlurTransform(
    # #     (0.5, 1.0),
    # #     different_sigma_per_channel=True,
    # #     p_per_sample=1,
    # #     p_per_channel=1,
    # # ) #0.88
    # # trafo = BrightnessMultiplicativeTransform(
    # #     multiplier_range=(0.75, 1.25), p_per_sample=1
    # # ) #0.01
    # #trafo = ContrastAugmentationTransform(p_per_sample=1) #0.21
    # # # GammaTransform(
    # # #     (0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1
    # # # ),
    # # trafo = GammaTransform(
    # #     gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True, p_per_sample=1
    # # ) #0.81
    # start = time.time()
    # res = trafo(**inp_dict)
    # end = time.time()
    # print(f'time ', end - start)
    # res_data = res['data'][0,0]
    # save_path = os.path.join(dest, f'transformed_{keep_organ}.nii.gz')
    # sitk.WriteImage(sitk.GetImageFromArray(res_data), save_path)
    # print(save_path)

