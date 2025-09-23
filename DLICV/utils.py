import os
import shutil
from typing import Tuple
import random

import numpy as np
import torch


def prepare_data_folder(folder_path: str) -> None:
    """
    prepare data folder, create one if not exist
    if exist, empty the folder
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def rename_and_copy_files(src_folder: str, des_folder: str) -> Tuple[dict, dict]:
    """
    Input:
         src_folder: a user input folder, name could be anything, will be convert into nnUnet
         format internally

         des_folder: where you want to store your folder

    Returns:
         rename_dict : a dictionary mapping your original name into nnUnet format name
         rename_back_dict:  a dictionary will be use to mapping backto the original name

    """
    files = os.listdir(src_folder)
    rename_dict = {}
    rename_back_dict = {}

    for idx, filename in enumerate(files):
        old_name = os.path.join(src_folder, filename)
        rename_file = f"case_{idx:04d}_0000.nii.gz"
        rename_back = f"case_{idx:04d}.nii.gz"
        new_name = os.path.join(des_folder, rename_file)
        shutil.copy2(old_name, new_name)
        rename_dict[filename] = rename_file
        rename_back_dict[rename_back] = "label_" + filename

    return rename_dict, rename_back_dict

def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import numpy as np
import SimpleITK as sitk

def analyze_connected_components_for_icv(binary_mask):
    """Analyze all components and select brain based on multiple criteria"""
    cc_image = sitk.ConnectedComponent(binary_mask)
    
    # Get label statistics
    stats_filter = sitk.LabelShapeStatisticsImageFilter()
    stats_filter.Execute(cc_image)
    
    components_info = []
    sizes = []
    
    for label in stats_filter.GetLabels():
        size = stats_filter.GetNumberOfPixels(label)
        roundness = stats_filter.GetRoundness(label)
        # Thresholding to exclude extremely small masks
        if size > 200000:
            components_info.append({
                'label': label,
                'size': size,
                'roundness': roundness
            })
            sizes.append(size)
    
    print("Component Analysis:")
    for i, comp in enumerate(components_info):
        print(f"Component {comp['label']}: Size={comp['size']}, "
              f"std Size={np.std(sizes)}, "
              f"Roundness={comp['roundness']:.3f}")

    if len(components_info) == 0:
        raise("Failed to identify the true ICV mask based on connected component analysis.")
    elif len(components_info) == 1:
        return sitk.Equal(cc_image,components_info[0]['label']), components_info[0]['label']

    # Sort by size
    components_info.sort(key=lambda x: x['size'], reverse=True)

    # Get std of valid sizes
    std_size = np.std(sizes)
    avg_size = np.mean(sizes)

    print(f"Avg Size: {avg_size}, Std Size: {std_size}")

    # select only ones within 99% conf intv
    updated_components_info = []
    for i, comp in enumerate(components_info):
        if comp['size'] > (avg_size - 3 * std_size) and comp['size'] < (avg_size + 3 * std_size):
            updated_components_info.append(comp)

    components_info = updated_components_info

    if len(components_info) > 1:
        # Select top 2 by size
        components_info = components_info[:2]
        # Select one with the higher roundness
        components_info.sort(key=lambda x: x['roundness'], reverse=True)
    
    try:
        true_icv_mask = components_info[0]['label']
        return sitk.Equal(cc_image,true_icv_mask), true_icv_mask
    except:
        raise("Failed to identify the true ICV mask based on connected component analysis.")

    