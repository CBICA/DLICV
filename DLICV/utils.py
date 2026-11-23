import os
import random
import shutil
from typing import Dict, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch

NIFTI_EXTENSIONS = (".nii.gz", ".nii")


def prepare_data_folder(folder_path: str) -> None:
    """
    prepare data folder, create one if not exist
    """
    os.makedirs(folder_path, exist_ok=True)


def is_nifti(filename: str) -> bool:
    return filename.lower().endswith(NIFTI_EXTENSIONS)


def nifti_stem(filename: str) -> str:
    """Return the filename without its .nii / .nii.gz extension."""
    lower = filename.lower()
    for ext in NIFTI_EXTENSIONS:
        if lower.endswith(ext):
            return filename[: -len(ext)]
    return filename


def output_name_for(filename: str) -> str:
    """Final output name of the segmentation for a given input filename.

    nnU-Net always writes .nii.gz, so uncompressed .nii inputs get a .nii.gz output.
    """
    if filename.lower().endswith(".nii.gz"):
        return "label_" + filename
    return "label_" + nifti_stem(filename) + ".nii.gz"


def _stage_file(src: str, dst: str) -> None:
    """Make `src` available at `dst` without duplicating data when possible."""
    if src.lower().endswith(".nii.gz"):
        try:
            os.symlink(os.path.abspath(src), dst)
            return
        except OSError:
            # symlinks not supported (e.g. Windows without privileges, some FUSE mounts)
            shutil.copy2(src, dst)
            return
    # Uncompressed .nii: nnU-Net only picks up .nii.gz, so convert.
    sitk.WriteImage(sitk.ReadImage(src), dst)


def rename_and_copy_files(
    src_folder: str, des_folder: str, skip: Optional[set] = None
) -> Tuple[dict, dict]:
    """
    Input:
         src_folder: a user input folder (or a single NIfTI file), names could be anything,
         will be converted into nnUNet format internally

         des_folder: where you want to store the staged files

         skip: optional set of original filenames that should not be staged

    Returns:
         rename_dict : a dictionary mapping your original name into nnUnet format name
         rename_back_dict:  a dictionary used to map back to the original (output) name

    Files are symlinked into des_folder (falling back to a copy) instead of being copied,
    and only NIfTI files are considered. Files are sorted so that the case mapping is stable
    across runs.
    """
    if os.path.isfile(src_folder):
        src_dir, files = os.path.dirname(src_folder), [os.path.basename(src_folder)]
    else:
        src_dir, files = src_folder, sorted(os.listdir(src_folder))

    skip = skip or set()
    rename_dict = {}
    rename_back_dict = {}

    idx = 0
    for filename in files:
        old_name = os.path.join(src_dir, filename)
        if not os.path.isfile(old_name) or not is_nifti(filename):
            continue
        if filename in skip:
            continue
        rename_file = f"case_{idx:04d}_0000.nii.gz"
        rename_back = f"case_{idx:04d}.nii.gz"
        new_name = os.path.join(des_folder, rename_file)
        if os.path.lexists(new_name):
            os.remove(new_name)
        _stage_file(old_name, new_name)
        rename_dict[filename] = rename_file
        rename_back_dict[rename_back] = output_name_for(filename)
        idx += 1

    return rename_dict, rename_back_dict


def validate_inputs(des_folder: str, rename_dict: dict) -> Dict[str, str]:
    """
    Check the header of every staged image. Unreadable images are removed from the
    staging folder so a single corrupt file cannot crash the whole nnU-Net run.

    Returns a dictionary {original_name: error_message} of invalid inputs.
    """
    invalid = {}
    reader = sitk.ImageFileReader()
    for original_name, staged_name in rename_dict.items():
        staged_path = os.path.join(des_folder, staged_name)
        try:
            reader.SetFileName(staged_path)
            reader.ReadImageInformation()
            if reader.GetDimension() != 3:
                raise ValueError(f"expected a 3D image, got {reader.GetDimension()}D")
        except Exception as e:  # noqa: BLE001 - any read error means invalid input
            msg = str(e).strip().splitlines()
            invalid[original_name] = msg[-1] if msg else type(e).__name__
            if os.path.lexists(staged_path):
                os.remove(staged_path)
    return invalid


def get_num_cpus() -> int:
    """Number of CPUs usable by this process (respects affinity / cgroup cpusets)."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    return max(1, n)


def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def analyze_connected_components_for_icv(binary_mask):
    """Analyze all components and select brain based on multiple criteria"""
    cc_image = sitk.ConnectedComponent(binary_mask)

    # Get label statistics
    stats_filter = sitk.LabelShapeStatisticsImageFilter()
    stats_filter.Execute(cc_image)

    if stats_filter.GetNumberOfLabels() == 0:
        qc_log = "Empty segmentation"
        print(qc_log)
        return None, None, qc_log

    components_info = []
    sizes = []

    for label in stats_filter.GetLabels():
        size = stats_filter.GetNumberOfPixels(label)
        # Thresholding to exclude extremely small masks
        if size > 200000:
            components_info.append({
                'label': label,
                'size': size,
                'roundness': stats_filter.GetRoundness(label),
            })
            sizes.append(size)

    if len(components_info) == 0:
        qc_log = "No connected component detected"
        print(qc_log)
        return None, None, qc_log

    std_size = np.std(sizes)
    avg_size = np.mean(sizes)

    print("Component Analysis:")
    for comp in components_info:
        print(f"Component {comp['label']}: Size={comp['size']}, "
              f"std Size={std_size}, "
              f"Roundness={comp['roundness']:.3f}")

    if len(components_info) == 1:
        qc_log = "Success. Single connected component."
        label = components_info[0]['label']
        return sitk.Equal(cc_image, label), label, qc_log

    # Sort by size
    components_info.sort(key=lambda x: x['size'], reverse=True)

    print(f"Avg Size: {avg_size}, Std Size: {std_size}")

    # select only ones within 99% conf intv
    components_info = [
        comp for comp in components_info
        if (avg_size - 3 * std_size) < comp['size'] < (avg_size + 3 * std_size)
    ]

    if len(components_info) > 1:
        # Select top 2 by size
        components_info = components_info[:2]
        # Select one with the higher roundness
        components_info.sort(key=lambda x: x['roundness'], reverse=True)

    if not components_info:
        qc_log = "CC analysis failed. Keeping the original mask"
        print(qc_log)
        return None, None, qc_log

    true_icv_mask = components_info[0]['label']
    qc_log = "Success. Correct CC detected and overwritten."
    return sitk.Equal(cc_image, true_icv_mask), true_icv_mask, qc_log
