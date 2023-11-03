# DLICV - Deep Learning Intra Cranial Volume

## Overview

DLICV uses a trained [nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) model to compute the intracranial volume from structural MRI scans in the nifti image format, oriented in LPS.

## Installation

### As a python package

```bash
pip install dlicv
```

### Directly from this repository

```bash
git clone https://github.com/georgeaidinis/DLICV
```

## Usage

### Import as a python package

```python
from dlicv.compute_icv import compute_volume

# Assuming your nifti file is named 'input.nii.gz'
volume_image = compute_volume("input.nii.gz")
```

### From the terminal

```bash
dlicv compute --input input.nii.gz --output output.nii.gz
```

Replace the `input.nii.gz` with the path to your input infit file.

## Contact

For more information, please contact [CBICA Software](mailto:software@cbica.upenn.edu).
