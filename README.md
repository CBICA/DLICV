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
cd DLICV
conda create -n DLICV -y python=3.8 && conda activate DLICV
pip install .
```

## Usage

### Import as a python package

```python
from dlicv.compute_icv import compute_volume

# Assuming your nifti file is named 'input.nii.gz'
volume_image = compute_volume("input.nii.gz", "output.nii.gz", "path/to/model/")
```

### From the terminal

```bash
DLICV --input input.nii.gz --output output.nii.gz --model path/to/model
```

Replace the `input.nii.gz` with the path to your input nifti file, as well as the model path.

Example:

Assuming a file structure like so:

```bash
.
├── in
│   ├── input1.nii.gz
│   ├── input2.nii.gz
│   └── input3.nii.gz
├── model
│   ├── fold_0
│   ├── fold_1
│   │   ├── debug.json
│   │   ├── model_final_checkpoint.model
│   │   ├── model_final_checkpoint.model.pkl
│   │   ├── model_latest.model
│   │   ├── model_latest.model.pkl
│   └── plans.pkl
└── out
```

An example command might be:

```bash
DLICV --input path/to/input/ --output path/to/output/ --model path/to/model/
```

## Contact

For more information, please contact [CBICA Software](mailto:software@cbica.upenn.edu).
