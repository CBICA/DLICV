### DLICV - Deep Learning Intra Cranial Volume

## Overview

DLICV uses a trained [nnUNet](https://github.com/MIC-DKFZ/nnUNet) model to compute the intracranial volume from structural MRI scans in the nifti image format, oriented in _**LPS**_ orientation.

## Installation

To install DLICV, just do
```bash
pip install DLICV
```

Or to manually download DLICV
```bash
pip install -e .
```

You can install the required packages with
```bash
pip install -r requirements.txt
```

## How to run
```bash
DLICV -i "input_folder" -o "output_folder" -m "model_path" -f 0 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans -d "id" -device cuda/cpu/mps
```

You can perform one example test run using the test input folder by running
```bash
DLICV -i test_input/DLICV_test_images -o test_input/DLICV_test_results -m nnunet_results -f 0 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans -d 901 -device cuda
```

## Optional (HCP SLURM SUBMISSION):
  - switch to your envioronment: remember your enviorment should contains nnunetv2 and pytorch, source / conda activate YOUR_ENV_NAME
  - Inside sample_submit_inference.sh
      - set input_folder_path
      - set output_folder_path
  - sbatch sample_submit_inference.sh
