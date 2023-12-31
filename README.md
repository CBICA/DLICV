# DLICV - Deep Learning Intra Cranial Volume

## Overview

DLICV uses a trained [nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) model to compute the intracranial volume from structural MRI scans in the nifti image format, oriented in _**LPS**_ orientation.

## Installation

### As a python package

```bash
pip install DLICV
```

### Directly from this repository

```bash
git clone https://github.com/CBICA/DLICV
cd DLICV
conda create -n DLICV -y python=3.8 && conda activate DLICV
pip install .
```

### Using docker

```bash
docker pull aidinisg/dlicv:0.0.0
```

## Usage

A pre-trained nnUNet model can be found in the [DLICV-0.0.0 release](https://github.com/CBICA/DLICV/releases/tag/v0.0.0) as an [artifact](https://github.com/CBICA/DLICV/releases/download/v0.0.0/model.zip). Feel free to use it under the package's [license](LICENSE).

### Import as a python package

```python
from DLICV.compute_icv import compute_volume

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

### Using the docker container

In the [docker container](https://hub.docker.com/repository/docker/aidinisg/dlicv/general), the default model is included, but you can also provide your own.

Without providing a model:

```bash
docker run --gpus all -it --rm -v /path/to/local/input:/workspace/input \
                               -v /path/to/local/output:/workspace/output \
                               aidinisg/dlicv:0.0.0  -i input/ -o output/
```

Providing a model:

```bash
docker run --gpus all -it --rm -v /path/to/local/model:/workspace/model \
                               -v /path/to/local/input:/workspace/input \
                               -v /path/to/local/output:/workspace/output \
                               aidinisg/dlicv:0.0.0  -i input/ -o output/  --model model/
```

## Contact

For more information, please contact [CBICA Software](mailto:software@cbica.upenn.edu).

## For Developers

Contributions are welcome! Please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) for more information on how to report bugs, suggest enhancements, and contribute code.

If you're a developer looking to contribute, you'll first need to set up a development environment. After cloning the repository, you can install the development dependencies with:

```bash
pip install -r requirements-test.txt
```

This will install the packages required for running tests and formatting code. Please make sure to write tests for new code and run them before submitting a pull request.

### Running Tests

You can run the test suite with the following command:

```bash
pytest
```
