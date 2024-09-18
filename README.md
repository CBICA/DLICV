### DLICV - Deep Learning Intra Cranial Volume

## Overview

DLICV uses a trained [nnUNet](https://github.com/MIC-DKFZ/nnUNet) model to compute the intracranial volume from structural MRI scans in the nifti image format, oriented in _**LPS**_ orientation.

## Installation

# As a python package
```bash
pip install DLICV
```
# Directly from this repository
```bash
git clone https://github.com/CBICA/DLICV
cd DLICV
pip install -e .
```

## Usage
A pre-trained nnUNet model can be found at our [hugging face account](https://huggingface.co/nichart/DLICV) or at [DLICV-V2 v1.0.0 release](https://github.com/CBICA/DLMUSE/releases/tag/v1.0.0).
Feel free to use it under the package's [licence](LICENCE)
```bash
DLICV -i "input_folder" -o "output_folder" -d "id" -device cuda/cpu/mps
```

You can perform one example test run using the test input folder by running
```bash
DLICV -i test_input/DLICV_test_images -o test_input/DLICV_test_results -d 901 -device cuda
```

## Contact
For more information, please contact [CBICA Software](mailto:software@cbica.upenn.edu).

## For developers
Contributions are welcome! Please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) for more information on how to report bugs, suggest enhancements, and contribute code.
If you're a developer looking to contribute, you'll first need to set up a development environment. After cloning the repository, you can install the development dependencies with:

```bash
pip install -r requirements.txt
```
This will install the packages required for running tests and formatting code. Please make sure to write tests for new code and run them before submitting a pull request.
