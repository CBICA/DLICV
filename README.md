# DLICV - Deep Learning Intra Cranial Volume

## Overview

DLICV uses a trained [nnUNet](https://github.com/MIC-DKFZ/nnUNet) model to compute the intracranial volume from structural MRI scans in the nifti image format, oriented in _**LPS**_ orientation.

## Installation

### As a python package
```bash
pip install DLICV
```
### Directly from this repository
```bash
git clone https://github.com/CBICA/DLICV
cd DLICV
pip install -e .
```

### Installing PyTorch
Depending on your system configuration and supported CUDA version, you may need to follow the [PyTorch Installation Instructions](https://pytorch.org/get-started/locally/). 

## Usage
A pre-trained nnUNet model can be found at our [hugging face account](https://huggingface.co/nichart/DLICV).
Feel free to use it under the package's [licence](LICENCE)
```bash
DLICV -i "input_folder" -o "output_folder" -device cuda
```

#### Troubleshooting model download failures
Our model download process creates several deep directory structures. If you are on Windows and your model download process fails, it may be due to Windows file path limitations. 

To enable long path support in Windows 10, version 1607, and later, the registry key `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem LongPathsEnabled (Type: REG_DWORD)` must exist and be set to 1.

If this affects you, we recommend re-running DLICV with the `--clear_cache` flag set on the first run.


## Contact
For more information, please contact [CBICA Software](mailto:software@cbica.upenn.edu).

## For developers
Contributions are welcome! Please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) for more information on how to report bugs, suggest enhancements, and contribute code.
Please make sure to write tests for new code and run them before submitting a pull request.
