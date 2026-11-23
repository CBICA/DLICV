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

`-i` can be a folder of `.nii.gz`/`.nii` images or a single image. Other files in the folder are ignored, and unreadable images are skipped without stopping the run. If the requested device (`cuda`/`mps`) is not available, DLICV falls back to CPU.

#### QC report
Each run writes a QC report to `output_folder/qc_report.csv`. Use `--qc_out_path` to save it somewhere else. It has one row per input image:

| column | description |
|---|---|
| `file_path` | path of the output mask (`label_<input name>`) |
| `file_name` | input image name |
| `QC_flag` | `Passed`, `Failed`, or `Not checked` (when `--post_processing false`) |
| `QC_log` | details, e.g. connected-component result, `Invalid input: ...`, `No segmentation produced` |

Masks that fail the connected-component QC keep the raw prediction. Use `--drop_failed` to delete them instead.

#### Troubleshooting model download failures
Our model download process creates several deep directory structures. If you are on Windows and your model download process fails, it may be due to Windows file path limitations. 

To enable long path support in Windows 10, version 1607, and later, the registry key `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem LongPathsEnabled (Type: REG_DWORD)` must exist and be set to 1.

If this affects you, we recommend re-running DLICV with the `--clear_cache` flag set on the first run.


## Contact
For more information, please contact [CBICA Software](mailto:software@cbica.upenn.edu).

## For developers
Contributions are welcome! Please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) for more information on how to report bugs, suggest enhancements, and contribute code.
Please make sure to write tests for new code and run them before submitting a pull request.
