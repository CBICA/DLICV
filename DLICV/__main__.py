import sys
import argparse
import shutil
from pathlib import Path
import tempfile
from DLICV.compute_icv import compute_volume

def validate_path(parser, arg):
    """Ensure the provided path exists."""
    if not Path(arg).exists():
        parser.error(f"The path {arg} does not exist.")
        sys.exit(1)
    return arg

def copy_and_rename_inputs(input_path, destination_dir):
    """Copy and rename input files according to nnUNet convention."""
    input_path = Path(input_path)
    destination_dir = Path(destination_dir)
    
    if input_path.is_dir():
        for filepath in input_path.glob('*.nii.gz'):
            new_filename = filepath.stem.split('.')[0] + '_0000.nii.gz'
            shutil.copy(filepath, destination_dir / new_filename)
    else:
        new_filename = input_path.stem + '_0000.nii.gz'
        shutil.copy(input_path, destination_dir / new_filename)

def main():
    """Main function to manage the prediction workflow."""
    parser = argparse.ArgumentParser(description="DLICV - Deep Learning Intra Cranial Volume")
    parser.add_argument("-i","--input", required=True, type=lambda x: validate_path(parser, x), help="Input .nii.gz image or folder path.")
    parser.add_argument("-o","--output", required=True, help="Output image or folder path. If folder path, it should exist.")
    parser.add_argument("-m","--model", required=True, type=lambda x: validate_path(parser, x), help="Model path.")
    parser.add_argument("-v","--version", action="version", version=Path(__file__).with_name("VERSION").read_text().strip(), help="Show program's version number and exit.")
    parser.add_argument("--kwargs", nargs=argparse.REMAINDER, help="Additional keyword arguments to pass to compute_icv.py")

    args = parser.parse_args()

    kwargs = {}
    if args.kwargs:
        for kwarg in args.kwargs:
            key, value = kwarg.split('=', 1)
            kwargs[key] = value

    input_path, output_path, model_path = args.input, args.output, args.model

    # Create cross-platform temp dir, and child dirs that nnUNet needs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_input_dir = Path(temp_dir) / "nnUNet_raw_data"
        temp_preprocessed_dir = Path(temp_dir) / "nnUNet_preprocessed"
        temp_output_dir = Path(temp_dir) / "nnUNet_out"
        temp_input_dir.mkdir()
        temp_output_dir.mkdir()

        copy_and_rename_inputs(input_path, temp_input_dir)
        compute_volume(
            str(temp_input_dir),
            str(temp_output_dir),
            model_path,
            **kwargs
        )

       # Move results to the specified output location, including only .nii.gz files
        for file in temp_output_dir.iterdir():
            if file.suffixes == ['.nii', '.gz']:
                shutil.move(str(file), output_path)
        print()
        print()
        print()
        print(f"Prediction complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
