import argparse
import os
import sys
from pathlib import Path
from DLICV.compute_icv import compute_volume

def main():
    # Read version from the VERSION file
    version = Path(__file__).with_name("VERSION").read_text().strip()

    parser = argparse.ArgumentParser(description="DLICV - Deep Learning Intra Cranial Volume estimation from structural MRI scans.")
    
    parser.add_argument(
        '-i', '--input', 
        type=str, 
        required=True, 
        help='Path to the input nifti image or a directory of nifti images.'
    )
    
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        required=True, 
        help='Path where the output nifti image or directory to save output images will be saved. Expecting an nnUNet model.'
    )
    
    parser.add_argument(
        '-m', '--model', 
        type=str, 
        required=False,
        default="default",
        help='Path where the model to be used is stored.'
    )

    parser.add_argument(
        '-v', '--version', 
        action='version', 
        version=f'%(prog)s {version}',
        help="Show program's version number and exit."
    )

    args = parser.parse_args()

    # Check if input is a directory or file
    input_path = Path(args.input)
    if input_path.is_dir():
        # Process each nifti image in the directory
        for file_path in input_path.glob('*.nii.gz'):
            process_image(file_path, args.output, args.model)
    elif input_path.is_file():
        # Process the single file
        process_image(input_path, args.output, args.model)
    else:
        print(f"The input path {args.input} is not valid.")
        sys.exit(1)

def process_image(input_file, output_base, model):
    # Calculate intracranial volume
    output_volume = compute_volume(input_file, model)
    
    # Determine the output file path
    if Path(output_base).is_dir():
        output_path = Path(output_base) / input_file.name
    else:
        output_path = Path(output_base)
    
    # Create the directory if it does not exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the output volume
    output_volume.to_filename(str(output_path))  # Assuming 'output_volume' is a nibabel Nifti1Image
    
    print(f"Intracranial volume computed for {input_file} and saved to {output_path}")

if __name__ == "__main__":
    main()
