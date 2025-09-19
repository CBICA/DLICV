import argparse
import json
import os
import shutil
import sys
import warnings
from pathlib import Path

import torch

from .utils import prepare_data_folder, rename_and_copy_files, set_random_seed, analyze_connected_components_for_icv

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# VERSION = pkg_resources.require("NiChart_DLMUSE")[0].version
VERSION = 1.0


def main() -> None:
    prog = "DLICV"
    parser = argparse.ArgumentParser(
        prog=prog,
        description="DLICV - Deep Learning Intra Cranial Volume.",
        usage="""
        DLICV v{VERSION}
        ICV calculation for structural MRI data.

        Required arguments:
            [-i, --in_dir]   The filepath of the input directory
            [-o, --out_dir]  The filepath of the output directory
        Optional arguments:
            [-device]        cpu|cuda|mps - Depending on your system configuration (default: cuda)
            [-h, --help]    Show this help message and exit.
            [-V, --version] Show program's version number and exit.
        EXAMPLE USAGE:
            DLICV  -i           /path/to/input     \
                   -o           /path/to/output    \
                   -device      cpu|cuda|mps

        """.format(
            VERSION=VERSION
        ),
    )

    # Required Arguments
    parser.add_argument(
        "-i",
        "--in_dir",
        type=str,
        required=True,
        help="[REQUIRED] Input folder with T1 sMRI images (nii.gz).",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        required=True,
        help="[REQUIRED] Output folder. If it does not exist it will be created. Predicted segmentations will have the same name as their source images.",
    )

    # Optional Arguments
    parser.add_argument(
        "-device",
        type=str,
        default="cuda",
        required=False,
        help="[Recommended] Use this to set the device the inference should run with. Available options are 'cuda' (GPU), "
        "'cpu' (CPU) or 'mps' (Apple M-series chips supporting 3D CNN).",
    )
    parser.add_argument(
        "--post_processing",
        type=str,
        required=False,
        default="true",
        help="Select the largest connected component. (Default: True)",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=prog + ": v{VERSION}.".format(VERSION=VERSION),
        help="Show the version and exit",
    )
    parser.add_argument(
        "-d",
        type=str,
        required=False,
        default="901",
        help="Dataset with which you would like to predict. You can specify either dataset name or id",
    )
    parser.add_argument(
        "-p",
        type=str,
        required=False,
        default="nnUNetPlans",
        help="Plans identifier. Specify the plans in which the desired configuration is located. "
        "Default: nnUNetPlans",
    )
    parser.add_argument(
        "-tr",
        type=str,
        required=False,
        default="nnUNetTrainer",
        help="What nnU-Net trainer class was used for training? Default: nnUNetTrainer",
    )
    parser.add_argument(
        "-c",
        type=str,
        required=False,
        default="3d_fullres",
        help="nnU-Net configuration that should be used for prediction. Config must be located "
        "in the plans specified with -p",
    )
    parser.add_argument(
        "-step_size",
        type=float,
        required=False,
        default=0.5,
        help="Step size for sliding window prediction. The larger it is the faster but less accurate "
        "the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.",
    )
    parser.add_argument(
        "--disable_tta",
        action="store_true",
        required=False,
        default=False,
        help="Set this flag to disable test time data augmentation in the form of mirroring. Faster, "
        "but less accurate inference. Not recommended.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set this if you like being talked to. You will have "
        "to be a good listener/reader.",
    )
    parser.add_argument(
        "--save_probabilities",
        action="store_true",
        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
        "multiple configurations.",
    )
    parser.add_argument(
        "--continue_prediction",
        action="store_true",
        help="Continue an aborted previous prediction (will not overwrite existing files)",
    )
    parser.add_argument(
        "-chk",
        type=str,
        required=False,
        default="checkpoint_final.pth",
        help="Name of the checkpoint you want to use. Default: checkpoint_final.pth",
    )
    parser.add_argument(
        "-npp",
        type=int,
        required=False,
        default=2,
        help="Number of processes used for preprocessing. More is not always better. Beware of "
        "out-of-RAM issues. Default: 2",
    )
    parser.add_argument(
        "-nps",
        type=int,
        required=False,
        default=2,
        help="Number of processes used for segmentation export. More is not always better. Beware of "
        "out-of-RAM issues. Default: 2",
    )
    parser.add_argument(
        "-prev_stage_predictions",
        type=str,
        required=False,
        default=None,
        help="Folder containing the predictions of the previous stage. Required for cascaded models.",
    )
    parser.add_argument(
        "-num_parts",
        type=int,
        required=False,
        default=1,
        help="Number of separate nnUNetv2_predict call that you will be making. Default: 1 (= this one "
        "call predicts everything)",
    )
    parser.add_argument(
        "-part_id",
        type=int,
        required=False,
        default=0,
        help="If multiple nnUNetv2_predict exist, which one is this? IDs start with 0 can end with "
        "num_parts - 1. So when you submit 5 nnUNetv2_predict calls you need to set -num_parts "
        "5 and use -part_id 0, 1, 2, 3 and 4. Simple, right? Note: You are yourself responsible "
        "to make these run on separate GPUs! Use CUDA_VISIBLE_DEVICES (google, yo!)",
    )
    parser.add_argument(
        "--disable_progress_bar",
        action="store_true",
        required=False,
        default=False,
        help="Set this flag to disable progress bar. Recommended for HPC environments (non interactive "
        "jobs)",
    )
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        required=False,
        default=False,
        help="Set this flag to clear any cached models before running. This is recommended if a previous download failed.",
    )
    # Set random seed to a fixed value
    set_random_seed(42)

    # input args
    args = parser.parse_args()
    args.f = [0]
    args.i = args.in_dir
    args.o = args.out_dir

    if args.post_processing.upper() == 'TRUE':
        import SimpleITK as sitk

    if args.clear_cache:
        shutil.rmtree(os.path.join(Path(__file__).parent, "nnunet_results"))
        shutil.rmtree(os.path.join(Path(__file__).parent, ".cache"))
        if not args.i or not args.o:
            print("Cache cleared and missing either -i / -o. Exiting.")
            sys.exit(0)

    if not args.i or not args.o:
        parser.error("The following arguments are required: -i, -o")

    # data conversion
    src_folder = args.i  # input folder
    if not os.path.exists(args.o):  # create output folder if it does not exist
        os.makedirs(args.o)

    des_folder = os.path.join(args.o, "renamed_image")

    # check if -i argument is a folder, list (csv), or a single file (nii.gz)
    if os.path.isdir(args.i):  # if args.i is a directory
        src_folder = args.i
        prepare_data_folder(des_folder)
        rename_dic, rename_back_dict = rename_and_copy_files(src_folder, des_folder)
        datalist_file = os.path.join(des_folder, "renaming.json")
        with open(datalist_file, "w", encoding="utf-8") as f:
            json.dump(rename_dic, f, ensure_ascii=False, indent=4)
        print(f"Renaming dic is saved to {datalist_file}")

    model_folder = os.path.join(
        Path(__file__).parent,
        "nnunet_results",
        "Dataset%s_Task%s_dlicv/nnUNetTrainer__nnUNetPlans__3d_fullres/"
        % (args.d, args.d),
    )

    if args.clear_cache:
        shutil.rmtree(os.path.join(Path(__file__).parent, "nnunet_results"))
        shutil.rmtree(os.path.join(Path(__file__).parent, ".cache"))

    # Check if model exists. If not exist, download using HuggingFace
    if not os.path.exists(model_folder):
        # HF download model
        print("DLICV model not found, downloading...")

        from huggingface_hub import snapshot_download

        local_src = Path(__file__).parent
        snapshot_download(repo_id="nichart/DLICV", local_dir=local_src)
        print("DLICV model has been successfully downloaded!")
    else:
        print("Loading the model...")

    prepare_data_folder(args.o)

    # Check for invalid arguments - advise users to see nnUNetv2 documentation
    assert args.part_id < args.num_parts, "See nnUNetv2_predict -h."

    assert args.device in [
        "cpu",
        "cuda",
        "mps",
    ], f"-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}."

    if args.device == "cpu":
        import multiprocessing

        torch.set_num_threads(
            multiprocessing.cpu_count() // 2
        )  # use half of the threads (better for PC)
        device = torch.device("cpu")
    elif args.device == "cuda":
        # multithreading in torch doesn't help nnU-Netv2 if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device("cuda")
    else:
        device = torch.device("mps")

    # exports for nnunetv2 purposes
    os.environ["nnUNet_raw"] = "/nnunet_raw/"
    os.environ["nnUNet_preprocessed"] = "/nnunet_preprocessed"
    os.environ["nnUNet_results"] = (
        "/nnunet_results"  # where model will be located (fetched from HF)
    )

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    # Keep the outputs consistent
    torch.use_deterministic_algorithms(True)

    # Initialize nnUnetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=args.verbose,
        verbose_preprocessing=args.verbose,
        allow_tqdm=not args.disable_progress_bar,
    )

    # Retrieve the model and its weight
    predictor.initialize_from_trained_model_folder(
        model_folder, args.f, checkpoint_name=args.chk
    )

    # Final prediction
    predictor.predict_from_files(
        des_folder,
        args.o,
        save_probabilities=args.save_probabilities,
        overwrite=not args.continue_prediction,
        num_processes_preprocessing=args.npp,
        num_processes_segmentation_export=args.nps,
        folder_with_segs_from_prev_stage=args.prev_stage_predictions,
        num_parts=args.num_parts,
        part_id=args.part_id,
    )

    # After prediction, convert the image name back to original, perform post processing
    files_folder = args.o

    for filename in os.listdir(files_folder):
        if filename.endswith(".nii.gz"):
            original_name = rename_back_dict[filename]
            os.rename(
                os.path.join(files_folder, filename),
                os.path.join(files_folder, original_name),
            )
            # Enable post processing based on connected component analysis
            if args.post_processing.upper() == 'TRUE':
                fpath = os.path.join(files_folder, original_name)
                # Make sure the file exists
                if os.path.exists(fpath):
                    mask_original = sitk.ReadImage(fpath)
                    mask_component, true_mask_value = analyze_connected_components_for_icv(mask_original)
                    print(f"True ICV mask value: {true_mask_value}, overwritting the output...")
                    # mask_component = sitk.ConnectedComponent(mask_original)
                    # mask_component = sitk.RelabelComponent(mask_component, sortByObjectSize=True)
                    # mask_component = sitk.Equal(mask_component, analyze_connected_components_for_icv(mask_component))
                    if mask_component.GetNumberOfPixels() > 10:
                        sitk.WriteImage(mask_component, fpath)
                    del mask_original, mask_component

    # Remove the (temporary) des_folder directory
    if os.path.exists(des_folder):
        shutil.rmtree(des_folder)

    print("DLICV Process Done!")

if __name__ == "__main__":
    main()
