import argparse
import warnings

from .dlicv_pipeline import run_pipeline

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# VERSION = pkg_resources.require("NiChart_DLMUSE")[0].version
VERSION = "1.0.4"


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

    args = parser.parse_args()

    run_pipeline(
        args.in_dir,
        args.out_dir,
        args.device,
        args.clear_cache,
        args.d,
        args.part_id,
        args.num_parts,
        args.step_size,
        args.disable_tta,
        args.verbose,
        args.disable_progress_bar,
        args.chk,
        args.save_probabilities,
        args.continue_prediction,
        args.npp,
        args.nps,
        args.prev_stage_predictions,
    )


if __name__ == "__main__":
    main()
