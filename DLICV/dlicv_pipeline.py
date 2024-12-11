import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

from .utils import prepare_data_folder, rename_and_copy_files


def run_pipeline(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.clear_cache:
        shutil.rmtree(os.path.join(Path(__file__).parent, "nnunet_results"))
        shutil.rmtree(os.path.join(Path(__file__).parent, ".cache"))
        if not args.in_dir or not args.out_dir:
            print("Cache cleared and missing either -i / -o. Exiting.")
            sys.exit(0)

    if not args.in_dir or not args.out_dir:
        parser.error("The following arguments are required: -i, -o")

    # data conversion
    src_folder = args.in_dir  # input folder
    if not os.path.exists(args.out_dir):  # create output folder if it does not exist
        os.makedirs(args.out_dir)

    des_folder = os.path.join(args.out_dir, "renamed_image")

    # check if -i argument is a folder, list (csv), or a single file (nii.gz)
    if os.path.isdir(args.in_dir):  # if args.i is a directory
        src_folder = args.in_dir
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

    prepare_data_folder(args.out_dir)

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
        args.out_dir,
        save_probabilities=args.save_probabilities,
        overwrite=not args.continue_prediction,
        num_processes_preprocessing=args.npp,
        num_processes_segmentation_export=args.nps,
        folder_with_segs_from_prev_stage=args.prev_stage_predictions,
        num_parts=args.num_parts,
        part_id=args.part_id,
    )

    # After prediction, convert the image name back to original
    files_folder = args.out_dir

    for filename in os.listdir(files_folder):
        if filename.endswith(".nii.gz"):
            original_name = rename_back_dict[filename]
            os.rename(
                os.path.join(files_folder, filename),
                os.path.join(files_folder, original_name),
            )
    # Remove the (temporary) des_folder directory
    if os.path.exists(des_folder):
        shutil.rmtree(des_folder)

    print("DLICV Process Done!")
