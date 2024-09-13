# content of tests/test_main.py
import os
import unittest

from DLICV.__main__ import main
from DLICV.utils import prepare_data_folder, rename_and_copy_files


# Fixture for creating mock paths
def mock_paths(tmp_path):
    input_path = tmp_path / "input"
    input_path.mkdir()
    (input_path / "test.nii.gz").write_text("dummy nii data")

    output_path = tmp_path / "output"
    output_path.mkdir()

    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "model.pkl").write_text("dummy model data")

    return input_path, output_path, model_path


# Fixture for setting up argparse arguments
def setup_args(mock_paths):
    input_path, output_path, model_path = mock_paths
    return [
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--model",
        str(model_path),
        # Add other necessary arguments here
    ]


class CheckUtils(unittest.TestCase):

    def test_prepare_data_folder(self):
        prepare_data_folder("test_to_remove")
        self.assertTrue(os.path.exists("test_to_remove"))

    def test_rename_and_copy_files(self):
        a, b = rename_and_copy_files(
            "../test_input/DLICV_test_images", "../test_input/DLICV_test_results"
        )
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)


class CheckInference(unittest.TestCase):

    def test_inference(self):
        try:
            # TODO: This won't run as we don't save nnunet_results anywhere, this will be worked on soon
            os.system(
                "DLICV -i ../test_input/DLICV_test_images -o .../test_input/DLICV_test_results/ -m ../nnunet_results -d 901 -f 0 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans -device cpu"
            )
        except:
            self.fail("Error raised while training")
