import nibabel as nib
from nnunet.inference.predict import predict_from_folder
from nnunet.network_architecture.neural_network import SegmentationNetwork
from pathlib import Path

def load_model(model_path):
    """
    Load the trained nnUNet model.
    :param model_path: Path to the saved model.
    :return: Loaded model.
    """
    # This is a placeholder function, adjust the loading method according to how nnUNet loads the model
    model = SegmentationNetwork()
    model.load_state_dict(model_path)
    model.eval()
    return model

def compute_volume(input_path):
    """
    Compute the intracranial volume from the input image using the nnUNet model.
    :param input_path: Path to the input image (single image or directory of images).
    :return: The output image with the computed volume or a list of output images.
    """
    # Load your nnUNet model (provide the correct model path here)
    model = load_model('path_to_saved_model')

    # If input_path is a directory, you'll need to modify this to handle each image
    if isinstance(input_path, (str, Path)) and Path(input_path).is_file():
        input_img = nib.load(str(input_path))
        # Assuming your nnUNet predict function takes a nibabel object and returns a nibabel object
        prediction = predict_from_folder(input_img, model)
        return prediction
    elif Path(input_path).is_dir():
        predictions = []
        for file_path in Path(input_path).glob('*.nii.gz'):
            input_img = nib.load(str(file_path))
            prediction = predict_from_folder(input_img, model)
            predictions.append(prediction)
        return predictions
    else:
        raise ValueError(f"The input path {input_path} is not a file or directory.")
