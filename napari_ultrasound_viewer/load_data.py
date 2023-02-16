from pathlib import Path
import logging
import h5py

from .reader_dicom_plugin import load_dicom_dataset
from .reader_hdf5_plugin import load_hdf5_dataset

logger = logging.getLogger(__name__)

def load_input_image(image_path: str):
    """
    Load image from the given path

    Parameters
    ----------
    image_path : str
        Path to the image

    Returns
    -------
        image : np.ndarray
            Image
        voxel_size : float
            Voxel size
    """
    if Path(image_path).suffix == ".dcm":
        data, pixel_spacing = load_dicom_dataset(image_path)
    elif Path(image_path).suffix == ".h5" or Path(image_path).suffix == ".hdf5":
        with h5py.File(image_path, "r") as f:
            keys = list(f.keys())
        if "raw" in keys:
            data, metadata = load_hdf5_dataset(image_path, "raw")
        elif "raw_rescaled" in keys:
            data, metadata = load_hdf5_dataset(image_path, "raw_rescaled")
        else:
            raise KeyError(f"No 'raw' or 'raw_rescaled' dataset found in {image_path}")
    else:
        raise ValueError(
            f"Unknown image format: {image_path}. Must be .dcm or .h5/.hdf5"
        )
    try:
        pixel_spacing = metadata["pixel_spacing"]
    except KeyError:
        logger.warning("No pixel spacing value saved. Assume the image was already rescaled.")
        pixel_spacing = None
    assert len(data.shape) == 3
    assert min(data.shape) > 3

    return data, pixel_spacing
