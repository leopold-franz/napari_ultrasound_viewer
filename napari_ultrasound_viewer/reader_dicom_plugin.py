import warnings
from pathlib import Path
from typing import List, Tuple

import napari
import numpy as np
import pydicom
from napari_plugin_engine import napari_hook_implementation


@napari_hook_implementation
def napari_get_reader(path):
    """napari_get_reader hook specification for dicom images.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # Does not load a stack of images into one 3d image.
        # If a list of paths is given the reader will cause an error in the next function.
        return None

    if path.endswith(".dcm"):
        # We return the *function* that can read ``path``.
        return read_dicom_image
    else:
        # if we know we cannot read the file, we immediately return None.
        return None


def load_dicom_dataset(input_path: str) -> Tuple[np.ndarray, float]:
    """
    Read a dicom image and return the image array and the pixel spacing.

    Parameters
    ----------
    input_path : str
        Path of the input dicom file

    Returns
    -------
    img_arr : np.array
        Image array of the dicom image
    pixel_spacing : float
        Pixel spacing of the dicom image
    """
    dcm_image = pydicom.dcmread(input_path)
    img_arr = np.asarray(dcm_image.pixel_array)

    x_res, y_res = np.asarray(dcm_image["PixelSpacing"].value).astype(float)
    z_res = float(dcm_image["SpacingBetweenSlices"].value)

    if (x_res != y_res) or (x_res != z_res):
        warnings.warn(
            f"Pixel Spacing of {input_path} is not equal: {x_res}, {y_res}, {z_res}"
        )
    return img_arr, x_res


def read_dicom_image(path) -> List["napari.types.LayerDataTuple"]:
    """Takes a hdf5 file path and returns a list of LayerData tuples. The hdf5 file must have at least one group called "left" or "right" in the top hierarchy. All the images (raw, denoised, intermediate and final segmentations) are saved within these two groups.

    Parameters
    ----------
    path : str or list of str
        Path to HDF5 file, that should be name in the following manner:
        "{patient_id}_{date}_{other_tags}.hdf5", where other tags can be any additional tags or descriptives your would like to give the file.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list is in the form
        (data, [add_kwargs, [layer_type]]), "add_kwargs", which represents any layer metadata, and "layer_type" are both optional.

    Warnings
    --------
    The path variable should not be a list of paths.
    """
    # handle only a single hdf5 path string and no list of strings
    layer = (
        np.asarray(pydicom.dcmread(path).pixel_array),
        {"name": Path(path).stem},
        "image",
    )
    return [layer]
