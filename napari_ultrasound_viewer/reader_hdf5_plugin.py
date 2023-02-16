import logging
from typing import List

import h5py
import napari
import numpy as np
import pandas as pd
from napari_plugin_engine import napari_hook_implementation

logger = logging.getLogger(__name__)


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

    if path.endswith(".hdf5") or path.endswith(".h5"):
        # We return the *function* that can read ``path``.
        return read_hdf5_segmentation_dataset
    else:
        # if we know we cannot read the file, we immediately return None.
        return None


def load_hdf5_dataset(path: str, dataset_name: str):
    """Load a dataset from a HDF5 file along with its metadata.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.
    dataset_name : str
        Name of the dataset to be loaded.

    Returns
    -------
    dataset : Union[np.ndarray, pd.DataFrame]
        The dataset loaded from the HDF5 file.
    metadata : dict
        The metadata of the dataset.
    """
    try:
        with h5py.File(path, "r") as f:
            dataset = f[dataset_name][:]
            meta_data = dict()
            for data in f[dataset_name].attrs:
                meta_data[data] = f[dataset_name].attrs[data]
        return dataset, meta_data
    except TypeError:
        try:
            dataset = pd.read_hdf(path, dataset_name)
            return dataset, None
        except TypeError:
            raise TypeError("The dataset is not a numpy Array nor a pandas DataFrame.")


def read_hdf5_segmentation_dataset(path) -> List["napari.types.LayerDataTuple"]:
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
    if isinstance(path, list):
        # Only read first file in list
        path = path[0]

    if isinstance(path, str):
        logger.info(f"Reading HDF5/H5 file at {path}")
        image_layers = []
        measurements_loaded = False
        with h5py.File(path, "r") as f:
            if "left" in f.keys() or "right" in f.keys():
                for side in ["left", "right"]:
                    try:
                        for img_name in f[side].keys():
                            # optional kwargs for the corresponding viewer.add_* method
                            add_kwargs = {"name": f"{side}.{img_name}"}
                            if (
                                "padded_follicle_labels" in img_name
                                and "measurements" in f.keys()
                                and not measurements_loaded
                            ):
                                add_kwargs["metadata"] = {
                                    "measurements": pd.read_hdf(path, "measurements")
                                }
                            layer_type = "image"  # optional, default is "image"
                            layer = (
                                np.asarray(f[side][img_name]),
                                add_kwargs,
                                layer_type,
                            )
                            image_layers.append(layer)
                    except KeyError:
                        logger.info(f"No image from {side} side.")
            else:
                for img_name in f.keys():
                    # optional kwargs for the corresponding viewer.add_* method
                    add_kwargs = {"name": img_name}

                    layer_type = "image"  # optional, default is "image"
                    layer = (
                        np.asarray(f[img_name]),
                        add_kwargs,
                        layer_type,
                    )
                    image_layers.append(layer)
        if len(image_layers):
            return image_layers
        else:
            logger.error("No images found in the HDF5 file 'left' or 'right' group.")
    else:
        logger.error("The HDF5 file reader does not support HDF5 path lists.")
