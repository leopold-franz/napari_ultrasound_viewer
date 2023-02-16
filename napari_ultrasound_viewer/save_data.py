import logging
from pathlib import Path
from typing import Union

import h5py
import napari
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def save_dataset_to_hdf5(
    dest_path: str,
    dataset: Union[np.ndarray, pd.DataFrame],
    dataset_name: str,
    metadata: dict = {},
):
    """
    Save dataset and its metadata to hdf5 file.

    Parameters
    ----------
    dest_path : str
        Path where destination hdf5 file exists or is to be created.
        Needs to finish with '.h5' or '.hdf5' extension.
    dataset : Union[np.ndarray, pd.DataFrame]
        Data containing the segmentation results (numpy.ndarrays if images or pd.DataFrame if analysis results).
    dataset_name : str
        Name of dataset to be saved.
    metadata : dict (optional)
        Metadata to be saved with the dataset.

    Returns
    -------
        None
    """
    logger.info(f"Saving dataset {dataset_name} to {dest_path}")
    if type(dataset) == pd.DataFrame:
        dataset.to_hdf(dest_path, key=dataset_name)
    elif type(dataset) == np.ndarray:
        with h5py.File(dest_path, "a") as f:
            f.create_dataset(
                name=dataset_name,
                data=dataset,
                shape=dataset.shape,
                compression="gzip",
                compression_opts=9,
            )
            for key, value in metadata.items():
                f[dataset_name].attrs.create(name=key, data=value)


def dict_to_hdf5(
    data_dict: dict, dest_path=Path.home() / Path("follicle_tracker/results/test.hdf5")
):
    """
    Saves the images in the data dict to an hdf5 file at the destination_path.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing the data (images as nup.ndarrays or analysis results as pd.DataFrame).
    dest_path :
        Destination Path to save the h5 file at. Needs to finish with '.h5' or '.hdf5' extension.

    Returns
    -------
        None
    """
    logger.info(f"Saving images to {dest_path}")
    for key, value in data_dict.items():
        save_dataset_to_hdf5(dest_path, value, key)
    logger.info("Saving complete")


def viewer_to_hdf5(
    napari_viewer,
    measurements_df: pd.DataFrame = None,
    dest_path=Path.home() / Path("follicle_tracker/results/test.hdf5"),
):
    """
    Saves the images and segmentation results labelled with "left.*" or "right.*" in the viewer to an hdf5 file at the destination_path.

    Parameters
    ----------
    napari_viewer : Napari Viewer
        Viewer to extract images from.
    measurements_df : pd.DataFrame
        Results from the follicle measurements
    dest_path : Path object
        Destination Path to save the h5 file at. Needs to finish with '.h5' or '.hdf5' extension.

    Returns
    -------
        None
    """
    logger.info(f"Saving images to {dest_path}")
    dest_path = Path(dest_path)
    # Create results dir in users ~/follicle_tracker dir if it does not exist yet.
    if "follicle_tracker/results" in str(dest_path.parent):
        dest_path.parent.mkdir(exist_ok=True)

    if dest_path.exists():
        logger.error(f"Following file already exists: {dest_path}")
        return None

    with h5py.File(dest_path, "a") as f:
        left = f.create_group("left")
        right = f.create_group("right")
        for layer in napari_viewer.layers:
            if isinstance(layer, napari.layers.Image):
                logger.info(f"Saving {layer.name}")
                if layer.name.split(".")[0] == "left":
                    left.create_dataset(
                        name=layer.name.split(".")[1],
                        data=np.asarray(layer.data),
                        shape=layer.data.shape,
                        compression="gzip",
                        compression_opts=9,
                    )
                elif layer.name.split(".")[0] == "right":
                    right.create_dataset(
                        name=layer.name.split(".")[1],
                        data=np.asarray(layer.data),
                        shape=layer.data.shape,
                        compression="gzip",
                        compression_opts=9,
                    )
    logger.info("Saving measurements dataframe")
    measurements_df.to_hdf(str(dest_path), "measurements")
    logger.info("Saving complete")
