"""
This module reads a sample dicom image
"""
import platform
from pathlib import Path
from typing import TYPE_CHECKING

from napari_plugin_engine import napari_hook_implementation
from pydicom import dcmread
from pydicom.data import get_testdata_file

if TYPE_CHECKING:
    import napari

from .reader_hdf5_plugin import read_hdf5_segmentation_dataset


def _load_sample_dcms() -> "napari.types.LayerDataTuple":
    """
    Loads two sample dicom images and returns the data as numpy dictionaries.

    Returns
    -------
    "napari.types.LayerDataTuple"
        Two Image layers of the sample dicom images
    """
    filename = get_testdata_file("MR_small.dcm")

    dcm_data = dcmread(filename).pixel_array
    return [(dcm_data, {"name": "left"}), (dcm_data, {"name": "right"})]


if platform.system() == "Darwin" or platform.system() == "Linux":
    sample_data_path = Path("/Volumes/iber/Projects/IVF/Follicle_Tracker/sample_data/")
elif platform.system() == "Windows":
    sample_data_path = Path("N:\\iber\\Projects\\IVF\\Follicle_Tracker\\sample_data\\")


def _load_segmented_follicles() -> "napari.types.LayerDataTuple":
    """
    Loads two sample dicom images and returns the data as numpy dictionaries.

    Returns
    -------
    "napari.types.LayerDataTuple"
        Two Image layers of the sample dicom images
    """
    filename = sample_data_path / Path("xxx_segmented.hdf5")

    hdf5_layer_list = read_hdf5_segmentation_dataset(str(filename))

    display_layer_list = []
    for (data, layer_kwargs, layer_type) in hdf5_layer_list:
        if layer_kwargs["name"] == "left.cropped_raw":
            layer_kwargs["opacity"] = 0.60
        elif layer_kwargs["name"] == "left.cropped_ovary_seg":
            layer_kwargs["opacity"] = 0.40
        elif layer_kwargs["name"] == "left.processed_follicle_segmentation":
            layer_kwargs["opacity"] = 0.50
            layer_kwargs["colormap"] = "inferno"
        else:
            layer_kwargs["visible"] = False

        display_layer_list.append((data, layer_kwargs, layer_type))

    return display_layer_list


@napari_hook_implementation
def napari_provide_sample_data():
    return {
        "Sample Dicom Image": _load_sample_dcms,
        "Follicle_Ultrasounds_Raw": sample_data_path / Path("A01_130417_raw.hdf5"),
        "Follicle_Ultrasounds_All": _load_segmented_follicles,
    }
