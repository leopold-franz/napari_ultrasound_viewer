# napari_ultrasound_viewer
Napari plugins to open Dicom and H5 files

## Usage
Once you have downloaded napari_ultrasound_viewer correctly and installed the required packages in your python environment, you can start napari 
by typing the following in the terminal:

```bash
napari
```

Open the napari_ultrasound_viewer plugin in napari by navigating to the plugin menu at the top and clicking on 
"napari-ultrasound-viewer: ..." option.

You can import 3D images in the standard DICOM format (".dcm") or Hierarchical Dataset Format 5 (HDF5) 
files (".h5"/".hdf5") by drag and dropping them into the napari viewer. Note that the raw image arrays stored 
in the HDF5 files need to be stored in the top hierarchical layer.


## License

Distributed under the terms of the [BSD-3] license,
"napari-ultrasound-viewer" is free and open source software