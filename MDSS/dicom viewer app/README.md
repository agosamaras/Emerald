# Emerald DICOM Viewer

A DICOM viewer module for CT and PET scans, built with Python, [ttkbootstrap](https://ttkbootstrap.readthedocs.io/), and [Tkinter](https://docs.python.org/3/library/tkinter.html). This app allows you to open, visualize, crop, anonymize, and export DICOM images, with support for contrast adjustment, zoom, pan, and modality switching.

## Features

- Open and visualize DICOM folders (CT and PET modalities)
- Adjust contrast (window/level) with sliders
- Zoom and pan images interactively
- Crop and save regions of interest as PNG
- Anonymize DICOM files
- Export current slice as PNGS
- Export all slices as MP4 video
- View DICOM metadata
- UI with [ttkbootstrap](https://ttkbootstrap.readthedocs.io/)

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/emerald-dicom-viewer.git
    cd emerald-dicom-viewer
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the app:**
    ```sh
    python app.py
    ```

## Project Structure

- `app.py` — Main application file
- `app_functions.py` — Helper functions (DICOM info, anonymization, etc.)
- `utils/` — Icons and images
- `test/` — (Optional) Test scripts

## Usage

1. Launch the app.
2. Click **Open** to select a folder containing DICOM files.
3. Use the **Visualize** button to load images.
4. Adjust contrast, zoom, and pan as needed.
5. Use **Crop Image** to select and save a region of interest.
6. Export images or anonymize DICOMs using the provided buttons.

## Requirements

- Python 3.10+
- See `requirements.txt` for all dependencies.

## License

MIT License

---

**Note:** This app is for research and educational purposes. Not for clinical use.
