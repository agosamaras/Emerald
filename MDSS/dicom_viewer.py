import streamlit as st
import pydicom
import os
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
from PIL import Image
from skimage.transform import resize

# Function to load DICOM files from uploaded files
def load_dicom_files(uploaded_files):
    dicom_slices = [pydicom.dcmread(file) for file in uploaded_files]
    dicom_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))  # Sort by slice position
    return dicom_slices

# Function to create 3D volume
def create_3d_volume(dicom_slices):
    reference_shape = dicom_slices[0].pixel_array.shape  # Use the first slice as a reference
    slices = [
        resize(s.pixel_array, reference_shape, mode='constant', preserve_range=True).astype(s.pixel_array.dtype)
        if s.pixel_array.shape != reference_shape else s.pixel_array
        for s in dicom_slices
    ]
    volume = np.stack(slices, axis=-1)
    return volume

# Function to normalize and convert a 2D slice to RGB
@st.cache_data
def normalize_to_rgb(slice_2d):
    norm_slice = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d)) * 255
    return Image.fromarray(norm_slice.astype(np.uint8))

# Main Streamlit app
st.title("DICOM Viewer")

# Function to create 3D volume with memory optimization
def create_3d_volume(dicom_slices):
    reference_shape = dicom_slices[0].pixel_array.shape  # Use the first slice as a reference
    slices = [
        resize(s.pixel_array, reference_shape, mode='constant', preserve_range=True).astype(s.pixel_array.dtype)
        if s.pixel_array.shape != reference_shape else s.pixel_array
        for s in dicom_slices
    ]
    volume = np.stack(slices, axis=-1)
    return volume

# File uploader widget
uploaded_files = st.sidebar.file_uploader("Upload DICOM files:", type="dcm", accept_multiple_files=True)

if uploaded_files:
    try:
        dicom_slices = load_dicom_files(uploaded_files)
        if dicom_slices:
            st.sidebar.header("Controls")

            # 3D Volume creation
            volume = create_3d_volume(dicom_slices)
            
            # Slice selection slider
            num_slices = volume.shape[2]
            selected_slice = st.sidebar.slider("Select Slice", 0, num_slices - 1, 0)

            # Zoom selection
            zoom_factor = st.sidebar.slider("Zoom", 1.0, 5.0, 1.0)

            # Grouping option
            group_size = st.sidebar.slider("Group Size (for averaging slices)", 1, 10, 1)

            # Display selected slice
            slice_2d = volume[:, :, selected_slice]
            grouped_slice = np.mean(volume[:, :, selected_slice:selected_slice + group_size], axis=2)

            normalized_slice = normalize_to_rgb(grouped_slice if group_size > 1 else slice_2d)
            st.image(normalized_slice, caption=f"Slice {selected_slice}", use_column_width=True)

            # Interactive 3D Visualization using Plotly
            st.sidebar.write("3D View")
            plot_3d = st.sidebar.checkbox("Enable 3D Volume View")

            if plot_3d:
                try:
                    x, y, z = np.mgrid[0:volume.shape[0]:complex(0, volume.shape[0]),
                                       0:volume.shape[1]:complex(0, volume.shape[1]),
                                       0:volume.shape[2]:complex(0, volume.shape[2])]

                    fig = go.Figure(data=go.Volume(
                        x=x.flatten(),
                        y=y.flatten(),
                        z=z.flatten(),
                        value=volume.flatten(),
                        isomin=volume.min(),
                        isomax=volume.max(),
                        opacity=0.1,  # Transparency of volume
                        surface_count=20,  # Number of surfaces to create
                        colorscale="Gray",
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                except MemoryError:
                    st.error("Memory Error: The 3D volume visualization requires too much memory.")
                except Exception as e:
                    st.error(f"Error creating 3D volume visualization: {e}")

        else:
            st.error("No DICOM files found in the uploaded files.")
    except Exception as e:
        st.error(f"Error loading DICOM files: {e}")
