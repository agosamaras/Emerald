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

# File uploader widget
uploaded_files = st.sidebar.file_uploader("Upload DICOM files:", type="dcm", accept_multiple_files=True)

if uploaded_files:
    try:
        dicom_slices = load_dicom_files(uploaded_files)
        if dicom_slices:
            st.sidebar.header("Controls")

            # 3D Volume creation
            volume = create_3d_volume(dicom_slices)
            
            # Slice selection sliders
            num_slices_axial = volume.shape[2]
            selected_slice_axial = st.sidebar.slider("Select Axial Slice", 0, num_slices_axial - 1, 0)

            num_slices_coronal = volume.shape[1]
            selected_slice_coronal = st.sidebar.slider("Select Coronal Slice", 0, num_slices_coronal - 1, 0)

            num_slices_sagittal = volume.shape[0]
            selected_slice_sagittal = st.sidebar.slider("Select Sagittal Slice", 0, num_slices_sagittal - 1, 0)

            # Display selected slices
            slice_axial = volume[:, :, selected_slice_axial]
            slice_coronal = volume[:, selected_slice_coronal, :]
            slice_sagittal = volume[selected_slice_sagittal, :, :]

            normalized_slice_axial = normalize_to_rgb(slice_axial)
            normalized_slice_coronal = normalize_to_rgb(slice_coronal)
            normalized_slice_sagittal = normalize_to_rgb(slice_sagittal)

            st.image(normalized_slice_axial, caption=f"Axial Slice {selected_slice_axial}", use_column_width=True)
            st.image(normalized_slice_coronal, caption=f"Coronal Slice {selected_slice_coronal}", use_column_width=True)
            st.image(normalized_slice_sagittal, caption=f"Sagittal Slice {selected_slice_sagittal}", use_column_width=True)

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