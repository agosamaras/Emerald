import streamlit as st
from PIL import Image
from streamlit_image_zoom import image_zoom

def main():
    st.title("Image Cropper with Zoom")

    # image upload box
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "TIFF"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # display the image
        st.write("Zoom the image as needed.")
        # Display image with custom settings
        image_zoom(image, mode="scroll", size=(800, 600), keep_aspect_ratio=False, zoom_factor=4.0, increment=0.2)



if __name__ == "__main__":
    main()
