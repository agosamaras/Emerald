import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

def main():
    st.title("Image Cropper with Zoom")

    # image upload box
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "TIFF"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # display the image
        st.write("Zoom and crop the image as needed.")
        # πerform cropping with a fixed aspect ratio of 1 (square)
        cropped_image = st_cropper(image, realtime_update=True, box_color='blue', aspect_ratio=(1,1))
        # Display the cropped image
        st.image(cropped_image, caption='Cropped Image', use_column_width=True)

        # add checkbox for proceeding
        if st.checkbox('Proceed with saving this cropped image?'):
            if st.button('Save Image'):
                # Save the cropped image
                save_path = "cropped_image.jpg"
                cropped_image.save(save_path)
                st.write(f"Cropped image saved as {save_path}")

if __name__ == "__main__":
    main()
