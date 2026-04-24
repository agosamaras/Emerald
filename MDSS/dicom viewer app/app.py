import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk, ImageOps
from tkinter import filedialog
from tkinter.messagebox import showinfo
from glob import glob
import os
import numpy as np
import pydicom
import app_functions
import cv2
import sys
from screeninfo import get_monitors

# Function to get the correct path for bundled files
def resource_path(relative_path):
    """Get the absolute path to a resource, works for dev and for PyInstaller."""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

############### Initialize the App ###############
app = ttk.Window()

# Get screen dimensions of the primary monitor
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

# Set window size relative to screen size
window_width = int(screen_width * 0.7)
window_height = int(screen_height * 0.7)
app.geometry(f"{window_width}x{window_height}")  # width and height of the window

# Enable full-screen mode
app.attributes('-fullscreen', True)

# Function to toggle full-screen mode
def toggle_fullscreen(event=None):
    app.attributes('-fullscreen', not app.attributes('-fullscreen'))

# Bind the Escape key to exit full-screen mode
app.bind('<Escape>', toggle_fullscreen)

# Set theme
app.style.theme_use('morph')
app.title('EMERALD DICOM Viewer')

# Set viewer size relative to window size
width_height_viewer = int(window_width * 0.6)

# Load the logo
img = Image.open(resource_path('./utils/logo_emerald.png'))
img = img.resize((int(width_height_viewer*0.5), int(width_height_viewer*0.5)))
img_tk = ImageTk.PhotoImage(img)

# Set font sizes relative to window size
title_font_size = int(window_width * 0.04)
label_font_size = int(window_width * 0.01)

app_title = ttk.Label(app, text='EMERALD DICOM', font=f'poppins {title_font_size} bold')

app_title.pack()

############### Global Variables ###############
global dicom_files
global ct_files
global pet_files
global max_v
global min_v
global actual_slice_number
global on_off_click
global zoom_level
global crop_start_x, crop_start_y, crop_end_x, crop_end_y
global is_zoom_mode
global tk_image
global current_modality
global pan_start_x, pan_start_y
global pan_offset_x, pan_offset_y
global is_crop_mode
global crop_rect

dicom_files = None
ct_files = []
pet_files = []
max_v = None
min_v = None
actual_slice_number = None
on_off_click = True
zoom_level = 1.0
crop_start_x, crop_start_y, crop_end_x, crop_end_y = None, None, None, None
is_zoom_mode = False
tk_image = None
current_modality = "CT"  # Default modality
pan_start_x, pan_start_y = None, None
pan_offset_x, pan_offset_y = 0, 0
is_crop_mode = False
crop_rect = None

############### Functions ###############
def return_min_max(dcm_file):
    array = pydicom.dcmread(dcm_file).pixel_array
    return np.min(array), np.max(array)

def open_dicoms():
    global dicom_files, ct_files, pet_files
    path_dicoms = filedialog.askdirectory()
    dicom_files = sorted(glob(os.path.join(path_dicoms, '*.dcm')))
    
    if dicom_files:
        # Group slices by modality
        ct_files = []
        pet_files = []
        for file in dicom_files:
            dcm = pydicom.dcmread(file)
            if hasattr(dcm, 'Modality'):
                if dcm.Modality == 'CT':
                    ct_files.append(file)
                elif dcm.Modality == 'PT':
                    pet_files.append(file)
        
        if ct_files or pet_files:
            notification_label.configure(text='Folder contains CT and/or PET files', bootstyle='success')
        else:
            notification_label.configure(text='Folder does not contain CT or PET files', bootstyle='warning')
    else:
        notification_label.configure(text='Folder does not contain dicom files', bootstyle='warning')

def show(img):
    global tk_image, pan_offset_x, pan_offset_y, crop_rect
    # Create a blank black image of the canvas size
    blank_image = Image.new("RGB", (width_height_viewer, width_height_viewer), "black")
    
    # Resize the slice based on zoom level
    img_width, img_height = img.size
    img = img.resize((int(img_width * zoom_level), int(img_height * zoom_level)))
    
    # Calculate the position to paste the slice (centered)
    x = (width_height_viewer - img.width) // 2 + pan_offset_x
    y = (width_height_viewer - img.height) // 2 + pan_offset_y
    
    # Paste the slice onto the blank image
    blank_image.paste(img, (x, y))
    
    # Convert to PhotoImage and display on canvas
    tk_image = ImageTk.PhotoImage(blank_image)
    canvas_viewer.create_image(0, 0, anchor=NW, image=tk_image)
    
    # Draw the crop rectangle if in crop mode
    if is_crop_mode and crop_rect:
        canvas_viewer.create_rectangle(crop_rect, outline="red", width=2, tags="crop_rect")

def prepare_dicoms(dcm_file, show=False, max_v=None, min_v=None):
    dicom_file_data = pydicom.dcmread(dcm_file).pixel_array

    if max_v:
        HOUNSFIELD_MAX = int(float(max_v))
    else:
        HOUNSFIELD_MAX = np.max(dicom_file_data)
    if min_v:
        HOUNSFIELD_MIN = int(float(min_v))
    else:
        HOUNSFIELD_MIN = np.min(dicom_file_data)

    HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN

    dicom_file_data[dicom_file_data < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
    dicom_file_data[dicom_file_data > HOUNSFIELD_MAX] = HOUNSFIELD_MAX
    normalized_image = (dicom_file_data - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE
    uint8_image = np.uint8(normalized_image * 255)

    return uint8_image

def visualize(slice_number=0):
    global actual_slice_number, min_v, max_v, current_modality, pan_offset_x, pan_offset_y
    if current_modality == "CT" and ct_files:
        files = ct_files
    elif current_modality == "PET" and pet_files:
        files = pet_files
    else:
        notification_label.configure(text='No files available for the selected modality', bootstyle='warning')
        return

    if files:
        actual_slice_number = slice_number
        slice_path = files[int(float(slice_number))]
        dcm_min_value, dcm_max_value = return_min_max(slice_path)
        slider_slices.configure(state='enabled', from_=0, to=len(files) - 1, value=actual_slice_number)
        slider_max.configure(state='enabled', value=dcm_max_value)
        slider_min.configure(state='enabled', value=dcm_min_value)
        slider_zoom.configure(state='enabled', value=zoom_level * 100)  # Set zoom slider to current zoom level
        apply_button.configure(state='enabled')
        change_button.configure(state='disabled')
        slider_slices_value.configure(text=int(float(actual_slice_number)))
        slider_max_value.configure(text=int(float(slider_max.get())))
        slider_min_value.configure(text=int(float(slider_min.get())))
        slider_zoom_value.configure(text=f'{int(float(slider_zoom.get()))}%')

        normalized_slice = prepare_dicoms(slice_path)
        img = Image.fromarray(normalized_slice)
        
        # Center the image in the canvas
        show(img)
        
        max_v = dcm_max_value
        min_v = dcm_min_value
        
        # Update the modality button text based on the current slice's modality
        dcm = pydicom.dcmread(slice_path)
        if hasattr(dcm, 'Modality'):
            if dcm.Modality == 'CT':
                modality_button.configure(text='Switch to PET')
            elif dcm.Modality == 'PT':
                modality_button.configure(text='Switch to CT')

        notification_label.configure(text='You can now adjust the contrast, zoom, and pan', bootstyle='info')

def scroll_slider(slice_number=0):
    global max_v, min_v, on_off_click, actual_slice_number, current_modality
    if current_modality == "CT" and ct_files:
        files = ct_files
    elif current_modality == "PET" and pet_files:
        files = pet_files
    else:
        return

    if files:
        slice_path = files[int(float(slice_number))]
        dcm_min_value, dcm_max_value = return_min_max(slice_path)
        slider_slices.configure(state='enabled', from_=0, to=len(files) - 1)
        slider_slices_value.configure(text=int(float(slice_number)))

        if on_off_click:
            slider_max.configure(value=dcm_max_value)
            slider_min.configure(value=dcm_min_value)
            normalized_slice = prepare_dicoms(slice_path, max_v=dcm_max_value, min_v=dcm_min_value)
        else:
            normalized_slice = prepare_dicoms(slice_path, max_v=max_v, min_v=min_v)

        actual_slice_number = slice_number
        img = Image.fromarray(normalized_slice)
        show(img)

def change_max(value):
    global max_v, min_v, actual_slice_number, current_modality
    if current_modality == "CT" and ct_files:
        files = ct_files
    elif current_modality == "PET" and pet_files:
        files = pet_files
    else:
        return

    if files:
        max_v = value
        slider_max_value.configure(text=int(float(value)))
        slice_path = files[int(float(actual_slice_number))]
        normalized_slice = prepare_dicoms(slice_path, max_v=int(float(max_v)), min_v=int(float(min_v)))
        img = Image.fromarray(normalized_slice)
        show(img)

def change_min(value):
    global min_v, max_v, actual_slice_number, current_modality
    if current_modality == "CT" and ct_files:
        files = ct_files
    elif current_modality == "PET" and pet_files:
        files = pet_files
    else:
        return

    if files:
        min_v = value
        slider_min_value.configure(text=int(float(value)))
        slice_path = files[int(float(actual_slice_number))]
        normalized_slice = prepare_dicoms(slice_path, min_v=int(float(min_v)), max_v=int(float(max_v)))
        img = Image.fromarray(normalized_slice)
        show(img)

def change_zoom(value):
    global zoom_level, pan_offset_x, pan_offset_y
    zoom_level = float(value) / 100  # Convert slider value to zoom level
    slider_zoom_value.configure(text=f'{int(float(value))}%')
    pan_offset_x, pan_offset_y = 0, 0  # Reset pan offset when zooming
    visualize(actual_slice_number)  # Update the image with the new zoom level

def start_pan(event):
    global pan_start_x, pan_start_y
    pan_start_x, pan_start_y = event.x, event.y

def move_pan(event):
    global pan_offset_x, pan_offset_y, pan_start_x, pan_start_y
    if pan_start_x is not None and pan_start_y is not None:
        # Calculate the new pan offset
        pan_offset_x += (event.x - pan_start_x)
        pan_offset_y += (event.y - pan_start_y)
        pan_start_x, pan_start_y = event.x, event.y
        visualize(actual_slice_number)  # Update the image with the new pan offset

def reset_pan(event):
    global pan_start_x, pan_start_y
    pan_start_x, pan_start_y = None, None

def toggle_crop_mode():
    global is_crop_mode, crop_rect
    is_crop_mode = not is_crop_mode
    if is_crop_mode:
        crop_button.configure(text="Disable Crop", bootstyle='danger')
        # Bind crop-related mouse events
        canvas_viewer.bind("<ButtonPress-1>", start_crop)
        canvas_viewer.bind("<B1-Motion>", update_crop)
        canvas_viewer.bind("<ButtonRelease-1>", end_crop)
    else:
        crop_button.configure(text="Crop Image", bootstyle='light')
        # Unbind crop-related mouse events
        canvas_viewer.unbind("<ButtonPress-1>")
        canvas_viewer.unbind("<B1-Motion>")
        canvas_viewer.unbind("<ButtonRelease-1>")
        canvas_viewer.delete("crop_rect")
        crop_rect = None
        
        # Re-bind pan-related mouse events
        canvas_viewer.bind("<ButtonPress-1>", start_pan)
        canvas_viewer.bind("<B1-Motion>", move_pan)
        canvas_viewer.bind("<ButtonRelease-1>", reset_pan)

def start_crop(event):
    global crop_start_x, crop_start_y
    crop_start_x, crop_start_y = event.x, event.y

def update_crop(event):
    global crop_rect
    crop_end_x, crop_end_y = event.x, event.y
    canvas_viewer.delete("crop_rect")
    crop_rect = (crop_start_x, crop_start_y, crop_end_x, crop_end_y)
    canvas_viewer.create_rectangle(crop_rect, outline="red", width=2, tags="crop_rect")

def end_crop(event):
    global crop_rect
    if crop_rect:
        save_cropped_image()

def save_cropped_image():
    global crop_rect, current_modality, zoom_level, pan_offset_x, pan_offset_y
    if current_modality == "CT" and ct_files:
        files = ct_files
    elif current_modality == "PET" and pet_files:
        files = pet_files
    else:
        return

    if files and crop_rect:
        path_to_save = filedialog.askdirectory()
        if path_to_save:
            # Get the current slice being viewed
            slice_path = files[int(float(actual_slice_number))]
            image_name = os.path.basename(slice_path)[:-4]
            
            # Prepare the current slice for saving
            array = prepare_dicoms(slice_path, max_v=max_v, min_v=min_v)
            img = Image.fromarray(array)
            
            # Get the crop coordinates relative to the canvas
            x1, y1, x2, y2 = crop_rect
            
            # Adjust for zoom and pan
            x1 = int((x1 - (width_height_viewer - img.width * zoom_level) // 2 - pan_offset_x) / zoom_level)
            y1 = int((y1 - (width_height_viewer - img.height * zoom_level) // 2 - pan_offset_y) / zoom_level)
            x2 = int((x2 - (width_height_viewer - img.width * zoom_level) // 2 - pan_offset_x) / zoom_level)
            y2 = int((y2 - (width_height_viewer - img.height * zoom_level) // 2 - pan_offset_y) / zoom_level)
            
            # Ensure coordinates are within the image bounds
            x1 = max(0, min(x1, img.width))
            y1 = max(0, min(y1, img.height))
            x2 = max(0, min(x2, img.width))
            y2 = max(0, min(y2, img.height))
            
            # Crop the image
            if x2 > x1 and y2 > y1:  # Ensure valid crop region
                cropped_img = img.crop((x1, y1, x2, y2))
                
                # Save the cropped image
                cropped_img.save(f'{path_to_save}/{image_name}_cropped.png')
                showinfo(message='The cropped image has been saved!')
            else:
                showinfo(message='Invalid crop region!')

def apply():
    global min_v, max_v, on_off_click
    on_off_click = False
    slider_max.configure(state='disabled')
    slider_min.configure(state='disabled')
    apply_button.configure(state='disabled')
    change_button.configure(state='enabled')

def change():
    global min_v, max_v, actual_slice_number, on_off_buttons
    on_off_click = True

    slider_max.configure(state='enabled', value=int(float(slider_max.get())))
    slider_min.configure(state='enabled', value=int(float(slider_min.get())))
    apply_button.configure(state='enabled')
    change_button.configure(state='disabled')
        
def show_info():
    global actual_slice_number, current_modality
    if current_modality == "CT" and ct_files:
        files = ct_files
    elif current_modality == "PET" and pet_files:
        files = pet_files
    else:
        notification_label.configure(text='No files available for the selected modality', bootstyle='warning')
        return

    if files:
        # Get the current slice being viewed
        slice_path = files[int(float(actual_slice_number))]
        patient = pydicom.dcmread(slice_path)
        infos_1, infos_2 = app_functions.return_information(patient)

        # Create a new window to display the information
        info_window = ttk.Toplevel(app)
        info_window.title('File Information')
        info_window.geometry(f'{int(window_width * 0.8)}x{int(window_height * 0.8)}')

        info_title = ttk.Label(info_window, text='File Information', font=f'poppins {int(title_font_size * 0.8)} bold')
        info_title.pack()
        info_label_frame = ttk.Frame(info_window)
        info_label_frame.pack()
        info_label_1 = ttk.Label(info_label_frame, text=infos_1, font=f'poppins {label_font_size}')
        info_label_1.grid(row=0, column=0, padx=(0, 30))
        info_label_2 = ttk.Label(info_label_frame, text=infos_2, font=f'poppins {label_font_size}')
        info_label_2.grid(row=0, column=1)

def save_png():
    global current_modality
    if current_modality == "CT" and ct_files:
        files = ct_files
    elif current_modality == "PET" and pet_files:
        files = pet_files
    else:
        return

    if files:
        path_to_save = filedialog.askdirectory()
        if path_to_save:
            # Get the current slice being viewed
            slice_path = files[int(float(actual_slice_number))]
            image_name = os.path.basename(slice_path)[:-4]
            
            # Prepare the current slice for saving
            array = prepare_dicoms(slice_path, max_v=max_v, min_v=min_v)
            image = Image.fromarray(array)
            
            # Save the current slice as a PNG
            image.save(f'{path_to_save}/{image_name}.png')
            
            showinfo(message='The current slice has been saved!')

def anonymize():
    if dicom_files:
        path_to_anonymize = filedialog.askdirectory()
        if path_to_anonymize:
            for dicom_file in dicom_files:
                image_name = os.path.basename(dicom_file)[:-4]
                anonymized_file = app_functions.anonymize_case(dicom_file)
                anonymized_file.save_as(f'{path_to_anonymize}/{image_name}.dcm')
            
            showinfo(message='The anonymization completed!')

def convert_to_mp4():
    global current_modality
    if current_modality == "CT" and ct_files:
        files = ct_files
    elif current_modality == "PET" and pet_files:
        files = pet_files
    else:
        return

    if files:
        path_to_mp4 = filedialog.askdirectory()
        if path_to_mp4:
            one_case = pydicom.dcmread(files[0]).pixel_array
            frameSize = one_case.shape
            out = cv2.VideoWriter(f'{path_to_mp4}/output_video.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, frameSize)

            for dicom_file in files:
                img = prepare_dicoms(dicom_file, max_v=max_v, min_v=min_v)
                cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                out.write(cv2_img)

            out.release()
            showinfo(message='The MP4 created!')

def switch_modality():
    global current_modality
    if current_modality == "CT":
        current_modality = "PET"
    else:
        current_modality = "CT"
    modality_button.configure(text=f'Switch to {current_modality}')
    visualize(0)  # Reset visualization for the new modality

############### GUI Layout ###############
frame_buttons_viewer = ttk.Frame(app)
frame_buttons_viewer.pack()

notification_label = ttk.Label(app, text='Please select a dicom directory', bootstyle='danger', font=f'poppins {label_font_size}')
notification_label.pack()

frame_buttons = ttk.Frame(frame_buttons_viewer)
frame_buttons.grid(row=0, column=0)

frame_viewer = ttk.Frame(frame_buttons_viewer)
frame_viewer.grid(row=1, column=0)

# Add a solid frame around the canvas
frame_canvas = ttk.Frame(frame_viewer, borderwidth=2, relief="solid")
frame_canvas.pack(pady=(20, 0))

canvas_viewer = ttk.Canvas(frame_canvas, width=int(width_height_viewer*0.5), height=int(width_height_viewer*0.5), bg="black")
canvas_viewer.pack()

canvas_viewer.create_image(0, 0, anchor=NW, image=img_tk)

# Bind mouse events for panning
canvas_viewer.bind("<ButtonPress-1>", start_pan)
canvas_viewer.bind("<B1-Motion>", move_pan)
canvas_viewer.bind("<ButtonRelease-1>", reset_pan)

############### Buttons ###############
open_button = ttk.Button(frame_buttons, text='Open', bootstyle='light', width=15, command=open_dicoms)
open_button.grid(row=0, column=0, padx=(5, 5), pady=(5, 5))  # Reduced padding

visualize_button = ttk.Button(frame_buttons, text='Visualize', bootstyle='light', width=15, command=visualize)
visualize_button.grid(row=0, column=1, padx=(5, 5), pady=(5, 5))  # Reduced padding

show_info_button = ttk.Button(frame_buttons, text='Show info', bootstyle='light', width=15, command=show_info)
show_info_button.grid(row=1, column=0, padx=(5, 5), pady=(5, 5))  # Reduced padding

anonymize_button = ttk.Button(frame_buttons, text='Anonymize', bootstyle='light', width=15, command=anonymize)
anonymize_button.grid(row=1, column=1, padx=(5, 5), pady=(5, 5))  # Reduced padding

save_png_button = ttk.Button(frame_buttons, text='Save PNG', bootstyle='light', width=15, command=save_png)
save_png_button.grid(row=2, column=0, padx=(5, 5), pady=(5, 5))  # Reduced padding

mp4_button = ttk.Button(frame_buttons, text='MP4', bootstyle='light', width=15, command=convert_to_mp4)
mp4_button.grid(row=2, column=1, padx=(5, 5), pady=(5, 5))  # Reduced padding

modality_button = ttk.Button(frame_buttons, text='Switch to PET', bootstyle='light', width=15, command=switch_modality)
modality_button.grid(row=3, column=0, padx=(5, 5), pady=(5, 5))  # Reduced padding

crop_button = ttk.Button(frame_buttons, text='Crop Image', bootstyle='light', width=15, command=toggle_crop_mode)
crop_button.grid(row=3, column=1, padx=(5, 5), pady=(5, 5))  # Reduced padding


style = ttk.Style()
style.configure('TButton', font=('poppins', int(window_width * 0.01)))
exit_button = ttk.Button(
    app, 
    text="Exit", 
    command=app.destroy, 
    width=20,  # Increase width
    style='TButton'  # Use the configured style
)
exit_button.pack(pady=5)

############### Sliders ###############
contrast_field = ttk.Frame(app)
contrast_field.pack()

# Slice Slider
slider_slices_label = ttk.Label(contrast_field, text='Slice', font=f'poppins {label_font_size}')
slider_slices_label.grid(pady=(30, 0), padx=(0, 20), row=0, column=0)

slider_slices = ttk.Scale(contrast_field, from_=0, to=1000, length=int(window_width * 0.5), command=scroll_slider, state="disabled")
slider_slices.grid(pady=(30, 0), padx=(0, 20), row=0, column=1)

slider_slices_value = ttk.Label(contrast_field, text=int(float(slider_slices.get())), font=f'poppins {label_font_size}')
slider_slices_value.grid(pady=(30, 0), row=0, column=2)

# Max Slider
slider_max_label = ttk.Label(contrast_field, text='Max', font=f'poppins {label_font_size}')
slider_max_label.grid(pady=(30, 0), padx=(0, 20), row=1, column=0)

slider_max = ttk.Scale(contrast_field, from_=-3000, to=3000, length=int(window_width * 0.5), command=change_max, state="disabled")
slider_max.grid(pady=(30, 0), padx=(0, 20), row=1, column=1)

slider_max_value = ttk.Label(contrast_field, text=int(float(slider_max.get())), font=f'poppins {label_font_size}')
slider_max_value.grid(pady=(30, 0), row=1, column=2)

# Min Slider
slider_min_label = ttk.Label(contrast_field, text='Min', font=f'poppins {label_font_size}')
slider_min_label.grid(pady=(30, 0), padx=(0, 20), row=2, column=0)

slider_min = ttk.Scale(contrast_field, from_=-3000, to=3000, length=int(window_width * 0.5), command=change_min, state="disabled")
slider_min.grid(pady=(30, 0), padx=(0, 20), row=2, column=1)

slider_min_value = ttk.Label(contrast_field, text=int(float(slider_min.get())), font=f'poppins {label_font_size}')
slider_min_value.grid(pady=(30, 0), row=2, column=2)

# Zoom Slider
slider_zoom_label = ttk.Label(contrast_field, text='Zoom', font=f'poppins {label_font_size}')
slider_zoom_label.grid(pady=(30, 0), padx=(0, 20), row=3, column=0)

slider_zoom = ttk.Scale(contrast_field, from_=50, to=200, length=int(window_width * 0.5), command=change_zoom, state="disabled")
slider_zoom.grid(pady=(30, 0), padx=(0, 20), row=3, column=1)

slider_zoom_value = ttk.Label(contrast_field, text='100%', font=f'poppins {label_font_size}')
slider_zoom_value.grid(pady=(30, 0), row=3, column=2)

on_off_buttons = ttk.Frame(app)
on_off_buttons.pack()

apply_button = ttk.Button(on_off_buttons, text="Apply", bootstyle='success, outline', width=20, command=apply, state='disabled')
apply_button.grid(pady=(20, 0), padx=(0, 10), row=0, column=0)

change_button = ttk.Button(on_off_buttons, text="Change", bootstyle='warning, outline', width=20, command=change, state='disabled')
change_button.grid(pady=(20, 0), padx=(10, 0), row=0, column=1)

app.mainloop()
