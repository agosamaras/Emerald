import tkinter as tk
from PIL import ImageTk, Image
import os

# Define your classes
classes = ["benign", "malignant"]

# Initialize variables
num_imgs = 243 # Adjust the size to match the number of images
current_image_index = 0
labels = [None] * num_imgs

# Load images from a directory
root_fs = f"F:/nsclc/Test1.v1i.folder"
image_directory = f'{root_fs}/whole'  # Replace with your image directory

# Function to handle label selection
def label_selected(class_index):
    global current_image_index, labels

    # Save the label for the current image
    labels[current_image_index] = class_index

    # Move to the next image
    current_image_index += 1

    # If all images are labeled, exit the application
    if current_image_index >= len(images):
        root.quit()
        return

    # Display the next image and its name
    img = ImageTk.PhotoImage(images[current_image_index])
    canvas.itemconfig(image_canvas, image=img)
    canvas.image = img
    image_name.config(text=image_names[current_image_index])

# Create the Tkinter window
root = tk.Tk()
root.title("Image Labeling")

# Load the images and names
images = []
image_names = []
for filename in os.listdir(image_directory):
    if filename.endswith(".jpg"):
        image_path = os.path.join(image_directory, filename)
        image = Image.open(image_path)
        images.append(image)
        image_names.append(filename)

# Create the Canvas to display images
canvas = tk.Canvas(root, width=600, height=600)
canvas.pack()

# Display the first image and its name
img = ImageTk.PhotoImage(images[0])
image_canvas = canvas.create_image(0, 0, anchor=tk.NW, image=img)
image_name = tk.Label(root, text=image_names[0])
image_name.pack()

# Create buttons for each class
buttons = []
for i, class_label in enumerate(classes):
    button = tk.Button(root, text=class_label, command=lambda class_index=i: label_selected(class_index))
    button.pack()
    buttons.append(button)

# Start the Tkinter event loop
root.mainloop()

# Print the labeled data
for i, label in enumerate(labels):
    if label is not None:
        print(f"Image {image_names[i]}: {classes[label]}")
    else:
        print(f"Image {image_names[i]}: No label")
