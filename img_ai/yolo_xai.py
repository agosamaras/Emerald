import os
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load the YOLOv8 model
yolo_model_path = 'F:/src/runs/classify/train8/weights/best.pt' # train7 -> ct, train8 -> pet
model = YOLO(yolo_model_path)

# Function to preprocess an image object
def preprocess_image(image, size=(640, 640)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# Function for Feature Ablation
def feature_ablation(model, image, mask_size=60):
    original_tensor = preprocess_image(image)
    num_rows = (image.height + mask_size - 1) // mask_size
    num_cols = (image.width + mask_size - 1) // mask_size
    heatmap = np.zeros((num_rows, num_cols))

    # Perform inference on the original image
    with torch.no_grad():
        original_pred = model(original_tensor)
    original_confidence = original_pred[0].probs.top1conf.item()

    for i in range(0, image.width, mask_size):
        for j in range(0, image.height, mask_size):
            ablated_image = image.copy()
            draw = ImageDraw.Draw(ablated_image)
            draw.rectangle([i, j, i + mask_size, j + mask_size], fill="black")
            
            ablated_tensor = preprocess_image(ablated_image)
            with torch.no_grad():
                ablated_pred = model(ablated_tensor)
            ablated_confidence = ablated_pred[0].probs.top1conf.item()

            diff = np.abs(original_confidence - ablated_confidence)
            heatmap[j // mask_size, i // mask_size] = diff

    return heatmap

# Function to overlay the heatmap on the image
def overlay_heatmap(image, heatmap, colormap=plt.cm.autumn):
    heatmap_resized = np.interp(heatmap, (heatmap.min(), heatmap.max()), (0, 1))
    heatmap_resized = colormap(heatmap_resized)
    heatmap_resized = Image.fromarray((heatmap_resized[:, :, :3] * 255).astype(np.uint8))
    heatmap_resized = heatmap_resized.resize(image.size, Image.LANCZOS)
    overlayed_image = Image.blend(image, heatmap_resized, alpha=0.5)
    return overlayed_image

# Function to process an image and save XAI result
def process_image(image_path, model, save_dir, mask_size=60):
    original_image = Image.open(image_path).convert('RGB')
    heatmap = feature_ablation(model, original_image, mask_size)
    overlayed_image = overlay_heatmap(original_image, heatmap, colormap=plt.cm.BuPu)

    # Save the XAI result
    image_name = os.path.basename(image_path)
    save_path = os.path.join(save_dir, image_name.replace('.png', '_xai.png'))
    overlayed_image.save(save_path)

# Directory containing images
input_dir = 'F:/nsclc/Test3.v1/shap/'
# Directory to save XAI results
output_dir = 'F:/nsclc/Test3.v1/shap_xai/'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate over each image in the input directory
image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
for image_file in tqdm(image_files, desc="Processing Images"):
    image_path = os.path.join(input_dir, image_file)
    process_image(image_path, model, output_dir)



######################### working ablation map (single image)
# import torch
# import numpy as np
# from PIL import Image, ImageDraw
# import torchvision.transforms as transforms
# from ultralytics import YOLO
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

# yolo_model_path = 'F:/src/runs/classify/train7/weights/best.pt'
# image_path = 'F:/nsclc/Test3.v1/shap/mal_16_pet.png'

# # Load the YOLOv8 model
# model = YOLO(yolo_model_path)

# # Function to preprocess an image object
# def preprocess_image(image, size=(640, 640)):
#     transform = transforms.Compose([
#         transforms.Resize(size),
#         transforms.ToTensor()
#     ])
#     return transform(image).unsqueeze(0)

# # Function for Feature Ablation
# def feature_ablation(model, image, mask_size=60):
#     original_tensor = preprocess_image(image)
#     num_rows = (image.height + mask_size - 1) // mask_size
#     num_cols = (image.width + mask_size - 1) // mask_size
#     heatmap = np.zeros((num_rows, num_cols))

#     # Perform inference on the original image
#     with torch.no_grad():
#         original_pred = model(original_tensor)
#     original_confidence = original_pred[0].probs.top1conf.item()

#     for i in range(0, image.width, mask_size):
#         for j in range(0, image.height, mask_size):
#             ablated_image = image.copy()
#             draw = ImageDraw.Draw(ablated_image)
#             draw.rectangle([i, j, i + mask_size, j + mask_size], fill="black")
            
#             ablated_tensor = preprocess_image(ablated_image)
#             with torch.no_grad():
#                 ablated_pred = model(ablated_tensor)
#             ablated_confidence = ablated_pred[0].probs.top1conf.item()

#             diff = np.abs(original_confidence - ablated_confidence)
#             heatmap[j // mask_size, i // mask_size] = diff

#     return heatmap

# # Function to overlay the heatmap on the image
# def overlay_heatmap(image, heatmap, colormap=plt.cm.autumn):
#     heatmap_resized = np.interp(heatmap, (heatmap.min(), heatmap.max()), (0, 1))
#     heatmap_resized = colormap(heatmap_resized)
#     heatmap_resized = Image.fromarray((heatmap_resized[:, :, :3] * 255).astype(np.uint8))
#     heatmap_resized = heatmap_resized.resize(image.size, Image.LANCZOS)
#     overlayed_image = Image.blend(image, heatmap_resized, alpha=0.5)
#     return overlayed_image


# # Load the original image
# original_image = Image.open(image_path).convert('RGB')

# # Perform Feature Ablation and Overlay Heatmap
# heatmap = feature_ablation(model, original_image)
# overlayed_image = overlay_heatmap(original_image, heatmap, colormap=plt.cm.BuPu)

# # Display or save the result
# overlayed_image.show()  # Or overlayed_image.save('result.png')



######################### GRAD CAM - needs underlying YOLO model
# import torch
# import torch.nn.functional as F
# import numpy as np
# from PIL import Image
# import torchvision.transforms as transforms
# from torchvision import models

# # Load the model
# model = models.resnet50(pretrained=True)
# model.eval()

# # Load and preprocess the image
# def preprocess_image(img_path, size=(224, 224)):
#     img = Image.open(img_path).convert('RGB')
#     transform = transforms.Compose([
#         transforms.Resize(size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     return transform(img).unsqueeze(0)

# # Grad-CAM
# def grad_cam(img_path, model, target_layer):
#     img_tensor = preprocess_image(img_path)

#     # Hook for the gradients and feature maps
#     gradients = []
#     def save_gradient(*args):
#         grad_input = args[1]
#         gradients.append(grad_input[0])

#     feature_maps = None
#     for name, module in model.named_children():
#         if name == target_layer:
#             module.register_forward_hook(lambda m, i, o: setattr(feature_maps, 'data', o))
#             module.register_backward_hook(save_gradient)
#             break

#     # Forward pass
#     output = model(img_tensor)
#     target_class = output.argmax().item()

#     # Backward pass
#     model.zero_grad()
#     class_loss = F.softmax(output, dim=1)[0, target_class]
#     class_loss.backward()

#     # Weighted feature map
#     weighted_feature_map = torch.mean(gradients[0], dim=[0, 2, 3])

#     # Generate heatmap
#     heatmap = np.maximum(feature_maps.data.squeeze(0).detach().numpy().dot(weighted_feature_map.detach().numpy()), 0)
#     heatmap = heatmap / np.max(heatmap)
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = Image.fromarray(heatmap).resize((img_tensor.shape[2], img_tensor.shape[3]))

#     # Superimpose heatmap onto original image
#     original_img = Image.open(img_path)
#     heatmap_img = Image.blend(original_img, heatmap, alpha=0.5)
#     return heatmap_img

# # Example usage
# image_path = 'F:/nsclc/Test3.v1/shap/mal_117_pet.png'
# grad_cam_image = grad_cam(image_path, model, "layer4")
# grad_cam_image.show()



######################### too complex for Kernel explainer
# import torch
# import shap
# from PIL import Image
# import torchvision.transforms as transforms
# from ultralytics import YOLO
# import numpy as np

# yolo_model_path = 'F:/src/runs/classify/train7/weights/best.pt'
# image_path = 'F:/nsclc/Test3.v1/shap/mal_117_pet.png'

# # Load the YOLOv8 model
# model = YOLO(yolo_model_path)

# # Function to preprocess the image
# def preprocess_image(image_path, size=(640, 640)):
#     image = Image.open(image_path).convert('RGB')
#     transform = transforms.Compose([
#         transforms.Resize(size),
#         transforms.ToTensor()
#     ])
#     return transform(image).unsqueeze(0)

# # Function to make predictions using the model
# def model_predict(data):
#     with torch.no_grad():
#         # Ensure the data is in the correct shape (BCHW format)
#         data = torch.tensor(data, dtype=torch.float32).view(-1, 3, 640, 640)
#         predictions = model(data)
#         # Simplify the output to just the top class probability
#         return np.array([pred.probs.top1conf.cpu().numpy() for pred in predictions])

# # Load and preprocess the image
# input_tensor = preprocess_image(image_path)
# input_array = input_tensor.squeeze().numpy()

# # Flatten the input array for SHAP
# input_array_flat = input_array.reshape(1, -1)

# # Create a background dataset
# background = np.zeros_like(input_array_flat)

# # Initialize SHAP KernelExplainer
# explainer = shap.KernelExplainer(model_predict, background)

# # Compute SHAP values for the input image
# shap_values = explainer.shap_values(input_array_flat, nsamples=100)

# # Reshape SHAP values to the original image dimensions
# shap_values_reshaped = shap_values[0].reshape(input_array.shape)

# # Visualize the SHAP values as an overlay on the original image
# original_image = Image.open(image_path)
# shap_overlay = Image.fromarray((shap_values_reshaped * 255).astype(np.uint8))
# shap_overlay = shap_overlay.resize(original_image.size, Image.LANCZOS)

# overlayed_image = Image.blend(original_image, shap_overlay, alpha=0.5)
# overlayed_image.show()



######################### not feasible to generate a saliency map using gradient backpropagation.
# import torch
# from PIL import Image
# import torchvision.transforms as transforms
# from ultralytics import YOLO
# import numpy as np
# import matplotlib.pyplot as plt

# yolo_model_path = 'F:/src/runs/classify/train7/weights/best.pt'
# image_path = 'F:/nsclc/Test3.v1/shap/mal_117_pet.png'

# # Load the YOLOv8 model using ultralytics YOLO class
# model = YOLO(yolo_model_path)

# # Function to preprocess the image
# def preprocess_image(image_path, size=(320, 320)):
#     image = Image.open(image_path).convert('RGB')
#     transform = transforms.Compose([
#         transforms.Resize(size),
#         transforms.ToTensor()
#     ])
#     return transform(image).unsqueeze(0)

# # Load and preprocess the image
# input_tensor = preprocess_image(image_path)
# input_tensor.requires_grad_(True)

# # Perform inference
# output = model(input_tensor)

# # Extract class probabilities from the output
# # Assuming the output has a 'probs' attribute with class probabilities
# class_probabilities = output[0].probs.data
# target_class_prob = class_probabilities[0]  # Assuming interest in the first class

# # Use the target class probability for backward pass
# target_class_prob.backward()

# # Extract gradients
# gradients = input_tensor.grad.data

# # Convert gradients to numpy for visualization
# gradients = gradients.squeeze().cpu().numpy()

# # Convert to positive values and normalize
# gradients = np.maximum(gradients, 0)
# gradients /= np.max(gradients)

# # Resize the saliency map to the original image size
# original_image = Image.open(image_path)
# saliency_map = Image.fromarray((gradients * 255).astype(np.uint8))
# saliency_map = saliency_map.resize(original_image.size, Image.LANCZOS)

# # Apply a colormap for better visualization
# saliency_map = plt.cm.jet(np.array(saliency_map) / 255.0)
# saliency_map = Image.fromarray((saliency_map[:, :, :3] * 255).astype(np.uint8))

# # Overlay the saliency map on the original image
# overlayed_image = Image.blend(original_image, saliency_map, alpha=0.5)

# # Display or save the result
# overlayed_image.show()  # Or overlayed_image.save('saliency_overlay.png')
