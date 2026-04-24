import torch
import torch.nn as nn
from torch.optim import SGD
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries
import warnings
warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Define dataset path
data_dir = 'Test3.v1/pet_synth'  # Change to your path
image_datasets = {x: datasets.ImageFolder(root=f'{data_dir}/{x}', transform=data_transforms[x])
                  for x in ['train', 'test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
               for x in ['train', 'test']}

# Load and customize model
model = models.vgg16(pretrained=True)
num_classes = len(image_datasets['train'].classes)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training function
def train_model(model, criterion, optimizer, dataloaders, num_epochs=10):
    for epoch in range(num_epochs):
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_accuracy = accuracy_score(all_labels, all_preds)

            print(f'Epoch {epoch+1}/{num_epochs} [{phase}] Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f}')

    torch.save(model.state_dict(), 'vgg16_custom_dataset.pth')

# Classification function
def classify_image(image_path, model, transform):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)[0] * 100
    predicted_class_idx = torch.argmax(output).item()
    predicted_class = image_datasets['train'].classes[predicted_class_idx]
    confidence = probabilities[predicted_class_idx].item()
    return predicted_class, confidence

# SHAP explanation
def explain_with_shap(image_path, model, transform):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Use GradientExplainer for CNNs
    background = image_tensor.clone()
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(image_tensor)

    # Convert tensor to image
    input_img = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    input_img = np.clip(input_img * np.array([0.229, 0.224, 0.225]) + 
                        np.array([0.485, 0.456, 0.406]), 0, 1)

    shap.image_plot(shap_values, np.array([input_img]))

# LIME explanation
def explain_with_lime(image_path, model, transform):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    def batch_predict(images):
        model.eval()
        batch = torch.stack([transform(Image.fromarray(img)).to(device) for img in images], dim=0)
        with torch.no_grad():
            logits = model(batch)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs.cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image_np, batch_predict, top_labels=1, hide_color=0, num_samples=1000)

    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(top_label, positive_only=True, num_features=10, hide_rest=False)
    img_boundry = mark_boundaries(temp / 255.0, mask)

    plt.figure(figsize=(8, 8))
    plt.title(f'LIME: {image_datasets["train"].classes[top_label]}')
    plt.imshow(img_boundry)
    plt.axis('off')
    plt.show()

# Main entry
if __name__ == '__main__':
    # Train model (if not trained)
    train_model(model, criterion, optimizer, dataloaders, num_epochs=10)

    # Load model for inference
    model.load_state_dict(torch.load('vgg16_custom_dataset.pth'))
    model.eval()

    # Path to the image to classify and explain
    ct = "Test3.v1/ct"
    pet = "Test3.v1/pet"

    # Load custom dataset
    data_directory = pet
    image_path = f"{data_directory}/valid/malignant/9_pet.png"  # Replace with the path to the test image

    predicted_class, confidence = classify_image(image_path, model, data_transforms['test'])
    print(f'Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%')

    # # Run explanations
    # explain_with_shap(image_path, model, data_transforms['test'])   # SHAP
    # explain_with_lime(image_path, model, data_transforms['test'])   # LIME


# import torch
# import torch.nn as nn
# from torch.optim import SGD
# from torchvision import datasets, models, transforms
# from torch.utils.data import DataLoader
# from sklearn.metrics import accuracy_score
# from PIL import Image

# # Set device (GPU if available, else CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define data transformations for training and testing
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'test': transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }

# ct = f"Test3.v1/ct"
# pet = f"Test3.v1/pet"

# # Load custom dataset (replace 'data_dir' with your dataset directory)
# data_dir = pet
# image_datasets = {x: datasets.ImageFolder(root=f'{data_dir}/{x}', transform=data_transforms[x])
#                   for x in ['train', 'test']}
# dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
#                for x in ['train', 'test']}

# # Use a pre-trained VGG16 model and modify the final fully connected layer for the number of classes in your dataset
# # weights=vgg16.VGG16_Weights.DEFAULT
# model = models.vgg16(pretrained=True)
# num_classes = len(image_datasets['train'].classes)
# model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)  # Modify the last FC layer

# # Send the model to device
# model = model.to(device)

# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

# def train_model(model, criterion, optimizer, dataloaders, num_epochs=10):
# # Training loop
#     # num_epochs = 10
#     for epoch in range(num_epochs):
#         for phase in ['train', 'test']:
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()

#             running_loss = 0.0
#             all_preds = []
#             all_labels = []

#             for inputs, labels in dataloaders[phase]:
#                 inputs, labels = inputs.to(device), labels.to(device)

#                 # Zero the gradients
#                 optimizer.zero_grad()

#                 # Forward pass
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     loss = criterion(outputs, labels)

#                     # Backpropagation and optimization only in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 running_loss += loss.item() * inputs.size(0)
#                 _, preds = torch.max(outputs, 1)
#                 all_preds.extend(preds.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())

#             epoch_loss = running_loss / len(image_datasets[phase])
#             epoch_accuracy = accuracy_score(all_labels, all_preds)

#             print(f'Epoch {epoch + 1}/{num_epochs} [{phase}] Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f}')

#     # Save the trained model
#     torch.save(model.state_dict(), 'vgg16_custom_dataset.pth')

# # # Load the trained model for inference
# # model.load_state_dict(torch.load('vgg16_custom_dataset.pth'))
# # model.eval()

# # Function to classify a new image using the trained model
# def classify_image(image_path, model, transform):
#     image = Image.open(image_path).convert("RGB")
#     image_tensor = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         output = model(image_tensor)
#     probabilities = torch.softmax(output, dim=1)[0] * 100
#     predicted_class_idx = torch.argmax(output).item()
#     predicted_class = image_datasets['train'].classes[predicted_class_idx]
#     confidence = probabilities[predicted_class_idx].item()
#     return predicted_class, confidence

# # # Example usage of the classify_image function
# # image_path = 'path/to/your/test/image.jpg'  # Replace with the path to the test image
# # predicted_class, confidence = classify_image(image_path, model, data_transforms['test'])
# # print(f'Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%')


# if __name__ == '__main__':
#     # Define your data transformations, model, and other necessary components here

#     # Check if CUDA (GPU) is available and set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Rest of your code (data loading, model training, etc.)

#     # Training loop
#     train_model(model, criterion, optimizer, dataloaders, num_epochs=10)

#     # Load the trained model for inference
#     model.load_state_dict(torch.load('vgg16_custom_dataset.pth'))
#     model.eval()

#     # Example usage of the classify_image function
#     # Specify the path to the folder containing images
#     ct = "Test3.v1/ct"
#     pet = "Test3.v1/pet"

#     # Load custom dataset
#     data_directory = ct
#     image_path = f"{data_directory}/valid/benign/12_ct.png"  # Replace with the path to the test image
#     predicted_class, confidence = classify_image(image_path, model, data_transforms['test'])
#     print(f'Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%')
