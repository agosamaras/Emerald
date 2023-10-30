# import torch
# import torchvision.transforms as transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader
# import torch.optim as optim
# import torch.nn as nn
# import timm
# from transformers import DeiTConfig, DeiTModel

# # Set the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define the data transforms
# transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# # Set the path to your image dataset
# ct = f"F:/nsclc/Test1.v1i.folder"
# pet = f"F:/nsclc/Test2.v1i.folder"

# root_fs = pet
# data_path = root_fs

# # Create the dataset
# dataset = ImageFolder(data_path, transform=transform)

# # Define the data loader
# batch_size = 32
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # Load the DeiT model
# configuration = DeiTConfig()
# model = DeiTModel(configuration)
# model = timm.create_model("deit_base_patch16_224", pretrained=False, num_classes=len(dataset.classes)).to(device)

# # Define the loss function
# criterion = nn.CrossEntropyLoss()

# # Define the optimizer
# optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# # Set the number of training epochs
# num_epochs = 10

# # Train the model
# model.train()

# for epoch in range(num_epochs):
#     running_loss = 0.0
#     for i, (images, labels) in enumerate(dataloader):
#         images = images.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()

#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         # Backward pass and optimize
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#         # Print training progress
#         if (i + 1) % 10 == 0:
#             print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}")
#             running_loss = 0.0

# print("Training finished.")


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import timm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the path to your image dataset
# ct = f"F:/nsclc/Test1.v1i.folder"
# pet = f"F:/nsclc/Test2.v1i.folder"
ct = f"F:/nsclc/Test3.v1/ct"
pet = f"F:/nsclc/Test3.v1/pet"

root_fs = pet
data_path = root_fs

# Create the dataset
dataset = ImageFolder(data_path)

# Define model architecture
# model = models.vit_deit_base_patch16_224(pretrained=False, num_classes=len(dataset.classes))
model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(dataset.classes))
model.to(device)

# Define training parameters
batch_size = 16
lr = 0.001
num_epochs = 50

# Load and preprocess dataset
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to (224, 224)
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to (224, 224)
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(f"{root_fs}/train", transform=train_transforms)
test_dataset = datasets.ImageFolder(f"{root_fs}/test", transform=test_transforms)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct_predictions / len(train_dataset) * 100

    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_acc:.2f}%")

# Testing loop
best_acc = 0.5
model.eval()
test_correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()

test_acc = test_correct / len(test_dataset) * 100
print(f"Test Accuracy: {test_acc:.2f}%")

#---
if test_acc > best_acc:
    best_acc = test_acc
    best_model = model.state_dict()
    torch.save(best_model, "deit_best_model.pt")

# Load the best model
best_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(dataset.classes))
best_model.load_state_dict(torch.load("deit_best_model.pt"))  # Assuming you saved the best model during training
best_model.to(device)
best_model.eval()

# Create data loader for the validation dataset
val_dataset = datasets.ImageFolder(f"{root_fs}/valid", transform=test_transforms)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Validation loop
val_correct = 0
val_tp, val_tn, val_fp, val_fn = 0, 0, 0, 0

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = best_model(images)
        _, predicted = torch.max(outputs, 1)

        val_correct += (predicted == labels).sum().item()

        # Calculate TP, TN, FP, FN
        val_tp += ((predicted == 1) & (labels == 1)).sum().item()
        val_tn += ((predicted == 0) & (labels == 0)).sum().item()
        val_fp += ((predicted == 1) & (labels == 0)).sum().item()
        val_fn += ((predicted == 0) & (labels == 1)).sum().item()

val_acc = val_correct / len(val_dataset) * 100
print(f"Validation Accuracy: {val_acc:.2f}%")
print(f"TP: {val_tp}, TN: {val_tn}, FP: {val_fp}, FN: {val_fn}")

