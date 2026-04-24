import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import warnings
import shap

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SafeResNet18(nn.Module):
    """Wrapper for ResNet18 that prevents in-place operations"""
    def __init__(self, num_classes):
        super().__init__()
        original_model = models.resnet18(pretrained=True)
        
        # Replace all ReLU layers with out-of-place versions
        def replace_relu(module):
            for name, child in module.named_children():
                if isinstance(child, nn.ReLU):
                    setattr(module, name, nn.ReLU(inplace=False))
                elif len(list(child.children())) > 0:
                    replace_relu(child)
        
        replace_relu(original_model)
        
        num_ftrs = original_model.fc.in_features
        original_model.fc = nn.Linear(num_ftrs, num_classes)
        self.model = original_model
        
    def forward(self, x):
        # Ensure no in-place operations
        x = x.clone()
        return self.model(x)

def train_model(data_dir, num_epochs=10):
    # Define data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    # Set data directories
    image_datasets = {x: ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'valid']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes

    # Load pre-trained ResNet model with our safe wrapper
    model = SafeResNet18(len(class_names))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    for epoch in range(num_epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = corrects.double() / dataset_sizes[phase]

            print(f'Epoch {epoch + 1}/{num_epochs} | {phase} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'resnet_model.pth')
    print('Model saved.')

    return model, class_names

def get_image_transform():
    """Return the transform used for validation images"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_pil_transform(): 
    """Convert tensor to PIL image for LIME"""
    return transforms.Compose([
        transforms.Lambda(lambda x: x.cpu().detach().numpy()),
        transforms.Lambda(lambda x: np.transpose(x, (1, 2, 0))),
        transforms.Lambda(lambda x: (x * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255),
        transforms.Lambda(lambda x: x.astype(np.uint8)),
        transforms.ToPILImage(),
    ])

def get_preprocess_transform():
    """Convert PIL image to tensor for LIME"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def explain_with_shap(model, input_tensor, class_names):
    """Explain the model's prediction using SHAP with proper tensor handling"""
    try:
        # Create a function that handles the model prediction correctly
        def predict(img: np.ndarray) -> np.ndarray:
            img = torch.tensor(img, dtype=torch.float32).to(input_tensor.device)
            img = img.permute(0, 3, 1, 2)  # NHWC to NCHW
            with torch.no_grad():
                output = model(img)
            return output.cpu().numpy()
        
        # Create explainer with the predict function
        explainer = shap.Explainer(
            predict, 
            masker=shap.maskers.Image("inpaint_telea", input_tensor[0].shape),
            output_names=class_names
        )
        
        # Prepare input for SHAP (convert to numpy and channel-last format)
        input_np = input_tensor.cpu().numpy().transpose(0, 2, 3, 1)
        
        # Calculate SHAP values
        shap_values = explainer(input_np, max_evals=500)
        
        # Plot the explanations
        shap.image_plot(
            shap_values.values, 
            -input_np,
            labels=shap_values.output_names
        )
        
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        # Fallback visualization
        plt.imshow(input_tensor[0].cpu().permute(1, 2, 0).numpy())
        plt.title(f"Predicted: {class_names[predicted_idx.item()]}")
        plt.axis('off')
        plt.show()

def explain_with_lime(model, image_tensor, class_names):
    """Explain the model's prediction using LIME"""
    try:
        # Convert tensor to PIL image
        pil_transform = get_pil_transform()
        image = pil_transform(image_tensor.squeeze(0))
        
        # Create batch predict function with model access
        def batch_predict(images):
            model.eval()
            batch = torch.stack(tuple(get_preprocess_transform()(i) for i in images), dim=0)
            device = image_tensor.device
            batch = batch.to(device)
            
            with torch.no_grad():
                outputs = model(batch)
            return outputs.detach().cpu().numpy()
        
        # Create LIME explainer
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            np.array(image), 
            batch_predict, 
            top_labels=len(class_names), 
            hide_color=0, 
            num_samples=500
        )
        
        # Show explanation for the top class
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], 
            positive_only=True, 
            num_features=5, 
            hide_rest=False
        )
        
        # Display the explanation
        plt.imshow(mark_boundaries(temp / 255.0, mask))
        plt.title(f'LIME Explanation for {class_names[explanation.top_labels[0]]}')
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"LIME explanation failed: {e}")

if __name__ == '__main__':
    # Specify the path to the folder containing images
    ct = "Test3.v1/ct"
    pet = "Test3.v1/pet"

    # Load custom dataset
    data_directory = ct
    trained_model, class_labels = train_model(data_directory)

    # Example usage of the trained model for classification
    # for i in ["16", "125", "132", "183", "417"]: #pet
    for i in ["109", "182", "301", "390", "408", "453"]: #ct
        # image_path = f"{data_directory}/valid/malignant/{i}_pet.png"
        image_path = f"{data_directory}/valid/malignant/{i}_ct.png"
        image = Image.open(image_path).convert('RGB')
        
        # Transform and prepare the image
        transform = get_image_transform()
        input_tensor = transform(image).unsqueeze(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)
        trained_model = trained_model.to(device)

        # Make prediction
        with torch.no_grad():
            output = trained_model(input_tensor)
            _, predicted_idx = torch.max(output, 1)

        predicted_class = class_labels[predicted_idx.item()]
        print(f'Predicted class: {predicted_class}')

        # Explain with SHAP
        print("\nGenerating SHAP explanation...")
        explain_with_shap(trained_model, input_tensor, class_labels)

        # Explain with LIME
        print("\nGenerating LIME explanation...")
        explain_with_lime(trained_model, input_tensor, class_labels)

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms, models
# from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder
# from PIL import Image

# def train_model(data_dir, num_epochs=10):
#     # Define data transforms
#     data_transforms = {
#         'train': transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ]),
#         'valid': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ]),
#     }

#     # Set data directories
#     image_datasets = {x: ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
#     dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'valid']}
#     dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
#     class_names = image_datasets['train'].classes

#     # Load pre-trained ResNet model
#     model = models.resnet18(pretrained=True)
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, len(class_names))

#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)

#     # Define loss function and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#     # Training loop
#     for epoch in range(num_epochs):
#         for phase in ['train', 'valid']:
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()

#             running_loss = 0.0
#             corrects = 0

#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 optimizer.zero_grad()

#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     loss = criterion(outputs, labels)

#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 running_loss += loss.item() * inputs.size(0)
#                 _, preds = torch.max(outputs, 1)
#                 corrects += torch.sum(preds == labels.data)

#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = corrects.double() / dataset_sizes[phase]

#             print(f'Epoch {epoch + 1}/{num_epochs} | {phase} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

#     # Save the trained model
#     torch.save(model.state_dict(), 'resnet_model.pth')
#     print('Model saved.')

#     return model, class_names

# if __name__ == '__main__':
#     # Specify the path to the folder containing images
#     ct = f"F:/nsclc/Test3.v1/ct"
#     pet = f"F:/nsclc/Test3.v1/pet"

#     # Load custom dataset (replace 'data_directory' with your dataset directory)
#     data_directory = ct
#     # data_directory = input("Enter the path to the folder containing images: ")
#     trained_model, class_labels = train_model(data_directory)

#     # Example usage of the trained model for classification
#     image_path = f"{data_directory}/ct/valid/benign/12_ct.png"
#     image = Image.open(image_path).convert('RGB')
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     input_tensor = transform(image).unsqueeze(0)
#     input_tensor = input_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

#     with torch.no_grad():
#         output = trained_model(input_tensor)
#         _, predicted_idx = torch.max(output, 1)

#     predicted_class = class_labels[predicted_idx.item()]
#     print(f'Predicted class: {predicted_class}')
