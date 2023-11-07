import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image

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

    # Load pre-trained MobileNetV2 model
    model = models.mobilenet_v2(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))

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
    torch.save(model.state_dict(), 'mobilenet_model.pth')
    print('Model saved.')

    return model, class_names

if __name__ == '__main__':
    # Specify the path to the folder containing images
    ct = f"F:/nsclc/Test3.v1/ct"
    pet = f"F:/nsclc/Test3.v1/pet"

    # Load custom dataset (replace 'data_directory' with your dataset directory)
    data_directory = ct
    trained_model, class_labels = train_model(data_directory)

    # # Example usage of the trained model for classification
    # image_path = input('Enter the path to the image file for classification: ')
    # image = Image.open(image_path).convert('RGB')
    # transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # input_tensor = transform(image).unsqueeze(0)
    # input_tensor = input_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # with torch.no_grad():
    #     output = trained_model(input_tensor)
    #     _, predicted_idx = torch.max(output, 1)

    # predicted_class = class_labels[predicted_idx.item()]
    # print(f'Predicted class: {predicted_class}')
