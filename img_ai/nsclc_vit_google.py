# import torchvision
import torchvision
from torchvision.transforms import ToTensor
from transformers import ViTModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import ViTFeatureExtractor
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# define dataset
root_fs = f"F:/nsclc/Test3.v1/pet"
# train_ds = torchvision.datasets.ImageFolder(f'{root_fs}/train/', transform=ToTensor())
train_ds = torchvision.datasets.ImageFolder(f'{root_fs}/train/', transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),  # Resize images to 224x224
        torchvision.transforms.ToTensor()
    ]))
# valid_ds = torchvision.datasets.ImageFolder(f'{root_fs}/valid/', transform=ToTensor())
valid_ds = torchvision.datasets.ImageFolder(f'{root_fs}/valid/', transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),  # Resize images to 224x224
        torchvision.transforms.ToTensor()
    ]))
# test_ds = torchvision.datasets.ImageFolder(f'{root_fs}/test/', transform=ToTensor())
test_ds = torchvision.datasets.ImageFolder(f'{root_fs}/test/', transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),  # Resize images to 224x224
        torchvision.transforms.ToTensor()
    ]))

# define Model
class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels=2):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels):
        outputs = self.vit(pixel_values=pixel_values)
        output = self.dropout(outputs.last_hidden_state[:,0])
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if loss is not None:
            return logits, loss.item()
        else:
            return logits, None


if __name__ == '__main__':
    # define Model parameters
    EPOCHS = 50
    BATCH_SIZE = 10 #8 #10 #duplicated 2 img from train/malignant (500 & 501) - both ct and pet
    LEARNING_RATE = 2e-5
    # Define Model
    model = ViTForImageClassification(len(train_ds.classes))
    # Feature Extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    # Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Cross Entropy Loss
    loss_func = nn.CrossEntropyLoss()
    # Use GPU if available  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    if torch.cuda.is_available():
        model.cuda()


    # train model
    print("Number of train samples: ", len(train_ds))
    print("Number of test samples: ", len(test_ds))
    print("Detected Classes are: ", train_ds.class_to_idx) 
    train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    test_loader  = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    best_accuracy = 0.0  # Variable to keep track of the best test accuracy
    best_test_results = None  # Variable to store the test results of the best overall run

    # Train the model
    for epoch in range(EPOCHS):        
        for step, (x, y) in enumerate(train_loader):
            # Change input array into list with each batch being one element
            x = np.split(np.squeeze(np.array(x)), BATCH_SIZE)
            # Remove unecessary dimension
            for index, array in enumerate(x):
                x[index] = np.squeeze(array)
            # Apply feature extractor, stack back into 1 tensor and then convert to tensor
            x = torch.tensor(np.stack(feature_extractor(x)['pixel_values'], axis=0))
            # Send to GPU if available
            x, y  = x.to(device), y.to(device)
            b_x = Variable(x)   # batch x (image)
            b_y = Variable(y)   # batch y (target)
            # Feed through model
            output, loss = model(b_x, None)
            # Calculate loss
            if loss is None: 
                loss = loss_func(output, b_y)   
                optimizer.zero_grad()           
                loss.backward()                 
                optimizer.step()

            if step % 50 == 0:
                # Get the next batch for testing purposes
                test = next(iter(test_loader))
                test_x = test[0]
                # Reshape and get feature matrices as needed
                test_x = np.split(np.squeeze(np.array(test_x)), BATCH_SIZE)
                for index, array in enumerate(test_x):
                    test_x[index] = np.squeeze(array)
                test_x = torch.tensor(np.stack(feature_extractor(test_x)['pixel_values'], axis=0))
                # Send to appropirate computing device
                test_x = test_x.to(device)
                test_y = test[1].to(device)
                # Get output (+ respective class) and compare to target
                test_output, loss = model(test_x, test_y)
                test_output = test_output.argmax(1)
                # Calculate Accuracy
                accuracy = (test_output == test_y).sum().item() / BATCH_SIZE
                print('Epoch: ', epoch, '| train loss: %.4f' % loss, '| test accuracy: %.2f' % accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # Get the predictions for the best overall run
                test_predictions = []
                test_targets = []
                with torch.no_grad():
                    for test_step, (test_x, test_y) in enumerate(test_loader):
                        test_x = test_x.to(device)
                        test_y = test_y.to(device)
                        test_output, _ = model(test_x, test_y)
                        test_predictions.extend(test_output.argmax(1).cpu().numpy())
                        test_targets.extend(test_y.cpu().numpy())

    # Convert lists to NumPy arrays
    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)

    # Calculate confusion matrix for the best overall run
    conf_matrix = confusion_matrix(test_targets, test_predictions)

    # Extract the TP, TN, FP, FN values
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]

    # Print the results
    print("Best Test Results (Overall Run):")
    print("True Positives (TP):", TP)
    print("True Negatives (TN):", TN)
    print("False Positives (FP):", FP)
    print("False Negatives (FN):", FN)


    # Evaluate on the validation images
    valid_loader = data.DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=4)
    model.eval()  # Set the model to evaluation mode

    # Variables to store the predictions and targets for the validation run
    val_predictions = []
    val_targets = []

    # Disable grad for inference
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Generate prediction
            predictions, _ = model(inputs, targets)

            # Append the predictions and targets to their respective lists
            val_predictions.extend(predictions.argmax(1).cpu().numpy())
            val_targets.extend(targets.cpu().numpy())

    # Convert lists to NumPy arrays
    val_predictions = np.array(val_predictions)
    val_targets = np.array(val_targets)

    # Calculate confusion matrix for the validation run
    conf_matrix_val = confusion_matrix(val_targets, val_predictions)

    # Extract the TP, TN, FP, FN values for the validation run
    TP_val = conf_matrix_val[1, 1]
    TN_val = conf_matrix_val[0, 0]
    FP_val = conf_matrix_val[0, 1]
    FN_val = conf_matrix_val[1, 0]

    # Print the results for the validation run
    print("Validation Results:")
    print("True Positives (TP):", TP_val)
    print("True Negatives (TN):", TN_val)
    print("False Positives (FP):", FP_val)
    print("False Negatives (FN):", FN_val)


    # # evaluate on a test image
    # EVAL_BATCH = 1
    # eval_loader  = data.DataLoader(valid_ds, batch_size=EVAL_BATCH, shuffle=True, num_workers=4) 
    # # Disable grad
    # with torch.no_grad():
    #     inputs, target = next(iter(eval_loader))
    #     # Reshape and get feature matrices as needed
    #     print(inputs.shape)
    #     inputs = inputs[0].permute(1, 2, 0)
    #     # Save original Input
    #     originalInput = inputs
    #     for index, array in enumerate(inputs):
    #         inputs[index] = np.squeeze(array)
    #     inputs = torch.tensor(np.stack(feature_extractor(inputs)['pixel_values'], axis=0))

    #     # Send to appropriate computing device
    #     inputs = inputs.to(device)
    #     target = target.to(device)

    #     # Generate prediction
    #     prediction, loss = model(inputs, target)

    #     # Predicted class value using argmax
    #     predicted_class = np.argmax(prediction.cpu())
    #     value_predicted = list(valid_ds.class_to_idx.keys())[list(valid_ds.class_to_idx.values()).index(predicted_class)]
    #     value_target = list(valid_ds.class_to_idx.keys())[list(valid_ds.class_to_idx.values()).index(target)]
            
    #     # Show result
    #     plt.imshow(originalInput)
    #     plt.xlim(224,0)
    #     plt.ylim(224,0)
    #     plt.title(f'Prediction: {value_predicted} - Actual target: {value_target}')
    #     plt.show()


    # save the model
    torch.save(model, f'{root_fs}/model.pt')
