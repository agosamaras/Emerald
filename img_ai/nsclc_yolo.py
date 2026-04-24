import os
import torch
import shutil
import yaml
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

# Set environment variables to fix CUDA issues
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def convert_classification_to_yolo(classif_data_dir, yolo_data_dir, train_ratio=0.8):
    """
    Convert ImageFolder classification dataset to YOLO format
    """
    # Create YOLO directory structure
    for split in ['train', 'valid']:
        for folder in ['images', 'labels']:
            os.makedirs(os.path.join(yolo_data_dir, split, folder), exist_ok=True)
    
    # Get class names and mappings
    train_dir = os.path.join(classif_data_dir, 'train')
    class_names = os.listdir(train_dir)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    # Process each split
    for split in ['train', 'valid']:
        split_dir = os.path.join(classif_data_dir, split)
        
        if not os.path.exists(split_dir):
            continue
            
        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            class_id = class_to_id[class_name]
            
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Copy image
                    src_img = os.path.join(class_dir, img_file)
                    dst_img = os.path.join(yolo_data_dir, split, 'images', img_file)
                    shutil.copy2(src_img, dst_img)
                    
                    # Create YOLO format label file
                    label_file = os.path.splitext(img_file)[0] + '.txt'
                    label_path = os.path.join(yolo_data_dir, split, 'labels', label_file)
                    
                    # For classification conversion, we assume the entire image is the object
                    with open(label_path, 'w') as f:
                        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
    
    print(f"Dataset converted to YOLO format in: {yolo_data_dir}")
    print(f"Classes: {class_names}")
    return class_names

def create_dataset_yaml(data_dir, class_names):
    """
    Create YAML configuration file for YOLOv8 dataset
    """
    yaml_content = {
        'path': os.path.abspath(data_dir),
        'train': 'train/images',
        'val': 'valid/images',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    yaml_path = 'dataset_config.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Dataset YAML created: {yaml_path}")
    return yaml_path

def calculate_metrics(model, data_dir, class_names, conf_threshold=0.5):
    """
    Calculate accuracy, specificity, and sensitivity using the validation set
    """
    print("\n" + "="*60)
    print("CALCULATING METRICS ON VALIDATION SET")
    print("="*60)
    
    # Get validation images path
    val_images_dir = os.path.join(data_dir, 'valid', 'images')
    val_labels_dir = os.path.join(data_dir, 'valid', 'labels')
    
    if not os.path.exists(val_images_dir):
        print(f"Validation images directory not found: {val_images_dir}")
        return None, None, None, None
    
    # Get all validation images
    image_files = [f for f in os.listdir(val_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No validation images found!")
        return None, None, None, None
    
    true_labels = []
    predicted_labels = []
    predicted_confidences = []
    
    print(f"Processing {len(image_files)} validation images...")
    
    for i, image_file in enumerate(image_files):
        if i % 50 == 0:
            print(f"Processed {i}/{len(image_files)} images...")
        
        # Get true label from label file
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(val_labels_dir, label_file)
        
        true_label = None
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    true_label = int(first_line.split()[0])
        
        if true_label is None:
            continue
        
        # Run prediction
        image_path = os.path.join(val_images_dir, image_file)
        results = model.predict(image_path, conf=conf_threshold, verbose=False)
        
        # Get predicted label (take the highest confidence detection)
        pred_label = None
        max_conf = 0
        
        for result in results:
            if len(result.boxes) > 0:
                for box in result.boxes:
                    if box.conf.item() > max_conf:
                        max_conf = box.conf.item()
                        pred_label = int(box.cls.item())
        
        # If no detection or confidence below threshold, count as incorrect
        if pred_label is None:
            pred_label = -1  # Mark as no detection
        
        true_labels.append(true_label)
        predicted_labels.append(pred_label)
        predicted_confidences.append(max_conf if pred_label != -1 else 0)
    
    # Convert to numpy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    
    # Filter out cases where model didn't detect anything
    valid_predictions = predicted_labels != -1
    true_labels_valid = true_labels[valid_predictions]
    predicted_labels_valid = predicted_labels[valid_predictions]
    
    if len(true_labels_valid) == 0:
        print("No valid predictions found!")
        return None, None, None, None
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels_valid, predicted_labels_valid, labels=range(len(class_names)))
    
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Calculate metrics for each class
    accuracy = accuracy_score(true_labels_valid, predicted_labels_valid)
    
    # Calculate sensitivity (recall) and specificity for each class
    sensitivities = []
    specificities = []
    precisions = []
    
    for class_idx in range(len(class_names)):
        # True positives, false positives, false negatives, true negatives
        tp = cm[class_idx, class_idx]
        fn = np.sum(cm[class_idx, :]) - tp
        fp = np.sum(cm[:, class_idx]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        precisions.append(precision)
        
        print(f"\nClass: {class_names[class_idx]}")
        print(f"  Sensitivity (Recall): {sensitivity:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  F1-Score: {2 * (precision * sensitivity) / (precision + sensitivity):.4f}" if (precision + sensitivity) > 0 else "  F1-Score: 0.0000")
    
    # Overall metrics
    print(f"\n{'='*40}")
    print("OVERALL METRICS")
    print(f"{'='*40}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Average Sensitivity: {np.mean(sensitivities):.4f}")
    print(f"Macro Average Specificity: {np.mean(specificities):.4f}")
    print(f"Macro Average Precision: {np.mean(precisions):.4f}")
    
    # Detection rate
    detection_rate = np.sum(valid_predictions) / len(true_labels)
    print(f"Detection Rate: {detection_rate:.4f} ({np.sum(valid_predictions)}/{len(true_labels)})")
    
    return accuracy, np.mean(sensitivities), np.mean(specificities), cm

def train_yolov8_model(data_dir, class_names, num_epochs=10):
    """
    Train YOLOv8 model on custom dataset with metrics calculation
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Create dataset YAML file
    dataset_yaml = create_dataset_yaml(data_dir, class_names)
    
    # Training configuration
    train_args = {
        'data': dataset_yaml,
        'epochs': num_epochs,
        'imgsz': 640,
        'batch': 16,
        'device': str(device),
        'workers': 2,
        'patience': 10,
        'save': True,
        'exist_ok': True,
        'verbose': False,
        'plots': False,
        'amp': False,
        'cache': False,
    }
    
    # Train the model
    print("Starting YOLOv8 training...")
    try:
        results = model.train(**train_args)
        
        # Save the trained model
        model.save('yolov8_custom_model.pt')
        print('YOLOv8 model saved.')
        
        # Calculate and print metrics
        accuracy, sensitivity, specificity, cm = calculate_metrics(model, data_dir, class_names)
        
        # Print best results summary
        print("\n" + "="*60)
        print("BEST RUN SUMMARY")
        print("="*60)
        if accuracy is not None:
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Sensitivity: {sensitivity:.4f}")
            print(f"Specificity: {specificity:.4f}")
        else:
            print("Metrics calculation failed!")
        
        return model, results, (accuracy, sensitivity, specificity, cm)
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        print("Trying with CPU only...")
        
        # Fallback to CPU training
        train_args['device'] = 'cpu'
        results = model.train(**train_args)
        
        model.save('yolov8_custom_model.pt')
        print('YOLOv8 model saved (CPU training).')
        
        # Calculate and print metrics
        accuracy, sensitivity, specificity, cm = calculate_metrics(model, data_dir, class_names)
        
        return model, results, (accuracy, sensitivity, specificity, cm)

def train_yolov8_model_simple(data_dir, class_names, num_epochs=10):
    """
    Alternative simple training approach with metrics
    """
    # Force CPU if CUDA issues persist
    device = 'cpu'
    
    print(f"Using device: {device}")
    
    model = YOLO('yolov8n.pt')
    dataset_yaml = create_dataset_yaml(data_dir, class_names)
    
    # Minimal training configuration
    results = model.train(
        data=dataset_yaml,
        epochs=num_epochs,
        imgsz=640,
        batch=8,
        device=device,
        workers=0,
        patience=5,
        save=True,
        exist_ok=True,
        verbose=True
    )
    
    model.save('yolov8_custom_model.pt')
    print('YOLOv8 model saved.')
    
    # Calculate and print metrics
    accuracy, sensitivity, specificity, cm = calculate_metrics(model, data_dir, class_names)
    
    # Print best results summary
    print("\n" + "="*60)
    print("BEST RUN SUMMARY")
    print("="*60)
    if accuracy is not None:
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
    else:
        print("Metrics calculation failed!")
    
    return model, results, (accuracy, sensitivity, specificity, cm)

if __name__ == '__main__':
    # Install required packages if not already installed
    try:
        from ultralytics import YOLO
        import sklearn
    except ImportError:
        print("Installing required packages...")
        os.system("pip install ultralytics scikit-learn")
        from ultralytics import YOLO
        import sklearn
    
    # Specify the paths
    classification_data = "Test3.v1/ct"
    yolo_data = "Test3.v1/ct_yolo"
    
    # Step 1: Convert classification dataset to YOLO format
    print("Converting dataset to YOLO format...")
    class_names = convert_classification_to_yolo(classification_data, yolo_data)
    
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    # Try the simple training approach first
    trained_model, training_results, metrics = train_yolov8_model_simple(yolo_data, class_names, num_epochs=10)
    
    accuracy, sensitivity, specificity, confusion_matrix = metrics
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    if accuracy is not None:
        print(f"✅ Accuracy: {accuracy:.4f}")
        print(f"✅ Sensitivity: {sensitivity:.4f}")
        print(f"✅ Specificity: {specificity:.4f}")
        
        # Save metrics to file
        with open('training_metrics.txt', 'w') as f:
            f.write("YOLOv8 Training Results\n")
            f.write("=" * 30 + "\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Sensitivity: {sensitivity:.4f}\n")
            f.write(f"Specificity: {specificity:.4f}\n")
            f.write(f"Classes: {class_names}\n")
        
        print(f"\nMetrics saved to 'training_metrics.txt'")
    else:
        print("❌ Metrics calculation failed!")
    
    print(f"\nModel saved as 'yolov8_custom_model.pt'")