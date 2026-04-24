#DeepSeek
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only mode

import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries

print("Starting NSCLC classification script with specific explanation image")

# Configuration
DATA_DIR = 'pet_ct_images'
SPECIFIC_IMAGE = "C:\src\Emerald\pet_ct_images\malignant\301_ct.png"  # The specific image to use for explanations
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 3
NUM_CLASSES = 2

def create_minimal_dataset():
    """Create a very small dataset for training"""
    print("Creating minimal training dataset...")
    os.makedirs(os.path.join(DATA_DIR, 'benign'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'malignant'), exist_ok=True)
    
    # Create just 5 images per class for training
    for i in range(5):
        # Benign (uniform)
        benign_img = np.random.randint(100, 150, (128, 128, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(DATA_DIR, 'benign', f'benign_{i}.png'), benign_img)
        
        # Malignant (with spot)
        malignant_img = np.random.randint(100, 150, (128, 128, 3), dtype=np.uint8)
        cv2.circle(malignant_img, (64, 64), 15, (200, 150, 100), -1)
        cv2.imwrite(os.path.join(DATA_DIR, 'malignant', f'malignant_{i}.png'), malignant_img)
    print("Minimal training dataset created")

def load_specific_image():
    """Load the specific image for explanations"""
    print(f"Looking for specific image: {SPECIFIC_IMAGE}")
    
    if os.path.exists(SPECIFIC_IMAGE):
        print("Specific image found - loading...")
        img = cv2.imread(SPECIFIC_IMAGE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0  # Normalize
        return img
    else:
        print("Specific image not found - creating example...")
        # Create a sample malignant image
        img = np.random.randint(100, 150, (128, 128, 3), dtype=np.uint8)
        cv2.circle(img, (64, 64), 15, (200, 150, 100), -1)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0
        return img

def load_minimal_data():
    """Load training data with minimal augmentation"""
    print("Loading training data...")
    
    if not os.path.exists(DATA_DIR):
        print(f"Directory {DATA_DIR} not found - creating minimal data")
        create_minimal_dataset()
    
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.3
    )
    
    print("Creating data generators...")
    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_gen, val_gen

def build_tiny_model():
    """Build a very small CNN model"""
    print("Building tiny model...")
    
    model = Sequential([
        Conv2D(8, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def generate_explanations(model, image, filename_prefix=''):
    """Generate SHAP and LIME explanations for a specific image"""
    print(f"Generating explanations for {filename_prefix}image...")
    
    # SHAP Explanation
    try:
        print("Creating SHAP explanation...")
        background = np.zeros((1, *IMG_SIZE, 3))
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(np.expand_dims(image, axis=0))
        
        plt.figure(figsize=(8, 4))
        shap.image_plot(shap_values, -np.expand_dims(image, axis=0))
        plt.title(f'SHAP Explanation for {filename_prefix}image')
        plt.tight_layout()
        plt.savefig(f'shap_{filename_prefix}explanation.png', bbox_inches='tight', dpi=120)
        plt.close()
        print("SHAP explanation saved")
    except Exception as e:
        print(f"SHAP failed: {str(e)[:100]}...")

    # LIME Explanation
    try:
        print("Creating LIME explanation...")
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image.astype('double'),
            model.predict,
            top_labels=1,
            hide_color=0,
            num_samples=100
        )
        
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=3,
            hide_rest=False
        )
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f'Original {filename_prefix}Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(mark_boundaries(temp, mask))
        plt.title(f'LIME Explanation for {filename_prefix}Image')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'lime_{filename_prefix}explanation.png', bbox_inches='tight', dpi=120)
        plt.close()
        print("LIME explanation saved")
    except Exception as e:
        print(f"LIME failed: {str(e)[:100]}...")

def main():
    """Main execution with specific image handling"""
    try:
        # Load training data
        train_gen, val_gen = load_minimal_data()
        
        # Build and train model
        model = build_tiny_model()
        print("Model summary:")
        model.summary()
        
        print("Minimal training...")
        model.fit(
            train_gen,
            steps_per_epoch=2,
            epochs=EPOCHS,
            validation_data=val_gen,
            validation_steps=1
        )
        
        # Load specific image for explanations
        specific_img = load_specific_image()
        
        # Generate explanations for the specific image
        generate_explanations(model, specific_img, filename_prefix='specific_')
        
        # Also explain a random validation image for comparison
        val_img, _ = next(val_gen)
        generate_explanations(model, val_img[0], filename_prefix='random_')
        
        print("Script completed successfully")
        print(f"Explanation images saved as:")
        print(f"- shap_specific_explanation.png (your image)")
        print(f"- lime_specific_explanation.png (your image)")
        print(f"- shap_random_explanation.png (validation set example)")
        print(f"- lime_random_explanation.png (validation set example)")
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()


# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only mode

# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import shap
# from lime import lime_image
# from skimage.segmentation import mark_boundaries

# print("Starting lightweight NSCLC classification script")

# # Configuration
# DATA_DIR = 'pet_ct_images'
# IMG_SIZE = (128, 128)  # Reduced from 224 to save memory
# BATCH_SIZE = 8
# EPOCHS = 3
# NUM_CLASSES = 2

# def create_minimal_dataset():
#     """Create a very small dataset for testing"""
#     print("Creating minimal dataset...")
#     os.makedirs(os.path.join(DATA_DIR, 'benign'), exist_ok=True)
#     os.makedirs(os.path.join(DATA_DIR, 'malignant'), exist_ok=True)
    
#     # Create just 5 images per class
#     for i in range(5):
#         # Benign (uniform)
#         benign_img = np.random.randint(100, 150, (128, 128, 3), dtype=np.uint8)
#         cv2.imwrite(os.path.join(DATA_DIR, 'benign', f'benign_{i}.png'), benign_img)
        
#         # Malignant (with spot)
#         malignant_img = np.random.randint(100, 150, (128, 128, 3), dtype=np.uint8)
#         cv2.circle(malignant_img, (64, 64), 15, (200, 150, 100), -1)
#         cv2.imwrite(os.path.join(DATA_DIR, 'malignant', f'malignant_{i}.png'), malignant_img)
#     print("Minimal dataset created")

# def load_minimal_data():
#     """Load data with minimal augmentation"""
#     print("Loading minimal data...")
    
#     if not os.path.exists(DATA_DIR):
#         print(f"Directory {DATA_DIR} not found - creating minimal data")
#         create_minimal_dataset()
    
#     datagen = ImageDataGenerator(
#         rescale=1./255,
#         validation_split=0.3
#     )
    
#     print("Creating generators...")
#     train_gen = datagen.flow_from_directory(
#         DATA_DIR,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode='categorical',
#         subset='training'
#     )
    
#     val_gen = datagen.flow_from_directory(
#         DATA_DIR,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode='categorical',
#         subset='validation'
#     )
    
#     return train_gen, val_gen

# def build_tiny_model():
#     """Build a very small CNN model"""
#     print("Building tiny model...")
    
#     model = Sequential([
#         Conv2D(8, (3, 3), activation='relu', input_shape=(128, 128, 3)),
#         MaxPooling2D((2, 2)),
#         Conv2D(16, (3, 3), activation='relu'),
#         MaxPooling2D((2, 2)),
#         Flatten(),
#         Dense(16, activation='relu'),
#         Dense(NUM_CLASSES, activation='softmax')
#     ])
    
#     model.compile(
#         optimizer='adam',
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )
    
#     return model

# def safe_explanations(model, sample_image):
#     """Generate explanations with memory safeguards"""
#     print("Attempting explanations...")
    
#     # SHAP with minimal background
#     print("Trying SHAP...")
#     try:
#         background = np.zeros((1, *IMG_SIZE, 3))
#         explainer = shap.DeepExplainer(model, background)
#         shap_values = explainer.shap_values(np.expand_dims(sample_image, axis=0)[:1])  # Limit to 1 sample
        
#         plt.figure(figsize=(6, 3))
#         shap.image_plot(shap_values, -np.expand_dims(sample_image, axis=0)[:1])
#         plt.savefig('shap_tiny.png', bbox_inches='tight', dpi=100)
#         plt.close()
#         print("SHAP completed")
#     except Exception as e:
#         print(f"SHAP failed: {str(e)[:100]}...")

#     # LIME with reduced samples
#     print("Trying LIME...")
#     try:
#         explainer = lime_image.LimeImageExplainer()
#         explanation = explainer.explain_instance(
#             sample_image.astype('double'),
#             model.predict,
#             top_labels=1,  # Only explain top class
#             hide_color=0,
#             num_samples=100  # Drastically reduced
#         )
        
#         temp, mask = explanation.get_image_and_mask(
#             explanation.top_labels[0],
#             positive_only=True,
#             num_features=2,  # Minimal features
#             hide_rest=True
#         )
        
#         plt.figure(figsize=(6, 3))
#         plt.imshow(mark_boundaries(temp, mask))
#         plt.axis('off')
#         plt.savefig('lime_tiny.png', bbox_inches='tight', dpi=100)
#         plt.close()
#         print("LIME completed")
#     except Exception as e:
#         print(f"LIME failed: {str(e)[:100]}...")

# def main():
#     """Main execution with memory safeguards"""
#     try:
#         # Load data
#         train_gen, val_gen = load_minimal_data()
        
#         # Build model
#         model = build_tiny_model()
#         print(model.summary())
        
#         # Minimal training
#         print("Minimal training...")
#         model.fit(
#             train_gen,
#             steps_per_epoch=2,  # Very few steps
#             epochs=EPOCHS,
#             validation_data=val_gen,
#             validation_steps=1
#         )
        
#         # Get sample
#         sample_image, _ = next(val_gen)
#         sample_image = sample_image[0]
        
#         # Attempt explanations
#         safe_explanations(model, sample_image)
        
#         print("Script completed (minimal execution)")
#     except Exception as e:
#         print(f"Fatal error: {str(e)}")

# if __name__ == "__main__":
#     main()

# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only mode

# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import shap
# from lime import lime_image
# from skimage.segmentation import mark_boundaries

# print("Imports successful - starting script")

# # Configuration
# DATA_DIR = 'pet_ct_images'  # Directory with subfolders 'benign' and 'malignant'
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 8  # Reduced to prevent memory issues
# EPOCHS = 5  # Reduced for demonstration
# NUM_CLASSES = 2

# def create_dummy_dataset():
#     """Create synthetic PET/CT images for testing"""
#     print("Creating dummy dataset...")
#     os.makedirs(os.path.join(DATA_DIR, 'benign'), exist_ok=True)
#     os.makedirs(os.path.join(DATA_DIR, 'malignant'), exist_ok=True)
    
#     for i in range(20):  # Create 20 images per class
#         # Benign (uniform noise)
#         benign_img = np.random.randint(50, 150, (256, 256, 3), dtype=np.uint8)
#         cv2.imwrite(os.path.join(DATA_DIR, 'benign', f'benign_{i}.png'), benign_img)
        
#         # Malignant (noise with bright spot)
#         malignant_img = np.random.randint(50, 150, (256, 256, 3), dtype=np.uint8)
#         cv2.circle(malignant_img, (128, 128), 30, (255, 200, 150), -1)
#         cv2.imwrite(os.path.join(DATA_DIR, 'malignant', f'malignant_{i}.png'), malignant_img)
#     print("Dummy dataset created")

# def load_data():
#     """Load and preprocess the image data"""
#     print("Loading data...")
    
#     if not os.path.exists(DATA_DIR):
#         print(f"Directory {DATA_DIR} not found - creating dummy data")
#         create_dummy_dataset()
    
#     # Simple data augmentation
#     datagen = ImageDataGenerator(
#         rescale=1./255,
#         validation_split=0.2,
#         rotation_range=15,  # Reduced from 20
#         width_shift_range=0.1,
#         height_shift_range=0.1,
#         horizontal_flip=True
#     )
    
#     print("Creating training generator...")
#     train_generator = datagen.flow_from_directory(
#         DATA_DIR,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode='categorical',
#         subset='training'
#     )
    
#     print("Creating validation generator...")
#     val_generator = datagen.flow_from_directory(
#         DATA_DIR,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode='categorical',
#         subset='validation'
#     )
    
#     return train_generator, val_generator

# def build_simpler_model():
#     """Build a lighter version of the model"""
#     print("Building model...")
    
#     # Use ResNet50 without top layers
#     base_model = ResNet50(
#         weights='imagenet',
#         include_top=False,
#         input_shape=(224, 224, 3),
#         pooling='avg'  # Adds GlobalAveragePooling2D automatically
#     )
    
#     # Freeze base layers
#     base_model.trainable = False
    
#     # Simplified architecture
#     inputs = Input(shape=(224, 224, 3))
#     x = base_model(inputs)
#     outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
#     model = Model(inputs, outputs)
#     model.compile(
#         optimizer='adam',
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )
    
#     return model

# def generate_explanations(model, sample_image):
#     """Generate SHAP and LIME explanations"""
#     print("Generating explanations...")
    
#     # SHAP Explanation
#     print("Creating SHAP explanation...")
#     try:
#         background = np.zeros((1, 224, 224, 3))
#         explainer = shap.DeepExplainer(model, background)
#         shap_values = explainer.shap_values(np.expand_dims(sample_image, axis=0))
        
#         plt.figure(figsize=(10, 5))
#         shap.image_plot(shap_values, -np.expand_dims(sample_image, axis=0))
#         plt.title('SHAP Explanation')
#         plt.tight_layout()
#         plt.savefig('shap_explanation.png', bbox_inches='tight', dpi=150)
#         plt.close()
#         print("SHAP explanation saved")
#     except Exception as e:
#         print(f"SHAP failed: {str(e)}")
    
#     # LIME Explanation
#     print("Creating LIME explanation...")
#     try:
#         explainer = lime_image.LimeImageExplainer()
#         explanation = explainer.explain_instance(
#             sample_image.astype('double'),
#             model.predict,
#             top_labels=2,
#             hide_color=0,
#             num_samples=500  # Reduced from 1000
#         )
        
#         temp, mask = explanation.get_image_and_mask(
#             explanation.top_labels[0],
#             positive_only=True,
#             num_features=3,  # Reduced from 5
#             hide_rest=False
#         )
        
#         plt.figure(figsize=(10, 5))
#         plt.subplot(1, 2, 1)
#         plt.imshow(sample_image)
#         plt.title('Original Image')
#         plt.axis('off')
        
#         plt.subplot(1, 2, 2)
#         plt.imshow(mark_boundaries(temp, mask))
#         plt.title('LIME Explanation')
#         plt.axis('off')
        
#         plt.tight_layout()
#         plt.savefig('lime_explanation.png', bbox_inches='tight', dpi=150)
#         plt.close()
#         print("LIME explanation saved")
#     except Exception as e:
#         print(f"LIME failed: {str(e)}")

# def main():
#     """Main execution flow"""
#     print("Starting NSCLC classification script")
    
#     # Load data
#     train_gen, val_gen = load_data()
    
#     # Build model
#     model = build_simpler_model()
    
#     # Train model
#     print("Training model...")
#     try:
#         history = model.fit(
#             train_gen,
#             steps_per_epoch=min(10, len(train_gen)),  # Limit steps
#             validation_data=val_gen,
#             validation_steps=min(5, len(val_gen)),
#             epochs=EPOCHS,
#             verbose=1
#         )
#     except Exception as e:
#         print(f"Training failed: {str(e)}")
#         return
    
#     # Save model
#     try:
#         model.save('nsclc_classifier.h5')
#         print("Model saved successfully")
#     except Exception as e:
#         print(f"Model save failed: {str(e)}")
    
#     # Get sample image for explanation
#     sample_image, _ = next(val_gen)
#     sample_image = sample_image[0]
    
#     # Generate explanations
#     generate_explanations(model, sample_image)
    
#     print("Script completed successfully")

# if __name__ == "__main__":
#     main()

#chatGPT
# import numpy as np
# import matplotlib.pyplot as plt
# import shap
# import lime
# from lime import lime_image
# from skimage.segmentation import mark_boundaries

# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import torchvision.models as models
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader
# from PIL import Image

# # Load pretrained ResNet for PET/CT classification
# model = models.resnet18(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, 2)  # Assume binary (malignant/benign)
# model.eval()

# # Transformation for PET/CT images
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# # Load example image
# img_path = 'example_pet_or_ct_image.jpg'
# img = Image.open(img_path).convert('RGB')
# img_tensor = transform(img).unsqueeze(0)

# # Get prediction
# with torch.no_grad():
#     output = model(img_tensor)
#     prediction = torch.argmax(output, dim=1).item()

# # LIME explanation
# explainer = lime_image.LimeImageExplainer()
# def batch_predict(images):
#     model.eval()
#     batch = torch.stack([transform(Image.fromarray(img)) for img in images], dim=0)
#     with torch.no_grad():
#         outputs = model(batch)
#     return outputs.numpy()

# explanation = explainer.explain_instance(np.array(img), batch_predict, top_labels=2, hide_color=0, num_samples=1000)

# # Visualize LIME
# temp, mask = explanation.get_image_and_mask(prediction, positive_only=True, num_features=5, hide_rest=False)
# plt.imshow(mark_boundaries(temp, mask))
# plt.title("LIME Explanation")
# plt.axis('off')
# plt.show()

# # SHAP explanation
# background = torch.cat([img_tensor for _ in range(50)], dim=0)  # dummy background
# e = shap.DeepExplainer(model, background)
# shap_values = e.shap_values(img_tensor)

# # Visualize SHAP
# shap.image_plot(shap_values, np.array([np.array(img.resize((224, 224)))]) / 255.0)
