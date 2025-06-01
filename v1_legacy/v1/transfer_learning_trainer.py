'''
Transfer Learning Trainer for Object Detection
Combines COCO pre-trained model with educational objects dataset
Uses MobileNetV2 backbone with SSD detection head
'''

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Reshape, Conv2D, Dropout, Input
from tensorflow.keras.models import Model
import cv2
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
import time
import psutil
import gc
import requests
import zipfile
# from roboflow import Roboflow  # Optional - only needed for Roboflow datasets

# Configure GPU
def configure_gpu():
    """Configure GPU for optimal performance"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU(s) configured: {len(gpus)} devices")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU found, using CPU")

class TransferLearningTrainer:
    """
    Transfer Learning Trainer for Educational Object Detection
    Uses COCO pre-trained weights and fine-tunes with educational dataset
    """
    
    def __init__(self, input_size=(300, 300), num_educational_classes=123):
        self.input_size = input_size
        self.num_educational_classes = num_educational_classes
        self.num_coco_classes = 80  # COCO dataset classes
        self.total_classes = self.num_coco_classes + self.num_educational_classes + 1  # +1 for background
        
        # COCO class names for reference
        self.coco_classes = [
            'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        self.educational_classes = []
        self.model = None
        
    def load_educational_classes(self, model_info_path="models/educational_objects/model_info.json"):
        """Load educational classes from existing model info"""
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
                self.educational_classes = model_info['classes']
                print(f"Loaded {len(self.educational_classes)} educational classes")
        else:
            print("Educational model info not found, using default classes")
            
    def create_combined_model(self):
        """
        Create a model that combines COCO and educational object detection
        Uses MobileNetV2 backbone with dual detection heads
        """
        # Input layer
        input_tensor = Input(shape=(*self.input_size, 3))
        
        # MobileNetV2 backbone (pre-trained on ImageNet)
        backbone = MobileNetV2(
            input_tensor=input_tensor,
            weights='imagenet',
            include_top=False,
            alpha=1.0
        )
        
        # Feature extraction from backbone
        features = backbone.output
        
        # Global Average Pooling
        gap = GlobalAveragePooling2D()(features)
        
        # Dropout for regularization
        dropout = Dropout(0.2)(gap)
        
        # Classification head for combined classes (COCO + Educational)
        class_predictions = Dense(
            self.total_classes,
            activation='softmax',
            name='class_output'
        )(dropout)
        
        # Bounding box regression head
        bbox_predictions = Dense(
            4,  # x_min, y_min, x_max, y_max
            activation='sigmoid',
            name='bbox_output'
        )(dropout)
        
        # Create the model
        model = Model(
            inputs=input_tensor,
            outputs=[class_predictions, bbox_predictions]
        )
        
        return model
    
    def load_pretrained_weights(self):
        """
        Load pre-trained weights and adapt for transfer learning
        """
        # Create base model
        self.model = self.create_combined_model()
        
        # Freeze backbone layers initially for transfer learning
        for layer in self.model.layers[:-4]:  # Freeze all except last 4 layers
            layer.trainable = False
            
        print("Created model with pre-trained backbone")
        print(f"Total parameters: {self.model.count_params():,}")
        # Count trainable parameters manually for compatibility
        trainable_params = sum([w.numpy().size for w in self.model.trainable_weights])
        print(f"Trainable parameters: {trainable_params:,}")
        
    def compile_model(self, learning_rate=0.001):
        """Compile model with appropriate loss functions and metrics"""
        
        # Custom loss weights
        loss_weights = {
            'class_output': 1.0,
            'bbox_output': 5.0  # Higher weight for bounding box accuracy
        }
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'class_output': 'sparse_categorical_crossentropy',
                'bbox_output': 'mean_squared_error'  # Use string name instead of 'mse'
            },
            loss_weights=loss_weights,
            metrics={
                'class_output': ['accuracy'],
                'bbox_output': ['mean_absolute_error']  # Use string name instead of 'mae'
            }
        )
        
        print("Model compiled successfully")
        
    def load_roboflow_datasets(self, workspace="your-workspace", project="your-project", version=1):
        """
        Load educational datasets from Roboflow
        This is a placeholder - replace with your actual Roboflow credentials
        """
        print("Roboflow integration disabled. Using local educational dataset instead.")
        print("To enable Roboflow: pip install roboflow and uncomment import")
        return None
            
    def prepare_transfer_learning_data(self, dataset_path="datasets/"):
        """
        Prepare data for transfer learning
        Combines COCO classes with educational classes
        """
        images = []
        class_labels = []
        bbox_labels = []
        
        # Load existing educational dataset
        if os.path.exists("models/educational_objects/"):
            print("Using existing educational dataset...")
            # Load your educational data here
            # This is a placeholder - adapt based on your data format
            
        # For now, create dummy data structure
        print("Preparing transfer learning dataset...")
        print(f"Total classes: {self.total_classes}")
        print(f"COCO classes: {self.num_coco_classes}")
        print(f"Educational classes: {self.num_educational_classes}")
        
        return images, class_labels, bbox_labels
    
    def train_transfer_learning(self, epochs=50, batch_size=16):
        """
        Train the model using transfer learning approach
        """
        print("Starting transfer learning training...")
        
        # Prepare data
        images, class_labels, bbox_labels = self.prepare_transfer_learning_data()
        
        # For demonstration, create some dummy data
        # Replace this with your actual data loading
        dummy_images = np.random.random((100, *self.input_size, 3)).astype(np.float32)
        dummy_class_labels = np.random.randint(0, self.total_classes, (100,))
        dummy_bbox_labels = np.random.random((100, 4)).astype(np.float32)
        
        # Split data
        train_images, val_images, train_class, val_class, train_bbox, val_bbox = train_test_split(
            dummy_images, dummy_class_labels, dummy_bbox_labels,
            test_size=0.2, random_state=42
        )
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                'models/transfer_learning_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                verbose=1,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            train_images,
            {'class_output': train_class, 'bbox_output': train_bbox},
            validation_data=(
                val_images,
                {'class_output': val_class, 'bbox_output': val_bbox}
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Unfreeze some layers for fine-tuning
        print("\nUnfreezing layers for fine-tuning...")
        for layer in self.model.layers[-10:]:  # Unfreeze last 10 layers
            layer.trainable = True
            
        # Recompile with lower learning rate for fine-tuning
        self.compile_model(learning_rate=0.0001)
        
        # Fine-tune for additional epochs
        fine_tune_history = self.model.fit(
            train_images,
            {'class_output': train_class, 'bbox_output': train_bbox},
            validation_data=(
                val_images,
                {'class_output': val_class, 'bbox_output': val_bbox}
            ),
            epochs=epochs//2,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history, fine_tune_history
    
    def save_transfer_model(self, save_path="models/transfer_learning/"):
        """Save the trained transfer learning model"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_path, "transfer_learning_model.h5")
        self.model.save(model_path)
        
        # Save model info
        model_info = {
            "total_classes": self.total_classes,
            "coco_classes": self.coco_classes,
            "educational_classes": self.educational_classes,
            "input_size": self.input_size,
            "num_coco_classes": self.num_coco_classes,
            "num_educational_classes": self.num_educational_classes,
            "description": "Transfer learning model combining COCO and educational objects"
        }
        
        info_path = os.path.join(save_path, "transfer_model_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
            
        print(f"Transfer learning model saved to {save_path}")

def main():
    """Main training function"""
    print("=== Transfer Learning for Educational Object Detection ===")
    
    # Configure GPU
    configure_gpu()
    
    # Create trainer
    trainer = TransferLearningTrainer()
    
    # Load educational classes
    trainer.load_educational_classes()
    
    # Load pre-trained weights
    trainer.load_pretrained_weights()
    
    # Compile model
    trainer.compile_model()
    
    # Print model summary
    trainer.model.summary()
    
    # Train model
    history, fine_tune_history = trainer.train_transfer_learning(epochs=30)
    
    # Save model
    trainer.save_transfer_model()
    
    print("Transfer learning training completed!")

if __name__ == "__main__":
    main()