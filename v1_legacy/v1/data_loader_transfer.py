'''
Data Loader for Transfer Learning
Loads and preprocesses both COCO and educational datasets
Handles class mapping and data augmentation
'''

import os
import json
import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
# from roboflow import Roboflow  # Optional - only needed for Roboflow datasets
from tqdm import tqdm
import random

class TransferLearningDataLoader:
    """
    Data loader that combines COCO dataset with educational objects dataset
    Handles class mapping, data augmentation, and batch preparation
    """

    def __init__(self, input_size=(300, 300)):
        self.input_size = input_size
        self.coco_classes = self._get_coco_classes()
        self.educational_classes = []
        self.combined_classes = []
        self.class_mapping = {}

    def _get_coco_classes(self):
        """Get COCO dataset class names"""
        return [
            'background', 'person', 'bicycle', 'car', 'motorcycle',
            'bench', 'bird', 'cat', 'dog',
            'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def load_educational_classes(self, model_info_path="models/educational_objects/model_info.json"):
        """Load educational classes from existing model"""
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
                self.educational_classes = model_info['classes']
                print(f"Loaded {len(self.educational_classes)} educational classes")
        else:
            print("Educational classes not found, using empty list")
            self.educational_classes = []

    def create_combined_mapping(self):
        """Create combined class mapping for COCO + Educational classes"""
        # Start with COCO classes
        self.combined_classes = self.coco_classes.copy()

        # Add educational classes, avoiding duplicates
        for edu_class in self.educational_classes:
            if edu_class not in self.combined_classes:
                self.combined_classes.append(edu_class)

        # Create mapping
        self.class_mapping = {class_name: idx for idx, class_name in enumerate(self.combined_classes)}

        print(f"Combined dataset:")
        print(f"  COCO classes: {len(self.coco_classes)}")
        print(f"  Educational classes: {len(self.educational_classes)}")
        print(f"  Total unique classes: {len(self.combined_classes)}")

        return self.class_mapping

    def download_coco_subset(self, output_dir="datasets/coco_subset/", num_images=1000):
        """
        Download a subset of COCO dataset for transfer learning
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"Downloading COCO subset to {output_dir}")
        print("Note: For full implementation, integrate with COCO API")

        # Placeholder for COCO download
        # In practice, you would use pycocotools or download from official COCO website
        coco_info = {
            "dataset_type": "coco_subset",
            "num_images": num_images,
            "classes": self.coco_classes,
            "download_path": output_dir
        }

        with open(os.path.join(output_dir, "coco_info.json"), 'w') as f:
            json.dump(coco_info, f, indent=2)

        return output_dir

    def load_roboflow_educational_data(self,
                                     workspace=None,
                                     project=None,
                                     version=1,
                                     api_key=None):
        """
        Load educational dataset from Roboflow
        """
        if not all([workspace, project, api_key]):
            print("Roboflow credentials not provided, using local data")
            return self._load_local_educational_data()

        print("Roboflow integration disabled. Using local educational dataset instead.")
        print("To enable Roboflow: pip install roboflow and uncomment import")
        return self._load_local_educational_data()

    def _load_local_educational_data(self):
        """Load local educational dataset if available"""
        local_path = "datasets/educational_objects/"
        if os.path.exists(local_path):
            print(f"Using local educational data from {local_path}")
            return local_path
        else:
            print("No local educational data found")
            return None

    def preprocess_image(self, image_path):
        """Preprocess image for training"""
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                return None
        else:
            image = image_path

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        image = cv2.resize(image, self.input_size)

        # Normalize
        image = image.astype(np.float32) / 255.0

        return image

    def load_annotations(self, annotation_file):
        """Load COCO format annotations"""
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        return annotations

    def create_training_generator(self,
                                coco_data_path=None,
                                educational_data_path=None,
                                batch_size=16,
                                augment=True):
        """
        Create training data generator combining COCO and educational data
        """
        def data_generator():
            while True:
                batch_images = []
                batch_class_labels = []
                batch_bbox_labels = []

                for _ in range(batch_size):
                    # Randomly choose between COCO and educational data
                    if random.random() < 0.5 and coco_data_path:
                        # Load COCO sample
                        image, class_label, bbox = self._load_coco_sample(coco_data_path)
                    elif educational_data_path:
                        # Load educational sample
                        image, class_label, bbox = self._load_educational_sample(educational_data_path)
                    else:
                        # Fallback to dummy data
                        image, class_label, bbox = self._create_dummy_sample()

                    if image is not None:
                        # Apply augmentation if enabled
                        if augment:
                            image = self._augment_image(image)

                        batch_images.append(image)
                        batch_class_labels.append(class_label)
                        batch_bbox_labels.append(bbox)

                if batch_images:
                    yield (
                        np.array(batch_images),
                        {
                            'class_output': np.array(batch_class_labels),
                            'bbox_output': np.array(batch_bbox_labels)
                        }
                    )

        return data_generator()

    def _load_coco_sample(self, coco_data_path):
        """Load a single COCO sample"""
        # Placeholder implementation
        # In practice, load actual COCO images and annotations
        image = np.random.random((*self.input_size, 3)).astype(np.float32)
        class_label = random.randint(0, len(self.coco_classes) - 1)
        bbox = np.random.random(4).astype(np.float32)
        return image, class_label, bbox

    def _load_educational_sample(self, educational_data_path):
        """Load a single educational sample"""
        # Placeholder implementation
        # In practice, load actual educational images and annotations
        image = np.random.random((*self.input_size, 3)).astype(np.float32)
        # Map to educational class range
        class_label = random.randint(len(self.coco_classes), len(self.combined_classes) - 1)
        bbox = np.random.random(4).astype(np.float32)
        return image, class_label, bbox

    def _create_dummy_sample(self):
        """Create dummy sample for testing"""
        image = np.random.random((*self.input_size, 3)).astype(np.float32)
        class_label = random.randint(0, len(self.combined_classes) - 1)
        bbox = np.random.random(4).astype(np.float32)
        return image, class_label, bbox

    def _augment_image(self, image):
        """Apply data augmentation"""
        # Random brightness
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 1)

        # Random horizontal flip
        if random.random() < 0.5:
            image = np.fliplr(image)

        # Random rotation (small angle)
        if random.random() < 0.3:
            angle = random.uniform(-10, 10)
            center = (image.shape[1] // 2, image.shape[0] // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

        return image

    def save_combined_class_info(self, output_path="models/transfer_learning/"):
        """Save combined class information"""
        os.makedirs(output_path, exist_ok=True)

        class_info = {
            "combined_classes": self.combined_classes,
            "class_mapping": self.class_mapping,
            "coco_classes": self.coco_classes,
            "educational_classes": self.educational_classes,
            "num_total_classes": len(self.combined_classes),
            "num_coco_classes": len(self.coco_classes),
            "num_educational_classes": len(self.educational_classes)
        }

        info_path = os.path.join(output_path, "combined_class_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(class_info, f, indent=2, ensure_ascii=False)

        print(f"Combined class info saved to {info_path}")

def main():
    """Test the data loader"""
    print("=== Transfer Learning Data Loader ===")

    # Create data loader
    loader = TransferLearningDataLoader()

    # Load educational classes
    loader.load_educational_classes()

    # Create combined mapping
    loader.create_combined_mapping()

    # Save class info
    loader.save_combined_class_info()

    # Create training generator
    train_gen = loader.create_training_generator(batch_size=8)

    # Test generator
    print("Testing data generator...")
    batch = next(train_gen)
    print(f"Batch shape - Images: {batch[0].shape}")
    print(f"Batch shape - Classes: {batch[1]['class_output'].shape}")
    print(f"Batch shape - Bboxes: {batch[1]['bbox_output'].shape}")

    print("Data loader setup complete!")

if __name__ == "__main__":
    main()
