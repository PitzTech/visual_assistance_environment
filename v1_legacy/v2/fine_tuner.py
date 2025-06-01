import os
from ultralytics import YOLO
import yaml
import shutil
from pathlib import Path
import glob

class YOLO11FineTuner:
    def __init__(self, base_model='yolo11s.pt', datasets_folder='./datasets'):
        """
        Initialize YOLO11 fine-tuner

        Args:
            base_model: Pre-trained YOLO11 model to start from
                       - yolo11n.pt (nano - fastest)
                       - yolo11s.pt (small - recommended)
                       - yolo11m.pt (medium - better accuracy)
            datasets_folder: Path to folder containing all datasets
        """
        self.base_model = base_model
        self.model = None
        self.datasets_folder = datasets_folder
        self.discovered_datasets = []

    def discover_datasets(self):
        """
        Automatically discover all datasets in the datasets folder
        
        Returns:
            list: List of discovered dataset information
        """
        print(f"Searching for datasets in {self.datasets_folder}...")
        
        datasets = []
        
        # Search for data.yaml files in subdirectories
        yaml_files = glob.glob(os.path.join(self.datasets_folder, "**/data.yaml"), recursive=True)
        
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                dataset_dir = os.path.dirname(yaml_file)
                dataset_name = os.path.basename(dataset_dir)
                
                dataset_info = {
                    'name': dataset_name,
                    'path': dataset_dir,
                    'yaml_file': yaml_file,
                    'classes': data.get('names', []),
                    'num_classes': data.get('nc', 0),
                    'train_path': data.get('train', ''),
                    'val_path': data.get('val', ''),
                    'test_path': data.get('test', '')
                }
                
                datasets.append(dataset_info)
                print(f"✓ Found dataset: {dataset_name}")
                print(f"  Classes ({dataset_info['num_classes']}): {dataset_info['classes']}")
                print(f"  Path: {dataset_dir}")
                
            except Exception as e:
                print(f"Error reading {yaml_file}: {e}")
        
        self.discovered_datasets = datasets
        print(f"\nTotal datasets discovered: {len(datasets)}")
        return datasets

    def prepare_dataset(self, roboflow_dataset_path, output_path='./classroom_dataset'):
        """
        Prepare Roboflow dataset for YOLO11 training

        Args:
            roboflow_dataset_path: Path to downloaded Roboflow dataset
            output_path: Where to organize the final dataset
        """
        print("Preparing dataset...")

        # Create output directory structure
        os.makedirs(output_path, exist_ok=True)

        # Copy the Roboflow dataset
        if os.path.exists(roboflow_dataset_path):
            shutil.copytree(roboflow_dataset_path, output_path, dirs_exist_ok=True)
            print(f"Dataset copied to {output_path}")
        else:
            print(f"Dataset path {roboflow_dataset_path} not found!")
            return False

        # Verify dataset structure
        required_files = ['data.yaml', 'train', 'valid']
        for item in required_files:
            if not os.path.exists(os.path.join(output_path, item)):
                print(f"Missing required file/folder: {item}")
                return False

        print("Dataset structure verified ✓")
        return True

    def create_combined_dataset(self, selected_datasets=None, output_path='./combined_dataset', include_coco_classes=True):
        """
        Create a combined dataset from selected discovered datasets
        
        Args:
            selected_datasets: List of dataset indices to combine (None = all)
            output_path: Where to create the combined dataset
            include_coco_classes: Whether to include relevant COCO classes
        
        Returns:
            str: Path to combined dataset yaml file
        """
        if not self.discovered_datasets:
            print("No datasets discovered. Run discover_datasets() first.")
            return None
            
        if selected_datasets is None:
            selected_datasets = list(range(len(self.discovered_datasets)))
        
        print(f"Creating combined dataset from {len(selected_datasets)} datasets...")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Get the standard COCO classes in the correct order
        if include_coco_classes:
            # Standard COCO 80 classes - maintaining original order for proper transfer learning
            coco_classes = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
            combined_classes = coco_classes.copy()
        else:
            combined_classes = []
        
        # Collect new classes from datasets and create class mapping
        new_classes = []
        class_mapping = {}  # Maps old class indices to new indices
        
        for idx in selected_datasets:
            dataset = self.discovered_datasets[idx]
            dataset_class_mapping = {}
            
            for old_idx, class_name in enumerate(dataset['classes']):
                if class_name not in combined_classes:
                    combined_classes.append(class_name)
                    new_classes.append(class_name)
                
                # Map old class index to new class index
                new_idx = combined_classes.index(class_name)
                dataset_class_mapping[old_idx] = new_idx
            
            class_mapping[idx] = dataset_class_mapping
        
        # Create directories for combined dataset
        for split in ['train', 'valid', 'test']:
            split_dir = os.path.join(output_path, split, 'images')
            os.makedirs(split_dir, exist_ok=True)
            dest_labels_dir = os.path.join(output_path, split, 'labels')
            os.makedirs(dest_labels_dir, exist_ok=True)
            
            # Copy images and labels from each dataset to combined dataset
            for idx in selected_datasets:
                dataset = self.discovered_datasets[idx]
                dataset_path = dataset['path']
                source_dir = os.path.join(dataset_path, split, 'images')
                source_labels_dir = os.path.join(dataset_path, split, 'labels')
                
                if os.path.exists(source_dir):
                    for img_file in os.listdir(source_dir):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            # Copy image
                            source_file = os.path.join(source_dir, img_file)
                            # Add dataset prefix to avoid filename conflicts
                            dest_img_name = f"{dataset['name']}_{img_file}"
                            dest_file = os.path.join(split_dir, dest_img_name)
                            if not os.path.exists(dest_file):
                                shutil.copy2(source_file, dest_file)
                            
                            # Copy and update corresponding label file
                            label_file = os.path.splitext(img_file)[0] + '.txt'
                            source_label_file = os.path.join(source_labels_dir, label_file)
                            dest_label_name = f"{dataset['name']}_{label_file}"
                            dest_label_file = os.path.join(dest_labels_dir, dest_label_name)
                            
                            if os.path.exists(source_label_file) and not os.path.exists(dest_label_file):
                                # Read and update label file with new class indices
                                with open(source_label_file, 'r') as f:
                                    lines = f.readlines()
                                
                                updated_lines = []
                                for line in lines:
                                    if line.strip():
                                        parts = line.strip().split()
                                        old_class_id = int(parts[0])
                                        # Map to new class index
                                        new_class_id = class_mapping[idx].get(old_class_id, old_class_id)
                                        parts[0] = str(new_class_id)
                                        updated_lines.append(' '.join(parts) + '\n')
                                
                                # Write updated label file
                                with open(dest_label_file, 'w') as f:
                                    f.writelines(updated_lines)

        # Create combined data.yaml
        combined_yaml = {
            'train': './train/images',
            'val': './valid/images', 
            'test': './test/images',
            'nc': len(combined_classes),
            'names': combined_classes
        }
        
        # Save combined yaml
        yaml_path = os.path.join(output_path, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(combined_yaml, f, default_flow_style=False)
        
        print(f"✓ Combined dataset created: {output_path}")
        print(f"✓ Total classes: {len(combined_classes)}")
        print(f"✓ COCO classes: {len(coco_classes) if include_coco_classes else 0}")
        print(f"✓ New custom classes: {new_classes}")
        print(f"✓ All classes: {combined_classes[:10]}..." if len(combined_classes) > 10 else f"✓ All classes: {combined_classes}")
        
        return yaml_path

    def train_on_dataset(self, dataset_info, epochs=100, imgsz=640, batch_size=16, transfer_learning=True):
        """
        Train on a specific dataset with transfer learning
        
        Args:
            dataset_info: Dataset information dictionary
            epochs: Number of training epochs
            imgsz: Image size for training
            batch_size: Batch size
            transfer_learning: Use COCO pretrained weights (keeps all COCO classes + new ones)
        """
        print(f"Training on dataset: {dataset_info['name']}")
        
        if transfer_learning:
            print("Using transfer learning - model will detect COCO classes + custom classes")
        
        # Load pre-trained model
        self.model = YOLO(self.base_model)
        
        # Start training with transfer learning
        results = self.model.train(
            data=dataset_info['yaml_file'],
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            name=f"{dataset_info['name']}_training",
            project='./runs/detect',
            patience=15,
            save=True,
            plots=True,
            verbose=True,
            pretrained=transfer_learning,  # Keep COCO weights
            freeze=0  # Don't freeze any layers
        )
        
        print(f"Training completed for {dataset_info['name']}!")
        return results

    def modify_yaml_config(self, dataset_path):
        """
        Modify the data.yaml file to include new classes alongside COCO classes

        Args:
            dataset_path: Path to dataset containing data.yaml
            new_classes: List of new class names to add
        """
        yaml_path = os.path.join(dataset_path, 'data.yaml')

        # Read existing YAML
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)

        print(f"Adding New classes: {data.get('names', [])}")
        new_classes = data.get('names', [])


        # Use only new classes (no COCO classes for pen detection)
        all_classes = new_classes

        # Update YAML configuration
        data['nc'] = len(all_classes)  # Number of classes
        data['names'] = all_classes

        # Write updated YAML
        with open(yaml_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)

        print(f"Updated classes ({len(all_classes)} total): {all_classes}")
        print(f"New classes added: {new_classes}")

    def start_training(self, dataset_path, epochs=100, imgsz=640, batch_size=16, use_transfer_learning=True, model_name='combined_training'):
        """
        Start fine-tuning the YOLO11 model with proper transfer learning

        Args:
            dataset_path: Path to prepared dataset
            epochs: Number of training epochs
            imgsz: Image size for training
            batch_size: Batch size (adjust based on GPU memory)
            use_transfer_learning: Whether to use COCO-pretrained weights
            model_name: Name for the training run
        """
        print(f"Starting fine-tuning with {self.base_model}...")

        # Load pre-trained model
        self.model = YOLO(self.base_model)
        
        if use_transfer_learning:
            print("Using transfer learning from COCO-pretrained weights...")
            print("The model will retain COCO classes and add new custom classes.")
        else:
            print("Training from scratch without COCO weights...")

        # Start training
        results = self.model.train(
            data=os.path.join(dataset_path, 'data.yaml'),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            name=model_name,
            project='./runs/detect',
            patience=15,  # Early stopping patience
            save=True,
            plots=True,
            verbose=True,
            pretrained=use_transfer_learning,  # Use pretrained weights
            freeze=0,  # Don't freeze any layers - allow full fine-tuning
            resume=False,  # Don't resume from previous training
        )

        print("Training completed!")
        print(f"Best model saved at: ./runs/detect/{model_name}/weights/best.pt")
        return results

    def validate_model(self, dataset_path):
        """
        Validate the fine-tuned model

        Args:
            dataset_path: Path to dataset for validation
        """
        if self.model is None:
            print("No model loaded. Train first or load a trained model.")
            return

        print("Validating model...")
        results = self.model.val(data=os.path.join(dataset_path, 'data.yaml'))
        return results

    def test_detection(self, image_path, model_path=None, conf_threshold=0.25, save_results=True):
        """
        Test the fine-tuned model on an image

        Args:
            image_path: Path to test image
            model_path: Path to trained model (if None, uses latest)
            conf_threshold: Confidence threshold for detections
            save_results: Whether to save detection results
        """
        if self.model is None:
            # Load the specified model or find the latest one
            if model_path is None:
                # Look for the latest best.pt in runs/detect
                best_models = glob.glob('./runs/detect/*/weights/best.pt')
                if best_models:
                    # Sort by modification time and get the latest
                    model_path = max(best_models, key=os.path.getmtime)
                    print(f"Loading latest model: {model_path}")
                else:
                    model_path = './trained_model/best.pt'
                    print(f"Loading model: {model_path}")
            
            self.model = YOLO(model_path)

        # Run detection
        results = self.model(image_path, conf=conf_threshold, save=save_results)

        # Display results
        for result in results:
            result.show()
            # Print detection details
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.model.names[class_id]
                    print(f"Detected: {class_name} (confidence: {confidence:.2f})")

        return results

def main():
    """
    Main function to run YOLO11 fine-tuning with automatic dataset discovery
    """
    print("YOLO11 Fine-tuning with Automatic Dataset Discovery")
    print("=" * 60)

    # Step 1: Initialize fine-tuner
    fine_tuner = YOLO11FineTuner(base_model='yolo11s.pt', datasets_folder='./datasets')

    # Step 2: Discover all datasets
    discovered_datasets = fine_tuner.discover_datasets()
    
    if not discovered_datasets:
        print("No datasets found in the datasets folder!")
        print("Please ensure your datasets are in ./datasets/ with proper data.yaml files")
        return

    # Step 3: Select datasets to train on
    print("\n" + "=" * 60)
    print("Available datasets:")
    for i, dataset in enumerate(discovered_datasets):
        print(f"{i + 1}. {dataset['name']} ({dataset['num_classes']} classes: {dataset['classes']})")
    
    print("\nTraining options:")
    print("1. Train on all datasets (combined)")
    print("2. Train on specific datasets (combined)")
    print("3. Train on individual datasets separately")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    # Training parameters
    print("\nTraining settings:")
    print("- epochs=50-100 (start with 50)")
    print("- batch_size=8-16 (depends on GPU memory)")
    print("- imgsz=640 (standard)")
    
    epochs = int(input("Enter number of epochs (default 50): ") or "50")
    batch_size = int(input("Enter batch size (default 8): ") or "8")
    
    if choice == "1":
        # Train on all datasets combined
        print("\nTraining on all datasets combined...")
        yaml_path = fine_tuner.create_combined_dataset()
        if yaml_path:
            print(f"Training with combined dataset: {yaml_path}")
            # Use the existing start_training method with the combined dataset
            results = fine_tuner.start_training(
                dataset_path=os.path.dirname(yaml_path),
                epochs=epochs,
                batch_size=batch_size,
                model_name='combined_all_datasets'
            )
    
    elif choice == "2":
        # Train on selected datasets combined
        print("\nSelect datasets to combine:")
        selected_indices = []
        for i, dataset in enumerate(discovered_datasets):
            include = input(f"Include {dataset['name']}? (y/n): ").strip().lower()
            if include == 'y':
                selected_indices.append(i)
        
        if selected_indices:
            print(f"\nTraining on {len(selected_indices)} selected datasets combined...")
            yaml_path = fine_tuner.create_combined_dataset(selected_indices)
            if yaml_path:
                results = fine_tuner.start_training(
                    dataset_path=os.path.dirname(yaml_path),
                    epochs=epochs,
                    batch_size=batch_size,
                    model_name='combined_selected_datasets'
                )
        else:
            print("No datasets selected!")
            return
    
    elif choice == "3":
        # Train on individual datasets
        for dataset in discovered_datasets:
            print(f"\nTraining on {dataset['name']}...")
            results = fine_tuner.train_on_dataset(
                dataset,
                epochs=epochs,
                batch_size=batch_size
            )
    
    else:
        print("Invalid choice!")
        return

    print("\nFine-tuning completed!")
    print("Your models are saved in: ./runs/detect/")

def interactive_dataset_selection():
    """
    Interactive function to select and train on specific datasets
    """
    fine_tuner = YOLO11FineTuner(datasets_folder='./datasets')
    datasets = fine_tuner.discover_datasets()
    
    if not datasets:
        print("No datasets found!")
        return None
    
    return fine_tuner, datasets

def quick_retrain():
    """
    Quick retrain function to fix the current model with proper class mapping
    """
    print("Quick Retrain - Fixing Class Mapping Issues")
    print("=" * 50)
    
    # Initialize fine-tuner
    fine_tuner = YOLO11FineTuner(base_model='yolo11s.pt', datasets_folder='./datasets')
    
    # Discover datasets
    datasets = fine_tuner.discover_datasets()
    
    if not datasets:
        print("No datasets found!")
        return
    
    # Remove old combined dataset
    if os.path.exists('./combined_dataset'):
        print("Removing old combined dataset...")
        shutil.rmtree('./combined_dataset')
    
    # Create new combined dataset with proper class mapping
    print("Creating new combined dataset with proper COCO + custom class mapping...")
    yaml_path = fine_tuner.create_combined_dataset(
        selected_datasets=None,  # Use all datasets
        output_path='./combined_dataset',
        include_coco_classes=True
    )
    
    if yaml_path:
        print("Training new model with proper class mapping...")
        results = fine_tuner.start_training(
            dataset_path=os.path.dirname(yaml_path),
            epochs=50,
            batch_size=8,
            use_transfer_learning=True,
            model_name='fixed_combined_training'
        )
        
        print("Retraining completed!")
        print("New model saved at: ./runs/detect/fixed_combined_training/weights/best.pt")
        
        # Copy the best model to trained_model folder
        best_model_path = './runs/detect/fixed_combined_training/weights/best.pt'
        if os.path.exists(best_model_path):
            os.makedirs('./trained_model', exist_ok=True)
            shutil.copy2(best_model_path, './trained_model/best.pt')
            print("Model also copied to: ./trained_model/best.pt")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "quick_retrain":
        quick_retrain()
    else:
        main()
