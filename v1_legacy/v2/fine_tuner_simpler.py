from ultralytics import YOLO
import torch
import os

def download_yolov5_model():
    """
    Download YOLOv5 model if not present.
    """
    if not os.path.exists("yolov5m.pt"):
        print("Downloading YOLOv5m model...")
        model = YOLO("yolov5m.pt")  # This will auto-download
        return model
    else:
        return YOLO("yolov5m.pt")

def freeze_layers(model, freeze_layers=10):
    """
    Freeze the first N layers of the model to preserve COCO features.

    Args:
        model: YOLO model
        freeze_layers: Number of layers to freeze (default: 10)
    """
    print(f"Freezing first {freeze_layers} layers...")

    # Access the model's PyTorch model
    if hasattr(model.model, 'model'):
        pytorch_model = model.model.model
    else:
        pytorch_model = model.model

    # Freeze backbone layers (preserve COCO features)
    for i, (name, param) in enumerate(pytorch_model.named_parameters()):
        if i < freeze_layers:
            param.requires_grad = False
            print(f"Frozen layer {i}: {name}")
        else:
            param.requires_grad = True

    return model

def train_with_freezing():
    """
    Train a YOLOv5 model with layer freezing to preserve COCO knowledge.
    """
    # Download/load YOLOv5 model (pre-trained on COCO)
    model = download_yolov5_model()

    # Freeze early layers to preserve COCO features
    model = freeze_layers(model, freeze_layers=10)

    print("Starting fine-tuning with frozen layers...")

    # Fine-tune on pen dataset while keeping COCO knowledge
    results = model.train(
        data="datasets/pen_grande/data.yaml",
        epochs=100,  # Increased epochs for better convergence
        imgsz=640,
        batch=16,
        optimizer="Adam",
        lr0=0.0005,  # Lower learning rate for fine-tuning
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        device="cuda:0",
        name='pen_yolov5_frozen',
        save=True,
        save_period=5,
        save_json=True,
        project="runs/simpler",
        exist_ok=True,
        resume=False,
        patience=15,
        workers=8,
        # Additional parameters for fine-tuning
        box=7.5,  # Box loss gain
        cls=0.5,  # Class loss gain
        dfl=1.5,  # DFL loss gain
        pose=12.0,  # Pose loss gain
        kobj=1.0,  # Keypoint obj loss gain
        label_smoothing=0.0,
        nbs=64,  # Nominal batch size
        hsv_h=0.015,  # Image HSV-Hue augmentation
        hsv_s=0.7,  # Image HSV-Saturation augmentation
        hsv_v=0.4,  # Image HSV-Value augmentation
        degrees=0.0,  # Image rotation
        translate=0.1,  # Image translation
        scale=0.5,  # Image scale
        shear=0.0,  # Image shear
        perspective=0.0,  # Image perspective
        flipud=0.0,  # Image flip up-down
        fliplr=0.5,  # Image flip left-right
        mosaic=1.0,  # Image mosaic
        mixup=0.0,  # Image mixup
        copy_paste=0.0  # Segment copy-paste
    )

    return results

def train_without_freezing():
    """
    Train YOLOv5 without freezing for comparison.
    """
    model = download_yolov5_model()

    print("Starting training without frozen layers...")

    results = model.train(
        data="datasets/pen_grande/data.yaml",
        epochs=15,
        imgsz=640,
        batch=16,
        optimizer="Adam",
        lr0=0.001,  # Higher learning rate for full training
        device="cuda:0",
        name='pen_yolov5_full',
        save=True,
        save_period=5,
        save_json=True,
        project="runs/simpler",
        exist_ok=True,
        resume=False,
        patience=15,
        workers=8
    )

    return results

def validate_frozen_model():
    """
    Validate the frozen YOLOv5 model on the pen dataset.
    """
    model_path = "runs/simpler/pen_yolov5_frozen/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Train the model first.")
        return None

    model = YOLO(model_path)

    print("Validating frozen model...")
    metrics = model.val(
        data="datasets/pen_grande/data.yaml",
        imgsz=640,
        project="runs/simpler",
        split="val",
        conf=0.25,  # Lower confidence for better recall
        iou=0.45,   # Lower IoU threshold
        device="cuda:0"
    )
    print("Frozen Model Metrics:", metrics.results_dict)
    return metrics

def validate_full_model():
    """
    Validate the full YOLOv5 model on the pen dataset.
    """
    model_path = "runs/simpler/pen_yolov5_full/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Train the model first.")
        return None

    model = YOLO(model_path)

    print("Validating full model...")
    metrics = model.val(
        data="datasets/pen_grande/data.yaml",
        imgsz=640,
        project="runs/simpler",
        split="val",
        conf=0.25,
        iou=0.45,
        device="cuda:0"
    )
    print("Full Model Metrics:", metrics.results_dict)
    return metrics

def test_coco_retention():
    """
    Test if the frozen model still retains COCO detection capabilities.
    """
    model_path = "runs/simpler/pen_yolov5_frozen/weights/best.pt"
    if not os.path.exists(model_path):
        print("Frozen model not found. Train first.")
        return

    model = YOLO(model_path)

    # Test on a sample image to see if it can still detect COCO objects
    print("Testing COCO object retention...")
    # You can add a test image path here to verify COCO detection still works

def compare_models():
    """
    Compare the performance of frozen vs full training approaches.
    """
    print("\n" + "="*50)
    print("COMPARING MODEL PERFORMANCE")
    print("="*50)

    frozen_metrics = validate_frozen_model()
    '''
    full_metrics = validate_full_model()

    if frozen_metrics and full_metrics:
        print("\nPerformance Comparison:")
        print(f"Frozen Model mAP50: {frozen_metrics.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"Full Model mAP50: {full_metrics.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"Frozen Model mAP50-95: {frozen_metrics.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"Full Model mAP50-95: {full_metrics.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    '''
if __name__ == "__main__":
    # Choose training approach
    print("YOLOv5 Fine-tuning with Freezing Technique")
    print("==========================================")
    print("1. Train with frozen layers (preserves COCO)")
    print("2. Train without frozen layers (full training)")
    print("3. Compare both models")
    print("4. Validate existing models")

    choice = input("Enter your choice (1-4): ").strip()

    if choice == "1":
        print("\nTraining with frozen layers...")
        train_with_freezing()
    elif choice == "2":
        print("\nTraining without frozen layers...")
        train_without_freezing()
    elif choice == "3":
        print("\nTraining both approaches for comparison...")
        train_with_freezing()
        train_without_freezing()
        compare_models()
    elif choice == "4":
        compare_models()
    else:
        print("Invalid choice. Running frozen training by default...")
        train_with_freezing()
