#!/usr/bin/env python3
"""
Test script to verify the fixed model correctly identifies pens and other objects
"""

import os
from ultralytics import YOLO
import glob

def test_fixed_model():
    """Test the fixed model with proper class mapping"""
    
    # Find the latest fixed model
    model_paths = [
        './runs/detect/fixed_combined_training/weights/best.pt',
        './trained_model/best.pt'
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("No trained model found. Please run training first.")
        return
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Print model class names to verify
    print("\nModel classes:")
    for i, name in enumerate(model.names.values()):
        print(f"{i}: {name}")
    
    # Find test images
    test_image_patterns = [
        './combined_dataset/test/images/*.jpg',
        './combined_dataset/valid/images/*.jpg'
    ]
    
    test_images = []
    for pattern in test_image_patterns:
        test_images.extend(glob.glob(pattern))
        if len(test_images) >= 5:  # Test on first 5 images
            break
    
    if not test_images:
        print("No test images found.")
        return
    
    print(f"\nTesting on {min(5, len(test_images))} images:")
    
    for i, image_path in enumerate(test_images[:5]):
        print(f"\n{i+1}. Testing: {os.path.basename(image_path)}")
        
        # Run inference
        results = model(image_path, conf=0.25, save=False, verbose=False)
        
        # Print detections
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]
                    print(f"   ✓ Detected: {class_name} (confidence: {confidence:.2f})")
            else:
                print("   ✗ No objects detected")

def test_pen_specifically():
    """Test specifically on pen images"""
    
    # Find the latest fixed model
    model_paths = [
        './runs/detect/fixed_combined_training/weights/best.pt',
        './trained_model/best.pt'
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("No trained model found.")
        return
    
    print(f"Loading model for pen testing: {model_path}")
    model = YOLO(model_path)
    
    # Find pen images (they should have 'pen' in the filename from the dataset)
    pen_images = []
    search_patterns = [
        './combined_dataset/test/images/*pen*.jpg',
        './combined_dataset/valid/images/*pen*.jpg',
        './datasets/pen_detection.v1i.yolov11/test/images/*.jpg',
        './datasets/pen_detection.v1i.yolov11/valid/images/*.jpg'
    ]
    
    for pattern in search_patterns:
        pen_images.extend(glob.glob(pattern))
    
    if not pen_images:
        print("No pen test images found.")
        return
    
    print(f"\nTesting pen detection on {min(3, len(pen_images))} pen images:")
    
    pen_detected = 0
    for i, image_path in enumerate(pen_images[:3]):
        print(f"\n{i+1}. Testing pen image: {os.path.basename(image_path)}")
        
        # Run inference
        results = model(image_path, conf=0.25, save=False, verbose=False)
        
        found_pen = False
        # Print all detections
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]
                    print(f"   Detected: {class_name} (confidence: {confidence:.2f})")
                    if class_name == 'pen':
                        found_pen = True
                        pen_detected += 1
            else:
                print("   No objects detected")
        
        if not found_pen:
            print("   ⚠️  No pen detected in this image")
    
    print(f"\nSummary: Pen detected in {pen_detected}/{min(3, len(pen_images))} test images")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'pen':
        test_pen_specifically()
    else:
        print("Testing Fixed Model - YOLO with Proper Class Mapping")
        print("=" * 60)
        test_fixed_model()
        print("\n" + "=" * 60)
        test_pen_specifically()