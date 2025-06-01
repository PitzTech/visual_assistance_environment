'''
Simple detector test to verify labels are showing
'''

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os

def main():
    print("=== SIMPLE DETECTOR TEST ===")
    
    # Load educational model (we know this works)
    model_path = "models/educational_objects/final_model.h5"
    model_info_path = "models/educational_objects/model_info.json"
    
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    
    # Load classes
    with open(model_info_path, 'r', encoding='utf-8') as f:
        model_info = json.load(f)
        classes = model_info['classes']
    
    print(f"Loaded {len(classes)} classes")
    print("First 10 classes:", classes[:10])
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera!")
        return
    
    print("Camera opened. Press 'q' to quit.")
    print("Testing with basic educational model...")
    
    confidence_threshold = 0.6
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess
        image = cv2.resize(frame, (300, 300))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Predict
        prediction = model.predict(image, verbose=0)
        
        # Handle different model output formats
        if isinstance(prediction, list) and len(prediction) == 2:
            # Transfer learning model with [class_probs, bbox_coords]
            class_probs, bbox_coords = prediction
        else:
            # Educational model with just class probabilities
            class_probs = prediction
            
        print(f"Class probs shape: {class_probs.shape}")  # Debug
        
        # Get class with highest confidence (skip background if it exists)
        start_idx = 1 if len(classes) > 0 and classes[0] in ['-', 'background'] else 0
        
        if class_probs.shape[1] > start_idx:
            class_idx = np.argmax(class_probs[0][start_idx:]) + start_idx
            confidence = class_probs[0][class_idx]
        else:
            class_idx = 0
            confidence = 0.0
        
        display_frame = frame.copy()
        
        if confidence > confidence_threshold:
            class_name = classes[class_idx] if class_idx < len(classes) else f"Class_{class_idx}"
            
            # Draw bounding box (center of image as example)
            h, w = frame.shape[:2]
            x1, y1 = w//4, h//4
            x2, y2 = 3*w//4, 3*h//4
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(display_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            print(f"Detected: {class_name} (confidence: {confidence:.2f})")
        
        # Instructions
        cv2.putText(display_frame, "Press 'q' to quit", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Label Test', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed.")

if __name__ == "__main__":
    main()