from classroom_object_detector import ClassroomObjectDetector

def main():
    """
    Main function to run the classroom object detector
    """
    print("Classroom Object Detection System")
    print("=" * 40)
    print("COCO Dataset can detect these classroom items:")
    print("✓ person, chair, dining table (desk), laptop")
    print("✓ book, backpack, handbag, scissors, clock")
    print("✓ mouse, keyboard, cell phone, bottle, cup")
    print("✓ tv (smart board), apple, banana")
    print()
    print("Items NOT in COCO (need custom training):")
    print("✗ pencil, pen, rubber/eraser, sharpener")
    print("✗ pencil bag, notebook (specific types)")
    print("=" * 40)

    try:
        # Initialize detector with recommended settings
        detector = ClassroomObjectDetector(
            model_path='yolo11m.pt',  # Recommended: balanced speed/accuracy
            # Alternative options:
            # model_path='yolov8n.pt',  # Fastest (for low-end hardware)
            # model_path='yolov8m.pt',  # Better accuracy
            #model_path='trained_model/best.pt',  # Custom fine-tuned model
            confidence_threshold=0.8  # Lowered for better detection of classroom items
        )

        # Start detection
        detector.run_detection(use_websocket=True, camera_id=0)

    except KeyboardInterrupt:
        print("\nDetection interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a webcam connected and the required packages installed:")
        print("pip install ultralytics opencv-python")
        print("\nFor custom classroom items (pencil, pen, etc.), you would need:")
        print("1. Create a custom dataset with these objects")
        print("2. Train a custom YOLO model")
        print("3. Or use a specialized educational object detection model")

if __name__ == "__main__":
    main()
