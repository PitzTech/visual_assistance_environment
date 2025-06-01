import cv2
import numpy as np
from ultralytics import YOLO
import time

class ClassroomObjectDetector:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """
        Initialize the classroom object detector

        Args:
            model_path: Path to YOLO model options:
                       - 'yolov8n.pt' (nano - fastest, least accurate)
                       - 'yolov8s.pt' (small - balanced)
                       - 'yolov8m.pt' (medium - good accuracy)
                       - 'yolov8l.pt' (large - high accuracy)
                       - 'yolov8x.pt' (extra large - highest accuracy)
                       - 'yolo11n.pt' (YOLO11 nano - latest version)
                       - 'yolo11s.pt' (YOLO11 small - latest version)
            confidence_threshold: Minimum confidence for detections
        """
        """
        Initialize the classroom object detector

        Args:
            model_path: Path to YOLO model (will download if not exists)
            confidence_threshold: Minimum confidence for detections
        """
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

        # Complete COCO dataset class names (80 classes)
        self.coco_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
            25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
            39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
            44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
            49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
            54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
            59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
            64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
            74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
            79: 'toothbrush'
        }

        # Classroom-specific items that COCO can detect
        self.classroom_items = {
            'person': 0,           # Students, teachers
            'chair': 56,           # Classroom chairs
            'dining table': 60,    # Desks/tables
            'laptop': 63,          # Computers
            'mouse': 64,           # Computer mouse
            'keyboard': 66,        # Computer keyboard
            'cell phone': 67,      # Mobile phones
            'book': 73,            # Textbooks, notebooks
            'clock': 74,           # Wall clocks
            'scissors': 76,        # School supplies
            'backpack': 24,        # Student bags
            'handbag': 26,         # Teacher bags
            'bottle': 39,          # Water bottles
            'cup': 41,             # Coffee/tea cups
            'apple': 47,           # Snacks/lunch
            'banana': 46,          # Snacks
            'tv': 62,              # Smart boards/displays
        }

        # Note: COCO doesn't have specific classes for:
        # - pencil, pen, rubber/eraser, sharpener, pencil bag, notebook
        # These would require custom training or a different dataset
        # You could train a custom YOLO model with these additional classes

        # Enhanced colors for classroom items
        self.colors = {
            'person': (255, 144, 30),      # Orange
            'chair': (255, 178, 50),       # Light orange
            'dining table': (0, 255, 0),   # Green
            'laptop': (255, 0, 255),       # Magenta
            'book': (0, 255, 255),         # Cyan
            'cell phone': (255, 255, 0),   # Yellow
            'clock': (128, 0, 128),        # Purple
            'keyboard': (255, 165, 0),     # Orange
            'mouse': (0, 128, 255),        # Light blue
            'scissors': (255, 20, 147),    # Deep pink
            'backpack': (50, 205, 50),     # Lime green
            'handbag': (255, 69, 0),       # Red orange
            'bottle': (0, 191, 255),       # Deep sky blue
            'cup': (139, 69, 19),          # Saddle brown
            'apple': (255, 0, 0),          # Red
            'banana': (255, 255, 0),       # Yellow
            'tv': (75, 0, 130),            # Indigo
            'default': (0, 255, 0)         # Default green
        }

    def detect_objects(self, frame):
        """
        Detect objects in a frame

        Args:
            frame: Input frame from webcam

        Returns:
            frame: Frame with bounding boxes and labels
            detections: List of detected objects
        """
        # Run YOLO inference
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        detections = []

        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Get confidence and class
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])

                    # Get class name
                    class_name = self.model.names[class_id]

                    # Store detection
                    detection = {
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2)
                    }
                    detections.append(detection)

                    # Draw bounding box
                    color = self.colors.get(class_name, self.colors['default'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label with confidence
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

                    # Background rectangle for text
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                                (x1 + label_size[0], y1), color, -1)

                    # Text
                    cv2.putText(frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame, detections

    def add_info_panel(self, frame, detections, fps):
        """
        Add information panel to the frame

        Args:
            frame: Input frame
            detections: List of detected objects
            fps: Current FPS

        Returns:
            frame: Frame with info panel
        """
        height, width = frame.shape[:2]

        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Add title
        cv2.putText(frame, "Classroom Object Detection", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Count objects
        object_counts = {}
        for detection in detections:
            class_name = detection['class']
            object_counts[class_name] = object_counts.get(class_name, 0) + 1

        # Display object counts
        y_offset = 80
        cv2.putText(frame, f"Detected Objects: {len(detections)}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        y_offset += 20
        for obj_class, count in list(object_counts.items())[:3]:  # Show top 3
            cv2.putText(frame, f"{obj_class}: {count}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 15

        # Add instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to save screenshot", (20, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame

    def run_detection(self, camera_id=0):
        """
        Run real-time object detection

        Args:
            camera_id: Camera device ID (default: 0)
        """
        print(f"Starting webcam (Camera ID: {camera_id})...")
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # FPS calculation and limiting
        fps_counter = 0
        fps_timer = time.time()
        fps = 0
        target_fps = 30
        frame_time = 1.0 / target_fps
        last_frame_time = time.time()

        print("Detection started! Press 'q' to quit, 's' to save screenshot")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Detect objects
            frame, detections = self.detect_objects(frame)

            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_timer = time.time()

            # Add info panel
            frame = self.add_info_panel(frame, detections, fps)

            # Display frame
            cv2.imshow('Classroom Object Detection', frame)

            # FPS limiting
            current_time = time.time()
            elapsed_time = current_time - last_frame_time
            if elapsed_time < frame_time:
                time.sleep(frame_time - elapsed_time)
            last_frame_time = time.time()

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"classroom_detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved as {filename}")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped.")

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
            model_path='yolov8m.pt',  # Recommended: balanced speed/accuracy
            # Alternative options:
            # model_path='yolov8n.pt',  # Fastest (for low-end hardware)
            # model_path='yolov8m.pt',  # Better accuracy
            #model_path='trained_model/best.pt',  # Custom fine-tuned model
            confidence_threshold=0.8  # Lowered for better detection of classroom items
        )

        # Start detection
        detector.run_detection(camera_id=0)

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
