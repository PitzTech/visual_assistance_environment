'''
Simple camera test to check if camera works
'''

import cv2
import sys

def test_camera():
    print("Testing camera access...")
    
    for i in range(3):
        print(f"Trying camera index {i}...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            print(f"Camera {i} opened successfully!")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"Camera {i} frame read successfully! Shape: {frame.shape}")
                
                # Show a test window for 3 seconds
                cv2.imshow(f'Camera {i} Test', frame)
                cv2.waitKey(3000)  # Wait 3 seconds
                cv2.destroyAllWindows()
                
                cap.release()
                print(f"Camera {i} works! Use index {i} for detection.")
                return i
            else:
                print(f"Camera {i} opened but couldn't read frame.")
        else:
            print(f"Camera {i} failed to open.")
        
        cap.release()
    
    print("No working camera found.")
    return None

if __name__ == "__main__":
    working_camera = test_camera()
    if working_camera is not None:
        print(f"\nWorking camera found at index: {working_camera}")
        print(f"Run your detection with camera index {working_camera}")
    else:
        print("\nNo working cameras detected.")
        print("Make sure:")
        print("1. Camera is connected")
        print("2. Camera permissions are correct")
        print("3. No other applications are using the camera")