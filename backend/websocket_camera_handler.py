import asyncio
import websockets
from PIL import Image
import io
import numpy as np
import cv2
import json
import logging
import time
import base64
import threading
from queue import Queue, Empty

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketCameraHandler:
    """
    Unified WebSocket camera stream handler that combines server functionality
    with VideoCapture-like interface for frame provision
    """

    def __init__(self):
        # WebSocket server properties
        self.frame_count = 0
        self.clients = set()

        # FPS calculation variables
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.fps = 0
        self.target_fps = 30
        self.frame_time = 1.0 / self.target_fps
        self.last_frame_time = time.time()

        # Screenshot saving
        self.save_screenshot = False

        # Frame provider properties
        self.frame_queue = Queue(maxsize=5)
        self.is_open = False
        self.latest_frame = None
        self.server = None

        # Detection results storage
        self.latest_processed_frame = None
        self.latest_detections = []

    async def register_client(self, websocket):
        """Register a new client connection"""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")

    async def unregister_client(self, websocket):
        """Unregister a client connection"""
        self.clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")

    async def process_frame(self, frame_data):
        """Process the received frame data and return unified message"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(frame_data))

            # Convert to numpy array for OpenCV processing
            frame = np.array(image)

            # Convert RGB to BGR for OpenCV (if needed)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Flip frame horizontally for mirror effect
            # frame = cv2.flip(frame, 1)

            # Update latest frame for frame provider functionality
            self.latest_frame = frame.copy()

            # Put in queue (non-blocking) for frame provider functionality
            try:
                self.frame_queue.put_nowait(frame.copy())
            except:
                pass  # Queue full, skip frame

            # Calculate FPS
            self.fps_counter += 1
            if time.time() - self.fps_timer >= 1.0:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.fps_timer = time.time()

            self.frame_count += 1

            logger.info(f"Processed frame {self.frame_count}: {image.size}, FPS: {self.fps}")

            # Create unified message with all data
            message = {
                "type": "processed_frame",
                "frame_count": self.frame_count,
                "image_size": list(image.size),
                "fps": self.fps,
                "status": "processed"
            }

            # Add detection data if available
            if self.latest_processed_frame is not None and self.latest_detections:
                # Convert processed frame to base64
                success, buffer = cv2.imencode('.jpg', self.latest_processed_frame)
                if success:
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')

                    # Prepare detection data
                    detection_data = []
                    for detection in self.latest_detections:
                        x1, y1, x2, y2 = detection['bbox']
                        detection_data.append({
                            'name': detection['class'],
                            'confidence': detection['confidence'],
                            'bbox': {
                                'x1': int(x1), 'y1': int(y1),
                                'x2': int(x2), 'y2': int(y2)
                            }
                        })

                    message.update({
                        "frame": frame_base64,
                        "detections": detection_data,
                        "detection_count": len(self.latest_detections)
                    })

                    logger.info(f"Added detection data: {len(detection_data)} detections")

            return message

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {"error": str(e)}

    def request_screenshot(self):
        """Request to save the next frame as screenshot"""
        self.save_screenshot = True
        logger.info("Screenshot requested - will save next frame")

    async def broadcast_message(self, message):
        """Send message to all connected clients"""
        if self.clients:
            await asyncio.gather(
                *[client.send(json.dumps(message)) for client in self.clients],
                return_exceptions=True
            )

    def send_processed_frame(self, processed_frame, detections):
        """Store processed frame and detections for next frame message"""
        try:
            self.latest_processed_frame = processed_frame.copy()
            self.latest_detections = detections.copy()
            # logger.info(f"Stored {len(detections)} detections for next frame")

        except Exception as e:
            logger.error(f"Error in send_processed_frame: {e}")


    async def handle_client(self, websocket, path=None):
        """Handle individual client connections"""
        await self.register_client(websocket)

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Handle image frame
                    result = await self.process_frame(message)

                    # Send result back to client (optional)
                    if result:
                        await websocket.send(json.dumps(result))

                elif isinstance(message, str):
                    # Handle text messages
                    try:
                        data = json.loads(message)
                        logger.info(f"Received text message: {data}")

                        # Handle different message types
                        if data.get("type") == "ping":
                            await websocket.send(json.dumps({"type": "pong"}))
                        elif data.get("type") == "screenshot":
                            self.request_screenshot()
                            await websocket.send(json.dumps({"type": "screenshot_requested"}))
                        elif data.get("type") == "get_stats":
                            stats = {
                                "type": "stats",
                                "frame_count": self.frame_count,
                                "current_fps": self.fps,
                                "connected_clients": len(self.clients)
                            }
                            await websocket.send(json.dumps(stats))

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON message: {message}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("Client connection closed")
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            await self.unregister_client(websocket)

    def start_websocket_server(self, host="localhost", port=8765):
        """Start WebSocket server to receive frames"""
        async def run_server():
            self.server = await websockets.serve(
                self.handle_client,
                host,
                port,
                ping_interval=20,
                ping_timeout=10,
                max_size=10 * 1024 * 1024  # 10MB max message size
            )

            self.is_open = True
            logger.info(f"WebSocket server started on ws://{host}:{port}")
            logger.info("Waiting for camera connections...")


            await self.server.wait_closed()

        # Run in background thread
        def start_server():
            asyncio.run(run_server())

        server_thread = threading.Thread(target=start_server)
        server_thread.daemon = True
        server_thread.start()

        # Wait a moment for server to start
        time.sleep(1)

    # VideoCapture-like interface methods
    def read(self):
        """
        Mimic cv2.VideoCapture.read() method
        Returns: (success, frame)
        """
        try:
            # Try to get latest frame from queue
            frame = self.frame_queue.get(timeout=0.1)
            return True, frame
        except Empty:
            # Return last known frame if no new frame available
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None

    def isOpened(self):
        """Mimic cv2.VideoCapture.isOpened() method"""
        return self.is_open

    def release(self):
        """Mimic cv2.VideoCapture.release() method"""
        self.is_open = False
        if self.server:
            self.server.close()

    def get_stats(self):
        """Get current statistics"""
        return {
            "frame_count": self.frame_count,
            "current_fps": self.fps,
            "connected_clients": len(self.clients),
            "queue_size": self.frame_queue.qsize(),
            "is_open": self.is_open
        }
