import { isCameraRunning } from './camera.js';

let ws = null;
let isStreaming = false;

// Create canvas for frame capture
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

function initWebSocket() {
  ws = new WebSocket('ws://localhost:8765'); // Adjust URL as needed

  ws.onopen = () => {
    console.log('WebSocket connected');
    isStreaming = true;
  };

  ws.onclose = () => {
    console.log('WebSocket disconnected');
    isStreaming = false;
    // Attempt to reconnect after 3 seconds
    setTimeout(() => {
      if (isCameraRunning) {
        initWebSocket();
      }
    }, 3000);
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
  };

  ws.onmessage = (event) => {
    // Handle messages from Python server if needed
    console.log('Received from server:', event.data);
  };
}


export const captureAndSendFrame = (videoElement) => {
  if (!isStreaming || !videoElement.videoWidth || !videoElement.videoHeight) {
    return;
  }

  // Set canvas dimensions to match video
  canvas.width = videoElement.videoWidth;
  canvas.height = videoElement.videoHeight;

  // Draw current video frame to canvas
  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

  // Convert to blob and send via WebSocket
  canvas.toBlob((blob) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(blob);
    }
  }, 'image/jpeg', 0.7); // Adjust quality as needed (0.8 = 80% quality)
}


initWebSocket()
