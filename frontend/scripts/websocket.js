import { isCameraRunning } from './camera.js';
import { updateDetections } from './objectSearch.js';

let ws = null;
export let isStreaming = false;

// Create canvas for frame capture
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

// Frame rate limiting for WebSocket sending
const MAX_FPS = 30;
const FRAME_INTERVAL = 1000 / MAX_FPS; // ~33ms between frames
let lastFrameTime = 0;

// Get detection canvas and info elements
const detectionCanvas = document.getElementById('detectionCanvas');
const detectionCtx = detectionCanvas.getContext('2d');
const detectionInfo = document.getElementById('detectionInfo');

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
    try {
      const data = JSON.parse(event.data);

      console.log('Full message data:', data);

      if (data.type === 'processed_frame') {
        // Display processed frame with detections
        displayProcessedFrame(data);
      } else {
        console.log('Received from server:', data);
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  };
}


export const captureAndSendFrame = (videoElement) => {
  if (!isStreaming || !videoElement.videoWidth || !videoElement.videoHeight) {
    return;
  }

  // Frame rate limiting - only send if enough time has passed
  const currentTime = Date.now();
  if (currentTime - lastFrameTime < FRAME_INTERVAL) {
    return; // Skip this frame to maintain max 30 FPS
  }
  lastFrameTime = currentTime;

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

function displayProcessedFrame(data) {
  // console.log('Received processed frame data:', {
  //   frameLength: data.frame ? data.frame.length : 0,
  //   detectionCount: data.detection_count,
  //   fps: data.fps
  // });

  if (!data.frame) {
    console.error('No frame data received');
    return;
  }

  // Create an image element to load the base64 frame
  const img = new Image();

  img.onload = () => {
    // console.log('Image loaded successfully, drawing to canvas');
    // Clear the detection canvas
    detectionCtx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);

    // Draw the processed frame
    detectionCtx.drawImage(img, 0, 0, detectionCanvas.width, detectionCanvas.height);
  };

  img.onerror = (error) => {
    console.error('Error loading image:', error);
  };

  // Set the base64 image data
  img.src = `data:image/jpeg;base64,${data.frame}`;

  // Update detection information
  updateDetectionInfo(data);

  // Update detections for object search
  if (data.detections) {
    updateDetections(data.detections);
  }

}

function updateDetectionInfo(data) {
  let infoHTML = `
    <strong>FPS:</strong> ${data.fps} |
    <strong>Objetos Detectados:</strong> ${data.detection_count}
  `;

  // if (data.detections && data.detections.length > 0) {
  //   infoHTML += '<br><br><strong>Detecções:</strong><br>';

  //   data.detections.forEach((detection, index) => {
  //     const { name, confidence, bbox } = detection;
  //     infoHTML += `
  //       <div style="margin: 5px 0; padding: 5px; background-color: rgba(0,0,0,0.1); border-radius: 3px;">
  //         <strong>${name}</strong> (${(confidence * 100).toFixed(1)}%)<br>
  //         <small>Posição: x1=${bbox.x1}, y1=${bbox.y1}, x2=${bbox.x2}, y2=${bbox.y2}</small>
  //       </div>
  //     `;
  //   });
  // }

  detectionInfo.innerHTML = infoHTML;
}


initWebSocket()
