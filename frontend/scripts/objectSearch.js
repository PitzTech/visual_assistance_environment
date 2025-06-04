import { generateAndPlaySpeech } from './audio.js';

let currentHandPosition = null;
let currentDetections = null;
let activeTracking = new Map(); // Track which gestures are actively being searched
export let isTrackingActive = false;

// Get DOM elements
const positionMessage = document.getElementById('positionMessage');
const gesturesList = document.getElementById('gesturesList');

// Text-to-speech function in Portuguese

// Check if hand is touching/overlapping with object
function isHandTouchingObject(handX, handY, objectBbox) {
  // Define hand area (approximate hand size)
  const handRadius = 30; // pixels

  // Check if hand center is within object bounding box
  const isInsideObject = handX >= objectBbox.x1 - handRadius &&
    handX <= objectBbox.x2 + handRadius &&
    handY >= objectBbox.y1 - handRadius &&
    handY <= objectBbox.y2 + handRadius;

  return isInsideObject;
}

// Calculate position relative to image
function getImageRelativePosition(handX, handY, objectCenterX, objectCenterY) {
  const positions = [];

  // Horizontal position
  if (objectCenterX > handX + 50) {
    positions.push('direita');
  } else if (objectCenterX < handX - 50) {
    positions.push('esquerda');
  }

  // Vertical position
  if (objectCenterY > handY + 50) {
    positions.push('baixo');
  } else if (objectCenterY < handY - 50) {
    positions.push('cima');
  }

  return positions.length > 0 ? positions.join(' e ') : 'centro';
}

// Calculate position relative to hand
function getHandRelativePosition(handX, handY, objectCenterX, objectCenterY) {
  const positions = [];

  // Horizontal position relative to hand
  if (objectCenterX > handX + 50) {
    positions.push('direita da mão');
  } else if (objectCenterX < handX - 50) {
    positions.push('esquerda da mão');
  }

  // Vertical position relative to hand
  if (objectCenterY > handY + 50) {
    positions.push('abaixo da mão');
  } else if (objectCenterY < handY - 50) {
    positions.push('acima da mão');
  }

  return positions.length > 0 ? positions.join(' e ') : 'na posição da mão';
}

// Process object search for specific gesture name
function processObjectSearch(targetGestureName, shouldSpeak = false) {
  if (!currentHandPosition) {
    const message = 'Mão não encontrada';
    positionMessage.textContent = message;
    if (shouldSpeak) generateAndPlaySpeech(message);
    return null;
  }

  if (!currentDetections || currentDetections.length === 0) {
    const message = 'Nenhum objeto detectado';
    positionMessage.textContent = message;
    if (shouldSpeak) generateAndPlaySpeech(message);
    return null;
  }

  // Find object matching the gesture name
  const targetObject = currentDetections.find(detection =>
    detection.name.toLowerCase() === targetGestureName.toLowerCase()
  );

  if (!targetObject) {
    const message = `${targetGestureName} não encontrado na imagem`;
    positionMessage.textContent = message;
    if (shouldSpeak) generateAndPlaySpeech(message);
    return null;
  }

  // Calculate object center from bbox
  const objectCenterX = (targetObject.bbox.x1 + targetObject.bbox.x2) / 2;
  const objectCenterY = (targetObject.bbox.y1 + targetObject.bbox.y2) / 2;

  // Get hand position (convert from normalized coordinates to canvas coordinates)
  const canvasWidth = 640;
  const canvasHeight = 480;
  const handX = currentHandPosition.x * canvasWidth;
  const handY = currentHandPosition.y * canvasHeight;

  // Check if hand is touching the object
  const isTouching = isHandTouchingObject(handX, handY, targetObject.bbox);

  // Calculate positions (always calculate for return object)
  const imagePosition = getImageRelativePosition(handX, handY, objectCenterX, objectCenterY);
  const handPosition = getHandRelativePosition(handX, handY, objectCenterX, objectCenterY);
  const confidence = (targetObject.confidence * 100).toFixed(0);

  let message;
  if (isTouching) {
    message = `Sua mão está no objeto ${targetGestureName}`;
  } else {
    // Create regular position message
    message = `${targetGestureName} encontrado (${confidence}%) - Posição na imagem: ${imagePosition}. Posição relativa: ${handPosition}`;
  }

  if (shouldSpeak && message.length) generateAndPlaySpeech(message);

  // Display the message
  positionMessage.textContent = message;

  return {
    object: targetGestureName,
    confidence: confidence,
    objectCenter: { x: objectCenterX, y: objectCenterY },
    handPosition: { x: handX, y: handY },
    isTouching: isTouching,
    imagePosition: imagePosition,
    handRelativePosition: handPosition
  };
}

// Continuously update position for active tracking
function updateActiveTracking() {
  if (activeTracking.size === 0) return;

  for (const [gestureName, trackingData] of activeTracking) {
    const result = processObjectSearch(gestureName, true);

    // Update tracking data if needed
    if (result) {
      trackingData.lastFound = Date.now();
    }
  }
}

// Handle procurar button click
function handleProcurarClick(event) {
  if (event.target.classList.contains('procurarBtn')) {
    const gestureName = event.target.dataset.gestureName;

    if (activeTracking.has(gestureName)) {
      // Stop tracking
      activeTracking.delete(gestureName);
      event.target.textContent = 'Procurar';
      event.target.classList.remove('tracking');
      positionMessage.textContent = '';
      isTrackingActive = false
    } else {
      isTrackingActive = true
      // Start tracking
      activeTracking.set(gestureName, {
        startTime: Date.now(),
        lastFound: null
      });
      event.target.textContent = 'Parar Buscar';
      event.target.classList.add('tracking');

      // Initial search with speech
      processObjectSearch(gestureName, true);
    }
  }
}

// Update hand position from MediaPipe results
export function updateHandPosition(landmarks) {
  if (landmarks && landmarks.length > 0) {
    // Use wrist position (landmark 0) as reference
    currentHandPosition = {
      x: landmarks[0].x,
      y: landmarks[0].y,
      z: landmarks[0].z
    };
  }
}

// Update detection data from WebSocket
export function updateDetections(detections) {
  currentDetections = detections;

  // Update active tracking when new detection data arrives
  updateActiveTracking();
}

// Initialize event listeners
function initObjectSearch() {
  // Handle procurar button clicks on gesture items
  gesturesList.addEventListener('click', handleProcurarClick);
}

// Initialize the module
initObjectSearch();
