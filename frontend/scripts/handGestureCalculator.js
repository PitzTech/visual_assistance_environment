

// MediaPipe hand landmark indices for better gesture analysis
const LANDMARK_INDICES = {
  WRIST: 0,
  THUMB_TIP: 4, INDEX_TIP: 8, MIDDLE_TIP: 12, RING_TIP: 16, PINKY_TIP: 20,
  INDEX_MCP: 5, MIDDLE_MCP: 9, RING_MCP: 13, PINKY_MCP: 17,
  INDEX_PIP: 6, MIDDLE_PIP: 10, RING_PIP: 14, PINKY_PIP: 18
};

// Enhanced normalization with gesture features
function addGestureFeatures(normalizedLandmarks) {
  const features = calculateGestureFeatures(normalizedLandmarks);

  // Attach features to the landmarks array for later use
  normalizedLandmarks.gestureFeatures = features;

  return normalizedLandmarks;
}

// Calculate important gesture features
function calculateGestureFeatures(landmarks) {
  const wrist = landmarks[LANDMARK_INDICES.WRIST];

  // Calculate finger extensions
  const fingerExtensions = {
    thumb: calculateFingerExtension(landmarks, [1, 2, 3, 4]),
    index: calculateFingerExtension(landmarks, [5, 6, 7, 8]),
    middle: calculateFingerExtension(landmarks, [9, 10, 11, 12]),
    ring: calculateFingerExtension(landmarks, [13, 14, 15, 16]),
    pinky: calculateFingerExtension(landmarks, [17, 18, 19, 20])
  };

  // Calculate hand openness
  const handOpenness = calculateHandOpenness(landmarks);

  // Calculate finger spread
  const fingerSpread = calculateFingerSpread(landmarks);

  // Calculate palm orientation
  const palmOrientation = calculatePalmOrientation(landmarks);

  return {
    fingerExtensions,
    handOpenness,
    fingerSpread,
    palmOrientation,
    extendedFingerCount: Object.values(fingerExtensions).filter(ext => ext > 0.6).length
  };
}

function calculateFingerExtension(landmarks, fingerIndices) {
  const [mcp, pip, dip, tip] = fingerIndices.map(i => landmarks[i]);
  const wrist = landmarks[LANDMARK_INDICES.WRIST];

  // Distance from wrist to tip vs distance from wrist to MCP
  const tipDistance = distance3D(tip, wrist);
  const mcpDistance = distance3D(mcp, wrist);

  // Angle between finger segments
  const segment1 = subtract3D(pip, mcp);
  const segment2 = subtract3D(tip, dip);
  const angle = calculateAngle3D(segment1, segment2);

  // Combine distance ratio and angle for extension score
  const distanceRatio = tipDistance / Math.max(mcpDistance, 0.001);
  const angleScore = Math.max(0, 1 - Math.abs(angle - Math.PI) / Math.PI);

  return Math.max(0, Math.min(1, (distanceRatio - 0.8) * 2 + angleScore * 0.3));
}

function calculateHandOpenness(landmarks) {
  const wrist = landmarks[LANDMARK_INDICES.WRIST];
  const fingertips = [
    landmarks[LANDMARK_INDICES.THUMB_TIP],
    landmarks[LANDMARK_INDICES.INDEX_TIP],
    landmarks[LANDMARK_INDICES.MIDDLE_TIP],
    landmarks[LANDMARK_INDICES.RING_TIP],
    landmarks[LANDMARK_INDICES.PINKY_TIP]
  ];

  const avgTipDistance = fingertips.reduce((sum, tip) =>
    sum + distance3D(tip, wrist), 0) / fingertips.length;

  const palmSize = distance3D(wrist, landmarks[LANDMARK_INDICES.MIDDLE_MCP]);

  return Math.max(0, Math.min(2, avgTipDistance / Math.max(palmSize, 0.001)));
}

function calculateFingerSpread(landmarks) {
  const tips = [
    landmarks[LANDMARK_INDICES.INDEX_TIP],
    landmarks[LANDMARK_INDICES.MIDDLE_TIP],
    landmarks[LANDMARK_INDICES.RING_TIP],
    landmarks[LANDMARK_INDICES.PINKY_TIP]
  ];

  let totalSpread = 0;
  for (let i = 0; i < tips.length - 1; i++) {
    totalSpread += distance3D(tips[i], tips[i + 1]);
  }

  const palmWidth = distance3D(
    landmarks[LANDMARK_INDICES.INDEX_MCP],
    landmarks[LANDMARK_INDICES.PINKY_MCP]
  );

  return totalSpread / Math.max(palmWidth, 0.001);
}

function calculatePalmOrientation(landmarks) {
  const wrist = landmarks[LANDMARK_INDICES.WRIST];
  const middleMcp = landmarks[LANDMARK_INDICES.MIDDLE_MCP];
  const indexMcp = landmarks[LANDMARK_INDICES.INDEX_MCP];
  const pinkyMcp = landmarks[LANDMARK_INDICES.PINKY_MCP];

  // Vectors for palm normal calculation
  const v1 = subtract3D(middleMcp, wrist);
  const v2 = subtract3D(indexMcp, pinkyMcp);

  // Cross product for palm normal
  return crossProduct3D(v1, v2);
}

// Enhanced similarity calculation using multiple features
function calculateGestureSimilarity(landmarks1, landmarks2) {
  // Position-based similarity (your original method)
  let positionDistance = 0;
  for (let i = 0; i < 21; i++) {
    const dist = distance3D(landmarks1[i], landmarks2[i]);
    positionDistance += dist;
  }
  positionDistance /= 21;

  // Feature-based similarity
  let featureDistance = 0;
  if (landmarks1.gestureFeatures && landmarks2.gestureFeatures) {
    const features1 = landmarks1.gestureFeatures;
    const features2 = landmarks2.gestureFeatures;

    // Compare finger extensions
    const fingerDiff = Object.keys(features1.fingerExtensions).reduce((sum, finger) => {
      return sum + Math.abs(features1.fingerExtensions[finger] - features2.fingerExtensions[finger]);
    }, 0) / 5;

    // Compare hand openness
    const opennessDiff = Math.abs(features1.handOpenness - features2.handOpenness);

    // Compare finger spread
    const spreadDiff = Math.abs(features1.fingerSpread - features2.fingerSpread);

    // Compare extended finger count
    const countDiff = Math.abs(features1.extendedFingerCount - features2.extendedFingerCount) / 5;

    featureDistance = (fingerDiff * 0.4 + opennessDiff * 0.3 + spreadDiff * 0.2 + countDiff * 0.1);
  }

  // Weighted combination of position and feature distances
  return positionDistance * 0.6 + featureDistance * 0.4;
}

// Filter out frames that are too different from the median
function filterOutlierFrames(frames) {
  if (frames.length <= 3) return frames;

  // Calculate frame-to-frame distances
  const frameDistances = [];
  for (let i = 0; i < frames.length; i++) {
    let totalDist = 0;
    for (let j = 0; j < frames.length; j++) {
      if (i !== j) {
        totalDist += calculateGestureSimilarity(frames[i], frames[j]);
      }
    }
    frameDistances.push({ index: i, avgDistance: totalDist / (frames.length - 1) });
  }

  // Sort by distance and remove outliers
  frameDistances.sort((a, b) => a.avgDistance - b.avgDistance);
  const keepCount = Math.max(3, Math.floor(frames.length * 0.8));

  return frameDistances.slice(0, keepCount).map(fd => frames[fd.index]);
}

// Utility functions
function distance3D(p1, p2) {
  return Math.sqrt(
    Math.pow(p1.x - p2.x, 2) +
    Math.pow(p1.y - p2.y, 2) +
    Math.pow(p1.z - p2.z, 2)
  );
}

function subtract3D(p1, p2) {
  return {
    x: p1.x - p2.x,
    y: p1.y - p2.y,
    z: p1.z - p2.z
  };
}

function calculateAngle3D(v1, v2) {
  const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
  const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);

  if (mag1 === 0 || mag2 === 0) return 0;

  return Math.acos(Math.max(-1, Math.min(1, dot / (mag1 * mag2))));
}

function crossProduct3D(v1, v2) {
  return {
    x: v1.y * v2.z - v1.z * v2.y,
    y: v1.z * v2.x - v1.x * v2.z,
    z: v1.x * v2.y - v1.y * v2.x
  };
}

// Enhanced average calculation with outlier filtering
export function calculateAverageGesture(frames) {
  if (frames.length === 0) return [];

  // Filter out potential outliers
  const filteredFrames = filterOutlierFrames(frames);

  const avgGesture = [];
  for (let i = 0; i < 21; i++) {
    let sumX = 0, sumY = 0, sumZ = 0;

    filteredFrames.forEach(frame => {
      sumX += frame[i].x;
      sumY += frame[i].y;
      sumZ += frame[i].z;
    });

    avgGesture.push({
      x: sumX / filteredFrames.length,
      y: sumY / filteredFrames.length,
      z: sumZ / filteredFrames.length
    });
  }

  // Add features to the average gesture
  return addGestureFeatures(avgGesture);
}

export function captureHandGesture(landmarks) {
  if (!landmarks || landmarks.length < 21) return [];

  const wrist = landmarks[0];
  const middleFingerTip = landmarks[12]; // Middle finger tip

  // Calculate hand span (wrist to middle finger tip)
  const handSpan = Math.sqrt(
    Math.pow(middleFingerTip.x - wrist.x, 2) +
    Math.pow(middleFingerTip.y - wrist.y, 2) +
    Math.pow(middleFingerTip.z - wrist.z, 2)
  );

  if (handSpan === 0) return landmarks;

  const normalized = landmarks.map(point => ({
    x: (point.x - wrist.x) / handSpan,
    y: (point.y - wrist.y) / handSpan,
    z: (point.z - wrist.z) / handSpan
  }));

  // Add gesture features for better recognition
  const result = addGestureFeatures(normalized);
  return result
}

export function recognizeRegisteredGesture(landmarks, registeredGestures) {
  const normalizedLandmarks = captureHandGesture(landmarks);
  let bestMatch = null;
  let bestScore = Infinity;
  const threshold = 0.35; // Adjusted threshold for better accuracy

  registeredGestures.forEach(gesture => {
    const score = calculateGestureSimilarity(normalizedLandmarks, gesture.landmarks);

    if (score < threshold && score < bestScore) {
      bestScore = score;
      bestMatch = gesture;
    }
  });

  return bestMatch;
}

