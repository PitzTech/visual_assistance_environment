import { generateAndPlaySpeech } from './audio.js';
import { isCameraRunning } from './camera.js';
import { captureHandGesture, recognizeRegisteredGesture, calculateAverageGesture } from './handGestureCalculator.js';

/**
 * GENERAL VARIABLES
 */

let registeredGestures = JSON.parse(localStorage.getItem('handGestures')) || [];
let currentGestureData = null;

/**
 * UI MANAGER
 */

// DOM Elements

const gesturesList = document.getElementById('gesturesList');

// General Functions

function showStatusMessage(message, type) {
  const statusMessage = document.getElementById('statusMessage');
  statusMessage.textContent = message;
  statusMessage.className = `status-message ${type}`;

  setTimeout(() => {
    statusMessage.textContent = "";
    statusMessage.className = "";
  }, 5000);
}

// Registered Gestures Menu

const updateGesturesList = () => {
  if (!registeredGestures.length) {
    gesturesList.innerHTML = "<p>Nenhum gesto cadastrado ainda.</p>";
    return;
  }

  const html = registeredGestures.map((gesture, index) => `
                <div class="gesture-item">
                    <div class="gesture-info">
                        <div class="gesture-name">${gesture.name}</div>
                    </div>
                    <button class="btn btn-danger removeGestureBtn" data-index="${index}">Remover</button>
                </div>
            `).join('');

  gesturesList.innerHTML = html;
}

function removeGesture(index) {
  registeredGestures.splice(index, 1);
  localStorage.setItem('handGestures', JSON.stringify(registeredGestures));
  updateGesturesList();
  showStatusMessage("Gesto removido com sucesso!", "success");
}

const handleRemoveRegisteredGesture = (e) => {
  if (e.target.classList.contains('removeGestureBtn')) {
    const index = parseInt(e.target.dataset.index);
    if (!isNaN(index)) {  // Make sure we got a valid index
      removeGesture(index);
    }
  }
}

const handleRemoveAllRegisteredGestures = () => {
  if (registeredGestures.length === 0) {
    showStatusMessage("Não há gestos para remover.", "warning");
    return;
  }

  if (confirm("Tem certeza que deseja remover todos os gestos cadastrados?")) {
    registeredGestures = [];
    localStorage.removeItem('handGestures');
    updateGesturesList();
    showStatusMessage("Todos os gestos foram removidos.", "success");
  }
}

const handleRegisterGesture = () => {
  const objectNameInput = document.getElementById('objectName');
  const objectName = objectNameInput.value.trim();

  if (!objectName) {
    showStatusMessage("Por favor, preencha o nome do objeto!", "error");
    return;
  }

  if (!currentGestureData) {
    showStatusMessage("Capture um gesto primeiro!", "error");
    return;
  }

  // Verificar se já existe um gesto com o mesmo nome
  const existingGesture = registeredGestures.find(g => g.name.toLowerCase() === objectName.toLowerCase());
  if (existingGesture) {
    if (!confirm("Já existe um gesto com este nome. Deseja substituí-lo?")) {
      return;
    }
    // Remover o gesto existente
    const index = registeredGestures.indexOf(existingGesture);
    registeredGestures.splice(index, 1);
  }

  // Criar novo gesto
  const newGesture = {
    id: Date.now(),
    name: objectName,
    landmarks: currentGestureData,
    createdAt: new Date().toISOString()
  };

  registeredGestures.push(newGesture);
  localStorage.setItem('handGestures', JSON.stringify(registeredGestures));

  // Limpar formulário
  objectNameInput.value = '';
  currentGestureData = null;
  registerBtn.disabled = true;

  updateGesturesList();
  showStatusMessage(`Gesto "${objectName}" cadastrado com sucesso!`, "success");
}

// listeners

// remove gesture from list
gesturesList.addEventListener('click', handleRemoveRegisteredGesture);

// clean all gestures
const clearGesturesBtn = document.getElementById('clearGestures');
clearGesturesBtn.addEventListener('click', handleRemoveAllRegisteredGestures);

// register new gesture
const registerBtn = document.getElementById('registerGesture');
registerBtn.addEventListener('click', handleRegisterGesture);

updateGesturesList();

/**
 * Recording Manager
 */

let isCapturingGesture = false;
let captureFrames = [];
const CAPTURE_FRAMES_NEEDED = 5;

const canvasElement = document.getElementById('outputCanvas');
const canvasCtx = canvasElement.getContext('2d');

// DOM Elements

const gestureOutput = document.getElementById('gestureOutput');

const handleUpdateGestureOutput = (message) => {
  const currentMessage = gestureOutput.textContent;
  if (currentMessage == message) return

  gestureOutput.textContent = message;
}


const handleStartGestureCapture = () => {
  if (!isCameraRunning) {
    showStatusMessage("Inicie a câmera primeiro!", "error");
    return;
  }

  if (isCapturingGesture) {
    // Cancelar captura
    isCapturingGesture = false;
    captureFrames = [];
    captureBtn.textContent = "Capturar Sinal";
    captureBtn.classList.remove('capturing');
    handleUpdateGestureOutput("Captura cancelada");
    showStatusMessage("Captura cancelada.", "warning");
  } else {
    // Iniciar captura
    isCapturingGesture = true;
    captureBtn.textContent = "Cancelar Captura";
    captureBtn.classList.add('capturing');
    captureFrames = [];
    currentGestureData = null;
    registerBtn.disabled = true;
    showStatusMessage("Mantenha o gesto estável por alguns segundos...", "warning");
  }
}

function finishCapture() {
  isCapturingGesture = false;
  captureBtn.textContent = "Capturar Sinal";
  captureBtn.classList.remove('capturing');

  // Validate captured frames
  if (captureFrames.length === 0) {
    showStatusMessage("Erro: Nenhum frame foi capturado.", "error");
    return;
  }

  // Calculate average gesture with improved algorithm
  currentGestureData = calculateAverageGesture(captureFrames);

  handleUpdateGestureOutput("Gesto capturado! Agora clique em 'Cadastrar Sinal'")
  registerBtn.disabled = false;
  showStatusMessage("Gesto capturado com sucesso! Preencha o nome e clique em cadastrar.", "success");

  // Clear frames
  captureFrames = [];
}

function captureGestureFrame(landmarks) {
  const normalizedLandmarks = captureHandGesture(landmarks);
  captureFrames.push(normalizedLandmarks);

  // Show capture progress with gesture info
  const features = normalizedLandmarks.gestureFeatures;
  const gestureInfo = features ?
    ` (${features.extendedFingerCount} dedos estendidos)` : '';

  handleUpdateGestureOutput(`Capturando... ${captureFrames.length}/${CAPTURE_FRAMES_NEEDED}${gestureInfo}`);

  if (captureFrames.length >= CAPTURE_FRAMES_NEEDED) {
    finishCapture();
  }
}

// Configuração do MediaPipe Hands
export const hands = new Hands({
  locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
  }
});

hands.setOptions({
  maxNumHands: 1, // otimizado para 1 mão (2 - duas mãos)
  modelComplexity: 1, //modelo 1 mais preciso e mais pesado e modelo 0 um pouco mais leve e menos preciso
  minDetectionConfidence: 0.85, //pode ser ajustado para 0,8 ou mais caso a detecção fique instável com o modelo 0
  minTrackingConfidence: 0.85
});

hands.onResults(handleAIGestureOutputs);

function handleAIGestureOutputs(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  // Desenhar a imagem da câmera
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const landmarks = results.multiHandLandmarks[0];

    // Desenhar as conexões e pontos de referência da mão
    drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
      color: '#00FF00',
      lineWidth: 4
    });
    drawLandmarks(canvasCtx, landmarks, {
      color: '#FF0000',
      lineWidth: 2,
      radius: (data) => {
        // Função para cálculo de distância (Linear Interpolation)
        const lerp = (value, inMin, inMax, outMin, outMax) => outMin + (outMax - outMin) * ((value - inMin) / (inMax - inMin))
        return lerp(data.from.z, -0.15, 0.1, 5, 1);
      }
    });

    // Se estiver capturando gesto
    if (isCapturingGesture) {
      captureGestureFrame(landmarks);
    } else {
      // Reconhecer gestos cadastrados
      const recognizedGesture = recognizeRegisteredGesture(landmarks, registeredGestures);
      if (recognizedGesture) {
        handleUpdateGestureOutput(`Gesto detectado: ${recognizedGesture.name}`);
      } else {
        handleUpdateGestureOutput("Gesto Não Reconhecido");
      }
    }
  } else if (!isCapturingGesture) {
    handleUpdateGestureOutput("Nenhuma mão detectada");
  }

  generateAndPlaySpeech(gestureOutput.textContent);

  canvasCtx.restore();
}

const captureBtn = document.getElementById('captureGesture');
captureBtn.addEventListener('click', handleStartGestureCapture);
