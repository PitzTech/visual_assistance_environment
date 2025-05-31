import { generateAndPlaySpeech } from './audio.js';
import { hands } from './recordGesture.js';
import { captureAndSendFrame } from './websocket.js';
import { FPSController } from './fpsControllerClass.js';

export let isCameraRunning = false;

const fpsController = new FPSController(30);

const videoDomElement = document.getElementById('inputVideo');

function startCamera() {
  if (isCameraRunning) return;

  const camera = new Camera(videoDomElement, {
    onFrame: async () => {
      await hands.send({ image: videoDomElement });
    },
    width: 320,
    height: 240
  });

  camera.start();

  fpsController.start(() => {
    if (isStreaming) {
      captureAndSendFrame();
    }
  });

  isCameraRunning = true;
  generateAndPlaySpeech('Olá, bem-vindo ao sistema de reconhecimento de gestos!');
}

//câmera do celular (Precisa do HTTPS para funcionar)
const constraints = {
  audio: false,
  video: {
    facingMode: { ideal: "environment" } // Use "environment" para traseira e "user" para frontal
  }
};

navigator.mediaDevices.getUserMedia(constraints)
  .then((stream) => {
    videoDomElement.srcObject = stream;
    videoDomElement.play();
  })
  .catch((error) => {
    console.error("Erro ao acessar a câmera:", error);
    alert("Não foi possível acessar a câmera. Verifique as permissões.");
  });
//câmera do celular

startCamera()


