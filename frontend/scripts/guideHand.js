import { generateAndPlaySpeech } from './textToSpeech.js';

let detectedObjects = []; // <<== Será preenchido pelo WebSocket que recebe a IA - estou simulando por enquanto
let currentTarget = null; // <<== Será preenchido pelo WebSocket que recebe a IA - estou simulando por enquanto

// Função para calcular distância entre dois pontos
function calculateDistance(point1, point2) {
  return Math.sqrt(
    Math.pow(point1.x - point2.x, 2) +
    Math.pow(point1.y - point2.y, 2)
  );
}

// Guia o usuário até o objeto
function guideUserToObject(handLandmarks, targetObject) {
  // Centro da mão (aproximadamente)
  const handCenter = {
    x: handLandmarks[0].x * canvasElement.width,
    y: handLandmarks[0].y * canvasElement.height
  };

  // Centro do objeto alvo
  const targetCenter = {
    x: (targetObject.x + targetObject.width / 2) * canvasElement.width,
    y: (targetObject.y + targetObject.height / 2) * canvasElement.height
  };

  // Calcular direção
  const dx = targetCenter.x - handCenter.x;
  const dy = targetCenter.y - handCenter.y;
  const distance = Math.sqrt(dx * dx + dy * dy);

  // Determinar orientação
  let direction = '';
  if (Math.abs(dx) > Math.abs(dy)) {
    direction = dx > 0 ? 'direita' : 'esquerda';
  } else {
    direction = dy > 0 ? 'abaixo' : 'acima';
  }

  // Feedback para o usuário
  if (distance < 50) {
    gestureOutput.textContent = `Objeto ${targetObject.name} encontrado!`;
    generateAndPlaySpeech(`Objeto ${targetObject.name} encontrado`);
  } else {
    gestureOutput.textContent = `Mova a mão para ${direction} para alcançar ${targetObject.name}`;
    generateAndPlaySpeech(`Mova para ${direction}`);
  }

  // Desenhar linha de guia
  canvasCtx.beginPath();
  canvasCtx.moveTo(handCenter.x, handCenter.y);
  canvasCtx.lineTo(targetCenter.x, targetCenter.y);
  canvasCtx.strokeStyle = 'rgba(0, 255, 255, 0.7)';
  canvasCtx.lineWidth = 3;
  canvasCtx.stroke();

  // Desenhar círculo no alvo
  canvasCtx.beginPath();
  canvasCtx.arc(targetCenter.x, targetCenter.y, 20, 0, 2 * Math.PI);
  canvasCtx.fillStyle = 'rgba(0, 255, 255, 0.3)';
  canvasCtx.fill();
}

