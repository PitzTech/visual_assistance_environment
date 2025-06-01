'''
Transfer Learning Object Detector
Updated detection system that uses the transfer learning model
Supports both COCO and educational objects
'''

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import json

class TransferLearningDetector:
    """
    Object detection system using transfer learning model
    Combines COCO and educational object detection
    """

    def __init__(self, model_path="models/transfer_learning/transfer_learning_model.h5"):
        # Configurações
        self.confidence_threshold = 0.3
        self.input_size = (300, 300)
        
        # Carregar modelo e informações
        self.model = None
        self.combined_classes = []
        self.coco_classes = []
        self.educational_classes = []
        self.class_mapping = {}
        
        self._load_transfer_model(model_path)
        
        print(f"Transfer Learning Detector inicializado!")
        print(f"Total de classes: {len(self.combined_classes)}")

    def _load_transfer_model(self, model_path):
        """
        Carrega o modelo de transfer learning e suas informações
        """
        # Verificar se o modelo existe
        if not os.path.exists(model_path):
            print(f"Modelo de transfer learning não encontrado em {model_path}")
            print("Usando modelo educacional padrão...")
            self._load_fallback_model()
            return
        
        try:
            # Carregar modelo
            self.model = load_model(model_path)
            print(f"Modelo de transfer learning carregado de {model_path}")
            
            # Carregar informações das classes
            info_path = model_path.replace("transfer_learning_model.h5", "combined_class_info.json")
            if os.path.exists(info_path):
                with open(info_path, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                    self.combined_classes = model_info.get('combined_classes', [])
                    self.coco_classes = model_info.get('coco_classes', [])
                    self.educational_classes = model_info.get('educational_classes', [])
                    print(f"Classes carregadas: {len(self.combined_classes)}")
                    print(f"COCO classes: {len(self.coco_classes)}")
                    print(f"Educational classes: {len(self.educational_classes)}")
            else:
                print("Informações das classes não encontradas, usando classes padrão")
                self._setup_default_classes()
                
        except Exception as e:
            print(f"Erro ao carregar modelo de transfer learning: {e}")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Carregar modelo educacional padrão se transfer learning falhar"""
        fallback_path = "models/educational_objects/final_model.h5"
        if os.path.exists(fallback_path):
            self.model = load_model(fallback_path)
            print(f"Usando modelo educacional padrão de {fallback_path}")
            
            # Carregar classes educacionais
            info_path = "models/educational_objects/model_info.json"
            if os.path.exists(info_path):
                with open(info_path, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                    self.educational_classes = model_info['classes']
                    self.combined_classes = self.educational_classes
        else:
            raise FileNotFoundError("Nenhum modelo encontrado!")
    
    def _setup_default_classes(self):
        """Setup de classes padrão caso não carregue do arquivo"""
        self.coco_classes = [
            'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Carregar classes educacionais se disponível
        info_path = "models/educational_objects/model_info.json"
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
                self.educational_classes = model_info['classes']
        
        self.combined_classes = self.coco_classes + self.educational_classes

    def get_available_cameras(self):
        """Detecta todas as câmeras disponíveis no sistema"""
        available_cameras = []
        
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    available_cameras.append({
                        'index': i,
                        'name': f"Camera {i}",
                        'resolution': f"{width}x{height}",
                        'fps': fps
                    })
                cap.release()
        
        return available_cameras

    def select_camera(self):
        """Permite ao usuário selecionar uma câmera"""
        cameras = self.get_available_cameras()
        
        if not cameras:
            print("Nenhuma câmera encontrada!")
            return None
        
        print("\nCâmeras disponíveis:")
        for i, camera in enumerate(cameras):
            print(f"{i + 1}. {camera['name']} - {camera['resolution']} @ {camera['fps']:.1f} FPS")
        
        if len(cameras) == 1:
            selected_camera = cameras[0]
            print(f"\nCâmera selecionada automaticamente: {selected_camera['name']}")
            return selected_camera['index']
        
        while True:
            try:
                choice = input(f"\nEscolha uma câmera (1-{len(cameras)}) ou Enter para câmera padrão: ")
                
                if choice.strip() == "":
                    selected_camera = cameras[0]
                    print(f"Câmera padrão selecionada: {selected_camera['name']}")
                    return selected_camera['index']
                
                choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(cameras):
                    selected_camera = cameras[choice_idx]
                    print(f"Câmera selecionada: {selected_camera['name']}")
                    return selected_camera['index']
                else:
                    print("Opção inválida. Tente novamente.")
            except (ValueError, EOFError):
                selected_camera = cameras[0]
                print(f"\nUsando câmera padrão: {selected_camera['name']}")
                return selected_camera['index']

    def preprocess_image(self, image):
        """Pré-processa a imagem para alimentar o modelo"""
        image = cv2.resize(image, self.input_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def detect_objects(self, image):
        """Detecta objetos na imagem usando o modelo de transfer learning"""
        processed_image = self.preprocess_image(image)
        
        # Realizar previsão
        prediction = self.model.predict(processed_image, verbose=0)
        
        # Verificar se o modelo tem duas saídas (transfer learning) ou uma (modelo educacional)
        if isinstance(prediction, list) and len(prediction) == 2:
            # Transfer learning model com duas saídas
            class_probs, bbox_coords = prediction
        else:
            # Modelo educacional padrão
            class_probs = prediction
            # Criar bbox dummy para compatibilidade
            bbox_coords = np.array([[0.2, 0.2, 0.8, 0.8]])
        
        detected_objects = []
        
        # Ensure we have valid class probabilities
        if len(class_probs.shape) == 1:
            class_probs = np.expand_dims(class_probs, 0)
        
        # Find class with highest probability (skip background if exists)
        start_idx = 1 if len(self.combined_classes) > 0 and 'background' in str(self.combined_classes[0]).lower() else 0
        
        if class_probs.shape[1] <= start_idx:
            print(f"Warning: No valid classes to detect. Shape: {class_probs.shape}")
            return detected_objects
            
        class_idx = np.argmax(class_probs[0][start_idx:]) + start_idx
        
        # Ensure class_idx is within bounds
        if class_idx >= len(self.combined_classes):
            print(f"Warning: class_idx {class_idx} >= len(combined_classes) {len(self.combined_classes)}")
            return detected_objects
            
        max_confidence = class_probs[0][class_idx]
        
        if max_confidence > self.confidence_threshold:
            # Extrair coordenadas da bounding box
            if hasattr(bbox_coords, 'shape') and len(bbox_coords.shape) > 1:
                y_min, x_min, y_max, x_max = bbox_coords[0]
            else:
                y_min, x_min, y_max, x_max = 0.2, 0.2, 0.8, 0.8  # Default bbox
            
            # Converter para coordenadas de pixel
            x_min = int(x_min * image.shape[1])
            y_min = int(y_min * image.shape[0])
            x_max = int(x_max * image.shape[1])
            y_max = int(y_max * image.shape[0])
            
            # Garantir que as coordenadas estão dentro da imagem
            x_min = max(0, min(x_min, image.shape[1]))
            y_min = max(0, min(y_min, image.shape[0]))
            x_max = max(0, min(x_max, image.shape[1]))
            y_max = max(0, min(y_max, image.shape[0]))
            
            # Determinar se é objeto COCO ou educacional
            object_type = "COCO" if class_idx < len(self.coco_classes) else "Educational"
            
            # Get class name safely
            class_name = self.combined_classes[class_idx] if class_idx < len(self.combined_classes) else f"Class_{class_idx}"
            
            detected_objects.append({
                'name': class_name,
                'confidence': float(max_confidence),
                'type': object_type,
                'coords': [
                    {'x': x_min, 'y': y_min},
                    {'x': x_max, 'y': y_min},
                    {'x': x_max, 'y': y_max},
                    {'x': x_min, 'y': y_max}
                ],
                'bbox': (x_min, y_min, x_max, y_max)
            })
        
        return detected_objects

    def get_object_position(self, bbox, image_shape):
        """Determina a posição do objeto na imagem"""
        x_min, y_min, x_max, y_max = bbox
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        h, w = image_shape[:2]
        
        # Posição horizontal
        if x_center < w/3:
            horizontal = "esquerda"
        elif x_center > 2*w/3:
            horizontal = "direita"
        else:
            horizontal = "centro"
        
        # Posição vertical
        if y_center < h/3:
            vertical = "fundo"
        elif y_center > 2*h/3:
            vertical = "inicio"
        else:
            vertical = "meio"
        
        return f"{horizontal} {vertical}"

    def _speak(self, text):
        """Função para fornecer feedback de áudio ao usuário"""
        print(f"Feedback: {text}")

    def provide_feedback(self, detected_objects, image_shape):
        """Fornece feedback sobre os objetos detectados"""
        if not detected_objects:
            self._speak("Nenhum objeto detectado.")
            return []
        
        detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
        
        output_json = []
        
        for obj in detected_objects:
            position = self.get_object_position(obj['bbox'], image_shape)
            
            output_json.append({
                'name': obj['name'],
                'type': obj.get('type', 'Unknown'),
                'coords': obj['coords'],
                'position': position,
                'confidence': obj['confidence']
            })
            
            # Feedback de áudio
            obj_type = f" ({obj.get('type', 'Unknown')})" if obj.get('type') else ""
            self._speak(f"{obj['name']}{obj_type} detectado na {position} com {int(obj['confidence'] * 100)}% de certeza")
        
        return output_json

    def run_camera_detection(self, camera_index=0):
        """Executa a detecção em tempo real usando a câmera"""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Erro ao abrir a câmera {camera_index}.")
            return
        
        print("Pressione 'q' para sair, 's' para silenciar/ativar áudio")
        
        window_name = 'Transfer Learning - Detecção de Objetos'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        last_feedback_time = 0
        feedback_cooldown = 2
        audio_enabled = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            key = cv2.waitKey(1) & 0xFF
            
            # Detectar objetos em tempo real
            detected_objects = self.detect_objects(frame)
            
            # Desenhar bounding boxes
            display_frame = frame.copy()
            for obj in detected_objects:
                x_min, y_min, x_max, y_max = obj['bbox']
                
                # Cores diferentes para tipos de objetos
                if obj.get('type') == 'COCO':
                    color = (255, 0, 0)  # Azul para COCO
                else:
                    color = (0, 255, 0)  # Verde para educacionais
                
                cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), color, 2)
                label = f"{obj['name']} ({obj.get('type', 'Unknown')}): {int(obj['confidence'] * 100)}%"
                cv2.putText(display_frame, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Feedback de áudio periodicamente
            current_time = time.time()
            if detected_objects and audio_enabled and (current_time - last_feedback_time > feedback_cooldown):
                detection_json = self.provide_feedback(detected_objects, frame.shape)
                last_feedback_time = current_time
                
                if detection_json:
                    print("\n=== OBJETOS DETECTADOS (TRANSFER LEARNING) ===")
                    print(json.dumps(detection_json, indent=2, ensure_ascii=False))
                    print("=" * 50)
            
            # Controlar áudio
            if key == ord('s'):
                audio_enabled = not audio_enabled
                status = "ativado" if audio_enabled else "silenciado"
                print(f"Áudio {status}")
            
            # Exibir instruções
            audio_status = "ON" if audio_enabled else "OFF"
            instructions = f"q: sair | s: audio ({audio_status}) | Objetos: {len(detected_objects)}"
            cv2.putText(display_frame, instructions,
                       (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Exibir classes detectáveis
            classes_info = f"Classes: COCO({len(self.coco_classes)}) + Educational({len(self.educational_classes)})"
            cv2.putText(display_frame, classes_info,
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.imshow(window_name, display_frame)
            
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        print("=== SISTEMA DE DETECÇÃO COM TRANSFER LEARNING ===")
        
        # Criar o sistema de detecção
        detection_system = TransferLearningDetector()
        
        # Imprimir classes disponíveis
        print(f"\n=== OBJETOS DETECTÁVEIS ===")
        print(f"COCO classes: {len(detection_system.coco_classes)}")
        print(f"Educational classes: {len(detection_system.educational_classes)}")
        print(f"Total classes: {len(detection_system.combined_classes)}")
        print("=" * 30)
        
        # Selecionar câmera
        camera_index = detection_system.select_camera()
        if camera_index is None:
            return
        
        # Iniciar detecção
        detection_system.run_camera_detection(camera_index)
        
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()