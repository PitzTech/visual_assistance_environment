import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import json

# Configure GPU usage - Optimized for RTX 4080 Super
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) available: {len(gpus)}")
        print(f"GPU names: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU available, using CPU")

class ObjectDetectionSystem:
    """
    Sistema de detecção de objetos em ambientes educacionais usando modelo treinado
    para inclusão de pessoas com deficiência visual.
    """

    def __init__(self):
        # Configurações
        self.confidence_threshold = 0.3
        self.input_size = (300, 300)
        
        # Carregar modelo treinado e informações
        self.model = None
        self.classes = []
        self.class_mapping = {}
        self._load_trained_model()
        
        print(f"Sistema inicializado com {len(self.classes)} classes!")

    def _load_trained_model(self):
        """
        Carrega o modelo treinado e suas informações
        """
        model_path = "models/educational_objects/final_model.h5"
        model_info_path = "models/educational_objects/model_info.json"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em {model_path}")
        
        # Carregar modelo
        self.model = load_model(model_path)
        print(f"Modelo carregado de {model_path}")
        
        # Carregar informações das classes
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
                self.classes = model_info['classes']
                self.class_mapping = model_info['class_mapping']
                print(f"\n=== OBJETOS DETECTÁVEIS ===")
                print(f"Total de classes: {len(self.classes)}")
                for i, class_name in enumerate(self.classes):
                    print(f"{i}: {class_name}")
                print("=" * 30)
    
    def get_available_cameras(self):
        """
        Detecta todas as câmeras disponíveis no sistema
        """
        available_cameras = []
        
        # Testar até 10 índices de câmera
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Obter informações da câmera
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
        """
        Permite ao usuário selecionar uma câmera
        """
        cameras = self.get_available_cameras()
        
        if not cameras:
            print("Nenhuma câmera encontrada!")
            return None
        
        print("\nCâmeras disponíveis:")
        for i, camera in enumerate(cameras):
            print(f"{i + 1}. {camera['name']} - {camera['resolution']} @ {camera['fps']:.1f} FPS")
        
        # Se há apenas uma câmera, selecioná-la automaticamente
        if len(cameras) == 1:
            selected_camera = cameras[0]
            print(f"\nCâmera selecionada automaticamente: {selected_camera['name']}")
            return selected_camera['index']
        
        while True:
            try:
                choice = input(f"\nEscolha uma câmera (1-{len(cameras)}) ou Enter para câmera padrão: ")
                
                # Se usuário pressionar Enter, usar primeira câmera
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
                # Em caso de EOF ou valor inválido, usar primeira câmera
                selected_camera = cameras[0]
                print(f"\nUsando câmera padrão: {selected_camera['name']}")
                return selected_camera['index']

    def _speak(self, text):
        """
        Função para fornecer feedback de áudio ao usuário
        """
        print(f"Feedback: {text}")

    def preprocess_image(self, image):
        """
        Pré-processa a imagem para alimentar o modelo
        """
        image = cv2.resize(image, self.input_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def detect_objects(self, image):
        """
        Detecta objetos na imagem usando o modelo treinado
        Retorna apenas o objeto com maior confiança e sua bounding box precisa
        """
        processed_image = self.preprocess_image(image)
        
        # Realizar previsão
        class_probs, bbox_coords = self.model.predict(processed_image, verbose=0)
        
        # Processar resultados
        detected_objects = []
        
        # Handle different model output formats
        if isinstance(class_probs, list):
            # If model returns list, take the first element
            class_probs = class_probs[0] if len(class_probs) > 0 else class_probs
        
        # Ensure we have valid class probabilities
        if len(class_probs.shape) == 1:
            class_probs = np.expand_dims(class_probs, 0)
        
        # Find class with highest probability (skip background if exists)
        start_idx = 1 if len(self.classes) > 0 and self.classes[0] in ['-', 'background'] else 0
        
        # Limit search to only the classes we have in our list
        max_class_idx = min(class_probs.shape[1], len(self.classes))
        
        if max_class_idx > start_idx:
            # Only search within our available classes
            valid_probs = class_probs[0][start_idx:max_class_idx]
            if len(valid_probs) > 0:
                class_idx = np.argmax(valid_probs) + start_idx
                max_confidence = class_probs[0][class_idx]
            else:
                return detected_objects
        else:
            return detected_objects  # No valid classes to detect
        
        # Ensure class_idx is within bounds (double check)
        if class_idx >= len(self.classes):
            print(f"Warning: class_idx {class_idx} >= len(classes) {len(self.classes)}")
            return detected_objects
        
        if max_confidence > self.confidence_threshold:
            # Handle bounding box coordinates
            if hasattr(bbox_coords, 'shape') and len(bbox_coords.shape) > 0:
                y_min, x_min, y_max, x_max = bbox_coords[0]
            else:
                # Default bounding box if coordinates are invalid
                y_min, x_min, y_max, x_max = 0.2, 0.2, 0.8, 0.8
            
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
            
            class_name = self.classes[class_idx] if class_idx < len(self.classes) else f"Class_{class_idx}"
            
            detected_objects.append({
                'name': class_name,
                'confidence': float(max_confidence),
                'coords': [
                    {'x': x_min, 'y': y_min},  # top-left
                    {'x': x_max, 'y': y_min},  # top-right
                    {'x': x_max, 'y': y_max},  # bottom-right
                    {'x': x_min, 'y': y_max}   # bottom-left
                ],
                'bbox': (x_min, y_min, x_max, y_max)
            })
        
        return detected_objects

    def get_object_position(self, bbox, image_shape):
        """
        Determina a posição do objeto na imagem
        """
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
            vertical = "fundo"  # parte de cima da imagem
        elif y_center > 2*h/3:
            vertical = "inicio"  # parte de baixo da imagem
        else:
            vertical = "meio"
        
        return f"{horizontal} {vertical}"
    
    def provide_feedback(self, detected_objects, image_shape):
        """
        Fornece feedback sobre os objetos detectados
        """
        if not detected_objects:
            self._speak("Nenhum objeto detectado.")
            return []
        
        # Ordenar objetos por confiança
        detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Preparar JSON de saída
        output_json = []
        
        for obj in detected_objects:
            position = self.get_object_position(obj['bbox'], image_shape)
            
            output_json.append({
                'name': obj['name'],
                'coords': obj['coords'],
                'position': position,
                'confidence': obj['confidence']
            })
            
            # Feedback de áudio
            self._speak(f"{obj['name']} detectado na {position} com {int(obj['confidence'] * 100)}% de certeza")
        
        return output_json

    def run_camera_detection(self, camera_index=0):
        """
        Executa a detecção em tempo real usando a câmera
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Erro ao abrir a câmera {camera_index}.")
            return
        
        print("Pressione 'q' para sair, 's' para silenciar/ativar áudio")
        
        window_name = 'Detecção de Objetos Educacionais'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        # Variáveis para controle de feedback
        last_feedback_time = 0
        feedback_cooldown = 2  # segundos entre feedbacks de áudio
        audio_enabled = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Verificar entrada do usuário
            key = cv2.waitKey(1) & 0xFF
            
            # Detectar objetos em tempo real (todo frame)
            detected_objects = self.detect_objects(frame)
            
            # Desenhar bounding boxes no frame
            display_frame = frame.copy()
            for obj in detected_objects:
                x_min, y_min, x_max, y_max = obj['bbox']
                # Usar cores diferentes para objetos diferentes
                color = (0, 255, 0)  # Verde padrão
                if 'book' in obj['name'].lower():
                    color = (255, 0, 0)  # Azul para livros
                elif 'pen' in obj['name'].lower():
                    color = (0, 0, 255)  # Vermelho para canetas
                
                cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), color, 2)
                label = f"{obj['name']}: {int(obj['confidence'] * 100)}%"
                cv2.putText(display_frame, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Fornecer feedback de áudio periodicamente
            current_time = time.time()
            if detected_objects and audio_enabled and (current_time - last_feedback_time > feedback_cooldown):
                detection_json = self.provide_feedback(detected_objects, frame.shape)
                last_feedback_time = current_time
                
                # Imprimir JSON apenas quando há feedback
                if detection_json:
                    print("\n=== OBJETOS DETECTADOS (JSON) ===")
                    print(json.dumps(detection_json, indent=2, ensure_ascii=False))
                    print("=" * 35)
            
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
            
            # Exibir o frame
            cv2.imshow(window_name, display_frame)
            
            # Sair do loop
            if key == ord('q'):
                break
        
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()


def main():
    try:
        # Criar o sistema de detecção
        detection_system = ObjectDetectionSystem()
        
        # Selecionar câmera
        camera_index = detection_system.select_camera()
        if camera_index is None:
            return
        
        # Iniciar o sistema com a câmera selecionada
        print("\n" + "="*50)
        print("SISTEMA DE DETECÇÃO DE OBJETOS EDUCACIONAIS")
        print("="*50)
        print("Este sistema auxilia pessoas com deficiência visual")
        print("a identificar objetos em tempo real.")
        print("\nO sistema fornecerá:")
        print("• Feedback visual (caixas ao redor dos objetos)")
        print("• Feedback de áudio sobre objetos detectados")
        print("• JSON com coordenadas e posições dos objetos")
        print("="*50)
        
        # Iniciar detecção
        detection_system.run_camera_detection(camera_index)
        
    except Exception as e:
        print(f"Erro: {e}")
        print("Verifique se o modelo foi treinado corretamente.")

if __name__ == "__main__":
    main()
