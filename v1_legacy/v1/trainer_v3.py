'''
Este código serve para treinar uma inteligencia artificial cujo objetivo é Identificação de Objetos em Ambientes Educacionais para Inclusão de Pessoas com Deficiência Visual.
A arquitetura deve ser  MobileNetSSD (MobileNet + SSD – Single Shot Multibox Detector) e
voce deve usar o TensorFlow (Keras).
A ia deve fazer o seguinte:
 - Reconhecimento de Objetos em Sala de Aula: RNC identifica objetos comuns em sala (mesa, mochila, quadro) e auxilia na navegação dos alunos.
 - Organizador Visual de Materiais Escolares: Detecta e nomeia materiais escolares via câmera (caderno, régua, lápis), ajudando na organização do espaço pessoal, e deve falar onde espacialmente aquele objeto está (direita, esquerda, fundo (parte de cima da imagem), no inicio (para de baixo da imagem)).
'''
'''
TODO:
- Data Augmentation
'''

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Reshape, Conv2D, Dropout
from tensorflow.keras.models import Model
import cv2
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
import time
import psutil
import gc

def log_gpu_info():
    """Log information about available GPUs"""
    print("=== GPU Information ===")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu}")
            try:
                # Get GPU details if available
                gpu_details = tf.config.experimental.get_device_details(gpu)
                device_name = gpu_details.get('device_name', 'Unknown')
                print(f"  Device Name: {device_name}")
                if 'nvidia' in device_name.lower() or 'geforce' in device_name.lower() or 'rtx' in device_name.lower() or 'gtx' in device_name.lower():
                    print(f"  Type: NVIDIA GPU")
                else:
                    print(f"  Type: Other GPU")
            except:
                print(f"  Details: Unable to retrieve")

        # Configure GPU memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPUs detected. Using CPU.")
    print("=======================\n")

def check_package_compatibility():
    """Check compatibility of required packages with current environment"""
    compatibility_issues = []

    try:
        import tensorflow as tf
        tf_version = tf.__version__
        print(f"✅ TensorFlow version: {tf_version}")

        # Check TensorFlow GPU support
        if tf.config.list_physical_devices('GPU'):
            print(f"✅ GPU support available")
        else:
            print(f"⚠️ No GPU detected - using CPU")

    except ImportError:
        compatibility_issues.append("TensorFlow not installed")

    try:
        import numpy as np
        np_version = np.__version__
        print(f"✅ NumPy version: {np_version}")

        # Check for NumPy compatibility
        if np.version.version >= "1.19.5":
            print(f"✅ NumPy version compatible")
    except ImportError:
        compatibility_issues.append("NumPy not installed")

    try:
        import cv2
        cv_version = cv2.__version__
        print(f"✅ OpenCV version: {cv_version}")
    except ImportError:
        compatibility_issues.append("OpenCV not installed")

    try:
        import pygame
        print(f"✅ pygame available")
    except ImportError:
        compatibility_issues.append("pygame not installed")

    if compatibility_issues:
        print(f"❌ Compatibility issues found: {', '.join(compatibility_issues)}")
        print(f"Please install missing packages using: pip install -r requirements.txt")
        return False
    else:
        print(f"✅ All packages compatible!")
        return True

# Check compatibility on import
print("🔍 Checking package compatibility...")
check_package_compatibility()

# Configure GPU usage - Optimized for RTX 4080 Super with system RAM backup
def configure_gpu_for_rtx_4080():
    """Configure GPU settings optimized for RTX 4080 Super with system RAM overflow"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for each GPU to use system RAM as backup
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Configure TensorFlow to allow system RAM usage for large models
            # This must be done BEFORE any operations create GPU contexts
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

            # Enable mixed precision for RTX 4080 Super (Tensor Cores)
            # Using compatible method with TensorFlow 2.x
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)

            # Configure for multi-threading optimization to handle RAM/GPU transfers
            tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
            tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores

            # Enable XLA compilation for better performance with mixed GPU/RAM usage
            tf.config.optimizer.set_jit(True)

            # Get GPU info - using compatible method
            try:
                # Alternative approach to get GPU name without deprecated method
                gpu_name = gpus[0].name.split('/')[-1] if gpus else "Unknown GPU"
            except:
                gpu_name = "RTX 4080 Super"

            print(f"🚀 GPU(s) configured for training: {len(gpus)}")
            print(f"📊 GPU: {gpu_name}")
            print(f"💾 Memory growth enabled (GPU + System RAM backup)")
            print(f"⚡ Mixed precision enabled for Tensor Cores")
            print(f"🔄 Multi-threading optimized for GPU/RAM transfers")
            print(f"🚀 XLA compilation enabled for performance")

            return True
        except RuntimeError as e:
            print(f"⚠️ GPU configuration error: {e}")
            print("Trying fallback configuration...")
            try:
                # Fallback: just enable memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("✅ Fallback GPU configuration successful")
                return True
            except:
                print("⚠️ Using default GPU settings...")
                return True
    else:
        print("❌ No GPU available, using CPU")
        return False

# Configure GPU on import
gpu_available = configure_gpu_for_rtx_4080()

# Memory management functions
def get_memory_usage():
    """Get current memory usage statistics including detailed GPU info"""
    process = psutil.Process()
    memory_info = process.memory_info()
    gpu_info = {
        'gpu_memory_used_mb': None,
        'gpu_memory_total_mb': None,
        'gpu_memory_free_mb': None,
        'gpu_utilization': None,
        'gpu_name': None,
        'gpu_temperature': None,
        'gpu_available': False
    }

    try:
        if gpu_available:
            # Get detailed GPU information using nvidia-ml-py
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)

                # Memory information
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_info['gpu_memory_used_mb'] = mem_info.used / (1024**2)
                gpu_info['gpu_memory_total_mb'] = mem_info.total / (1024**2)
                gpu_info['gpu_memory_free_mb'] = mem_info.free / (1024**2)

                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_info['gpu_utilization'] = util.gpu

                # GPU name
                gpu_info['gpu_name'] = pynvml.nvmlDeviceGetName(handle).decode('utf-8')

                # GPU temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_info['gpu_temperature'] = temp
                except:
                    gpu_info['gpu_temperature'] = None

                gpu_info['gpu_available'] = True

            except Exception as e:
                # Fallback: try TensorFlow method
                try:
                    gpus = tf.config.list_physical_devices('GPU')
                    if gpus:
                        gpu_info['gpu_name'] = "GPU Detected (via TensorFlow)"
                        gpu_info['gpu_available'] = True
                except:
                    pass
    except:
        pass

    # System RAM information
    virtual_mem = psutil.virtual_memory()

    return {
        'ram_used_mb': memory_info.rss / (1024**2),
        'ram_available_mb': virtual_mem.available / (1024**2),
        'ram_total_mb': virtual_mem.total / (1024**2),
        'ram_percent': virtual_mem.percent,
        **gpu_info
    }

def optimize_memory():
    """Optimize memory usage by clearing cache and garbage collection"""
    gc.collect()
    if gpu_available:
        try:
            tf.keras.backend.clear_session()
        except:
            pass

class ProgressTracker:
    """Enhanced class to track training progress with detailed time estimation and GPU monitoring"""

    def __init__(self, total_epochs, steps_per_epoch):
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.start_time = None
        self.epoch_times = []
        self.current_epoch = 0
        self.epoch_start_time = None

    def start_training(self):
        """Mark the start of training"""
        self.start_time = time.time()
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n{'='*80}")
        print(f"🚀 TRAINING STARTED at {current_time}")
        print(f"📈 Total Epochs: {self.total_epochs}")
        print(f"🔄 Steps per Epoch: {self.steps_per_epoch}")
        print(f"⚡ GPU Available: {gpu_available}")
        print(f"{'='*80}\n")

    def on_epoch_begin(self, epoch):
        """Mark the start of an epoch"""
        self.epoch_start_time = time.time()
        self.current_epoch = epoch

    def update_epoch(self, epoch, logs=None):
        """Update progress after each epoch with comprehensive system information"""
        current_time = time.time()

        if self.start_time and self.epoch_start_time:
            # Calculate times
            total_elapsed = current_time - self.start_time
            epoch_time = current_time - self.epoch_start_time
            self.epoch_times.append(epoch_time)

            # Calculate averages and estimates
            avg_epoch_time = total_elapsed / (epoch + 1)
            recent_avg = np.mean(self.epoch_times[-5:]) if len(self.epoch_times) >= 5 else avg_epoch_time
            remaining_epochs = self.total_epochs - (epoch + 1)
            estimated_remaining = recent_avg * remaining_epochs

            # Get comprehensive memory and system stats
            memory_stats = get_memory_usage()

            # Progress calculations
            progress = (epoch + 1) / self.total_epochs
            bar_length = 60
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)

            # Display comprehensive progress information
            print(f"\n{'='*120}")
            print(f"🚀 EPOCH {epoch + 1}/{self.total_epochs} COMPLETED - PROGRESS REPORT")
            print(f"{'='*120}")

            # ============ TIME INFORMATION ============
            print(f"\n⏰ TIME ANALYSIS:")
            print(f"   ⏱️  This Epoch Duration: {self._format_time(epoch_time)}")
            print(f"   🕐 Total Elapsed Time: {self._format_time(total_elapsed)}")
            print(f"   📊 Average per Epoch: {self._format_time(avg_epoch_time)}")
            print(f"   ⚡ Recent Average (last 5): {self._format_time(recent_avg)}")
            print(f"   ⏳ Time Remaining: {self._format_time(estimated_remaining)}")
            print(f"   🏁 Estimated Total Duration: {self._format_time(total_elapsed + estimated_remaining)}")

            # ETA calculation
            if remaining_epochs > 0:
                eta_time = current_time + estimated_remaining
                eta_formatted = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(eta_time))
                print(f"   🕒 Estimated Completion Time: {eta_formatted}")

            # ============ PROGRESS BAR ============
            print(f"\n📈 TRAINING PROGRESS:")
            print(f"   [{bar}] {progress*100:.1f}% ({epoch + 1}/{self.total_epochs} epochs)")
            print(f"   Epochs Remaining: {remaining_epochs}")

            # ============ GPU INFORMATION ============
            print(f"\n🎮 GPU STATUS:")
            if memory_stats['gpu_available']:
                print(f"   ✅ GPU in Use: YES")
                if memory_stats['gpu_name']:
                    print(f"   🖥️  GPU Model: {memory_stats['gpu_name']}")
                if memory_stats['gpu_memory_used_mb'] is not None:
                    total_gpu = memory_stats['gpu_memory_total_mb']
                    used_gpu = memory_stats['gpu_memory_used_mb']
                    free_gpu = memory_stats['gpu_memory_free_mb']
                    gpu_percent = (used_gpu / total_gpu) * 100 if total_gpu else 0
                    print(f"   💾 GPU Memory: {used_gpu:.0f} MB / {total_gpu:.0f} MB ({gpu_percent:.1f}% used)")
                    print(f"   🆓 GPU Free Memory: {free_gpu:.0f} MB")
                if memory_stats['gpu_utilization'] is not None:
                    print(f"   ⚡ GPU Utilization: {memory_stats['gpu_utilization']}%")
                if memory_stats['gpu_temperature'] is not None:
                    print(f"   🌡️  GPU Temperature: {memory_stats['gpu_temperature']}°C")
            else:
                print(f"   ❌ GPU in Use: NO (Using CPU)")

            # ============ SYSTEM RAM INFORMATION ============
            print(f"\n💻 SYSTEM RAM STATUS:")
            ram_total = memory_stats['ram_total_mb']
            ram_used = memory_stats['ram_used_mb']
            ram_available = memory_stats['ram_available_mb']
            ram_percent = memory_stats['ram_percent']
            print(f"   💾 Total RAM: {ram_total:.0f} MB ({ram_total/1024:.1f} GB)")
            print(f"   🔥 Used RAM: {ram_used:.0f} MB (Process: {ram_used:.0f} MB)")
            print(f"   🆓 Available RAM: {ram_available:.0f} MB ({ram_available/1024:.1f} GB)")
            print(f"   📊 RAM Usage: {ram_percent:.1f}%")

            # ============ TRAINING METRICS ============
            if logs:
                print(f"\n📊 TRAINING METRICS (Epoch {epoch + 1}):")
                
                # Loss metrics
                total_loss = logs.get('loss', 0)
                val_loss = logs.get('val_loss', 0)
                class_loss = logs.get('class_output_loss', 0)
                bbox_loss = logs.get('bbox_output_loss', 0)
                val_class_loss = logs.get('val_class_output_loss', 0)
                val_bbox_loss = logs.get('val_bbox_output_loss', 0)
                
                print(f"   📉 Total Loss: Train={total_loss:.4f}, Val={val_loss:.4f}")
                print(f"   🎯 Classification Loss: Train={class_loss:.4f}, Val={val_class_loss:.4f}")
                print(f"   📦 BBox Regression Loss: Train={bbox_loss:.4f}, Val={val_bbox_loss:.4f}")
                
                # Accuracy metrics
                if 'class_output_accuracy' in logs:
                    train_acc = logs.get('class_output_accuracy', 0)
                    val_acc = logs.get('val_class_output_accuracy', 0)
                    print(f"   ✅ Classification Accuracy: Train={train_acc:.4f} ({train_acc*100:.1f}%), Val={val_acc:.4f} ({val_acc*100:.1f}%)")
                
                # Learning rate
                lr = logs.get('lr', logs.get('learning_rate', 0))
                if lr > 0:
                    print(f"   📈 Learning Rate: {lr:.2e}")
                
                # Loss improvement analysis
                if epoch > 0 and 'val_loss' in logs:
                    try:
                        current_val_loss = logs.get('val_loss', float('inf'))
                        if hasattr(self, 'best_val_loss'):
                            if current_val_loss < self.best_val_loss:
                                improvement = self.best_val_loss - current_val_loss
                                improvement_percent = (improvement / self.best_val_loss) * 100
                                print(f"   🎉 NEW BEST! Val Loss improved by {improvement:.4f} ({improvement_percent:.1f}%)")
                                self.best_val_loss = current_val_loss
                            else:
                                degradation = current_val_loss - self.best_val_loss
                                print(f"   ⚠️  Val Loss +{degradation:.4f} from best ({self.best_val_loss:.4f})")
                        else:
                            self.best_val_loss = current_val_loss
                            print(f"   📌 First validation loss: {current_val_loss:.4f}")
                    except:
                        pass
                
                # Performance indicators
                if epoch > 0:
                    print(f"   📈 Training Progress:")
                    if 'val_loss' in logs and hasattr(self, 'prev_val_loss'):
                        loss_trend = "📈 Improving" if val_loss < self.prev_val_loss else "📉 Degrading"
                        print(f"      Loss Trend: {loss_trend}")
                    if 'val_class_output_accuracy' in logs and hasattr(self, 'prev_val_acc'):
                        acc_trend = "📈 Improving" if logs['val_class_output_accuracy'] > self.prev_val_acc else "📉 Degrading"
                        print(f"      Accuracy Trend: {acc_trend}")
                
                # Store for next comparison
                self.prev_val_loss = val_loss
                self.prev_val_acc = logs.get('val_class_output_accuracy', 0)

            # ============ EPOCH SUMMARY ============
            print(f"\n🏁 EPOCH {epoch + 1} SUMMARY:")
            if logs:
                print(f"   ⏱️  Duration: {self._format_time(epoch_time)}")
                print(f"   📊 Best Metrics This Epoch:")
                print(f"      • Total Loss: {logs.get('loss', 0):.4f} → {logs.get('val_loss', 0):.4f}")
                if 'class_output_accuracy' in logs:
                    print(f"      • Accuracy: {logs.get('class_output_accuracy', 0)*100:.1f}% → {logs.get('val_class_output_accuracy', 0)*100:.1f}%")
                
                # Performance assessment
                val_loss = logs.get('val_loss', float('inf'))
                val_acc = logs.get('val_class_output_accuracy', 0)
                
                if val_loss < 1.0 and val_acc > 0.8:
                    status = "🟢 EXCELLENT"
                elif val_loss < 2.0 and val_acc > 0.6:
                    status = "🟡 GOOD"
                elif val_loss < 3.0 and val_acc > 0.4:
                    status = "🟠 FAIR"
                else:
                    status = "🔴 NEEDS IMPROVEMENT"
                
                print(f"   📈 Performance Status: {status}")
            
            print(f"{'='*120}\n")

    def _format_time(self, seconds):
        """Format time in human readable format"""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
        elif minutes > 0:
            return f"{minutes:02d}m {secs:02d}s"
        else:
            return f"{secs:02d}s"

class ModelTrainer:
    """
    Classe para treinar o modelo MobileNetSSD para detecção de objetos educacionais
    Otimizada para RTX 4080 Super com monitoramento de progresso
    """

    def __init__(self, input_size=(300, 300), batch_size=None):
        """
        Inicializa o treinador do modelo

        Args:
            input_size: Tamanho da entrada do modelo (altura, largura)
            batch_size: Tamanho do lote para treinamento (auto-detectado se None)
        """
        self.input_size = input_size
        self.input_shape = (input_size[0], input_size[1], 3)  # Add input_shape for compatibility

        # Auto-detect optimal batch size for RTX 4080 Super
        if batch_size is None:
            if gpu_available:
                # RTX 4080 Super has 16GB VRAM - optimize batch size accordingly
                # With mixed precision, we can use larger batch sizes
                self.batch_size = 64  # Aggressive for RTX 4080 Super with mixed precision
            else:
                self.batch_size = 8  # Smaller batch for CPU
        else:
            self.batch_size = batch_size

        self.model = None
        self.progress_tracker = None

        print(f"🔧 ModelTrainer initialized:")
        print(f"   📐 Input size: {input_size}")
        print(f"   📦 Batch size: {self.batch_size}")
        print(f"   🎮 GPU available: {gpu_available}")

    def build_model(self, num_classes):
        """
        Constrói o modelo MobileNetSSD baseado em MobileNetV2

        Args:
            num_classes: Número de classes a serem detectadas (incluindo background)

        Returns:
            Modelo compilado
        """
        # Store num_classes as instance attribute
        self.num_classes = num_classes
        # Modelo base: MobileNetV2
        base_model = MobileNetV2(
            input_shape=(self.input_size[0], self.input_size[1], 3),
            include_top=False,
            weights='imagenet'
        )

        # Descongelar algumas camadas superiores para fine-tuning
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        # Adicionar camadas SSD para detecção de objetos
        x = base_model.output

        # Feature Pyramid Network simplificado para melhorar detecção em múltiplas escalas
        x1 = Conv2D(256, kernel_size=3, padding='same', activation='relu')(base_model.get_layer('block_16_project').output)
        x2 = Conv2D(256, kernel_size=3, padding='same', activation='relu')(base_model.output)

        # Global pooling para cada nível da pirâmide
        p1 = GlobalAveragePooling2D()(x1)
        p2 = GlobalAveragePooling2D()(x2)

        # Concatenar features de diferentes níveis
        x = tf.keras.layers.Concatenate()([p1, p2])

        # Camadas fully connected para classificação e regressão
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)

        # Cabeça de classificação
        class_output = Dense(num_classes, activation='softmax', name='class_output')(x)

        # Cabeça de regressão para bounding boxes (4 valores: x, y, width, height)
        bbox_output = Dense(4, activation='sigmoid', name='bbox_output')(x)

        # Criar modelo completo
        self.model = Model(inputs=base_model.input, outputs=[class_output, bbox_output])

        # Compilar o modelo
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss={
                'class_output': 'categorical_crossentropy',
                'bbox_output': 'huber'  # Huber loss é mais robusto para coordenadas
            },
            loss_weights={'class_output': 1.0, 'bbox_output': 1.0},
            metrics={'class_output': 'accuracy'}
        )

        return self.model

    def _prepare_dataset(self, annotations_file, images_dir):
        """
        Prepara o dataset para treinamento a partir de um arquivo de anotações
        Formato esperado de anotações: COCO ou similar com bounding boxes
        Otimizado para uso eficiente de memória

        Args:
            annotations_file: Caminho para arquivo de anotações (JSON)
            images_dir: Diretório contendo as imagens

        Returns:
            X_train, X_val, y_train_class, y_train_bbox, y_val_class, y_val_bbox
        """
        print(f"📂 Loading annotations from {annotations_file}...")

        # Carregar anotações
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)

        print(f"📈 Found {len(annotations.get('images', []))} images and {len(annotations.get('annotations', []))} annotations")

        images = []
        class_labels = []
        bbox_labels = []

        # Processar cada imagem e suas anotações com progress bar
        print(f"🔄 Processing annotations...")
        for item in tqdm(annotations['annotations'], desc="Loading images"):
            image_id = item['image_id']
            image_info = next((img for img in annotations['images'] if img['id'] == image_id), None)

            if not image_info:
                continue

            image_path = os.path.join(images_dir, image_info['file_name'])

            if not os.path.exists(image_path):
                continue

            # Carregar e redimensionar a imagem
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.input_size)
            image = image.astype(np.float32) / 255.0

            # Extrair classe e bounding box
            category_id = item['category_id']

            # One-hot encoding para classes
            class_vector = np.zeros(len(annotations['categories']) + 1)  # +1 para background
            class_vector[category_id] = 1.0

            # Normalizar bounding box (x, y, width, height) para [0,1]
            bbox = np.array(item['bbox'], dtype=np.float32)
            img_width = image_info['width']
            img_height = image_info['height']

            # Converter para formato [x_min, y_min, x_max, y_max] normalizado
            x_min = bbox[0] / img_width
            y_min = bbox[1] / img_height
            x_max = (bbox[0] + bbox[2]) / img_width
            y_max = (bbox[1] + bbox[3]) / img_height

            # Garantir que estão no intervalo [0,1]
            x_min = max(0, min(1, x_min))
            y_min = max(0, min(1, y_min))
            x_max = max(0, min(1, x_max))
            y_max = max(0, min(1, y_max))

            # Criar vetor de bbox (apenas 4 coordenadas)
            bbox_vector = np.array([y_min, x_min, y_max, x_max], dtype=np.float32)

            images.append(image)
            class_labels.append(class_vector)
            bbox_labels.append(bbox_vector)

        # Converter para arrays numpy
        X = np.array(images)
        y_class = np.array(class_labels)
        y_bbox = np.array(bbox_labels)

        # Dividir em conjuntos de treinamento e validação
        X_train, X_val, y_train_class, y_val_class, y_train_bbox, y_val_bbox = train_test_split(
            X, y_class, y_bbox, test_size=0.2, random_state=42
        )

        return X_train, X_val, y_train_class, y_train_bbox, y_val_class, y_val_bbox

    def prepare_roboflow_datasets(self, combined_dataset_info):
        """
        Prepara múltiplos datasets do Roboflow para treinamento

        Args:
            combined_dataset_info: Informações dos datasets combinados do RoboflowDatasetLoader

        Returns:
            X_train, X_val, y_train_class, y_train_bbox, y_val_class, y_val_bbox, num_classes
        """
        all_train_images = []
        all_train_class_labels = []
        all_train_bbox_labels = []
        all_val_images = []
        all_val_class_labels = []
        all_val_bbox_labels = []

        num_classes = len(combined_dataset_info['classes'])
        class_mapping = combined_dataset_info['class_mapping']

        print(f"Preparando {len(combined_dataset_info['datasets'])} datasets do Roboflow...")
        print(f"Total de classes: {num_classes}")
        print(f"Classes: {combined_dataset_info['classes']}")

        for dataset in tqdm(combined_dataset_info['datasets'], desc="Processando datasets"):
            print(f"\nProcessando dataset: {dataset['name']}")

            # Processar cada split do dataset
            for split_name, split_info in dataset['splits'].items():
                print(f"  Processando split: {split_name}")

                # Determinar se é treino ou validação
                is_train = split_name in ['train', 'all']

                # Carregar dados deste split
                images, class_labels, bbox_labels = self._load_split_data(
                    split_info, class_mapping, num_classes
                )

                if is_train:
                    all_train_images.extend(images)
                    all_train_class_labels.extend(class_labels)
                    all_train_bbox_labels.extend(bbox_labels)
                else:
                    all_val_images.extend(images)
                    all_val_class_labels.extend(class_labels)
                    all_val_bbox_labels.extend(bbox_labels)

        # Converter para numpy arrays
        X_train = np.array(all_train_images, dtype=np.float32)
        y_train_class = np.array(all_train_class_labels, dtype=np.float32)
        y_train_bbox = np.array(all_train_bbox_labels, dtype=np.float32)

        # Se não há dados de validação separados, criar split
        if len(all_val_images) == 0:
            print("Criando split de validação dos dados de treino...")
            X_train, X_val, y_train_class, y_val_class, y_train_bbox, y_val_bbox = train_test_split(
                X_train, y_train_class, y_train_bbox, test_size=0.2, random_state=42
            )
        else:
            X_val = np.array(all_val_images, dtype=np.float32)
            y_val_class = np.array(all_val_class_labels, dtype=np.float32)
            y_val_bbox = np.array(all_val_bbox_labels, dtype=np.float32)

        print(f"\nDataset preparado:")
        print(f"  Treino: {X_train.shape[0]} amostras")
        print(f"  Validação: {X_val.shape[0]} amostras")
        print(f"  Classes: {num_classes}")

        return X_train, X_val, y_train_class, y_train_bbox, y_val_class, y_val_bbox, num_classes

    def _load_split_data(self, split_info, class_mapping, num_classes):
        """
        Carrega dados de um split específico

        Args:
            split_info: Informações do split
            class_mapping: Mapeamento de nomes de classes para IDs
            num_classes: Número total de classes

        Returns:
            images, class_labels, bbox_labels
        """
        # Carregar anotações
        with open(split_info['annotations_file'], 'r') as f:
            annotations = json.load(f)

        images = []
        class_labels = []
        bbox_labels = []

        # Criar mapeamento de IDs de categoria originais para novos IDs
        original_to_new_id = {}
        for category in annotations['categories']:
            original_id = category['id']
            class_name = category['name'].lower()
            new_id = class_mapping.get(class_name, 0)  # 0 = background se não encontrar
            original_to_new_id[original_id] = new_id

        # Agrupar anotações por imagem para lidar com múltiplos objetos por imagem
        annotations_by_image = {}
        for annotation in annotations['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(annotation)

        # Processar cada imagem
        for image_info in tqdm(annotations['images'], desc="Carregando imagens"):
            image_id = image_info['id']
            image_path = os.path.join(split_info['images_dir'], image_info['file_name'])

            if not os.path.exists(image_path):
                continue

            try:
                # Carregar e redimensionar imagem
                image = cv2.imread(image_path)
                if image is None:
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, self.input_size)
                image = image.astype(np.float32) / 255.0

                # Processar anotações desta imagem
                image_annotations = annotations_by_image.get(image_id, [])

                if len(image_annotations) == 0:
                    # Imagem sem anotações (background)
                    class_vector = np.zeros(num_classes, dtype=np.float32)
                    class_vector[0] = 1.0  # background
                    bbox_vector = np.zeros(4, dtype=np.float32)  # [0, 0, 0, 0] for background
                else:
                    # Usar a primeira anotação (para simplificar, poderia ser melhorado)
                    annotation = image_annotations[0]

                    # Mapear categoria
                    original_category_id = annotation['category_id']
                    new_category_id = original_to_new_id.get(original_category_id, 0)

                    # One-hot encoding para classes
                    class_vector = np.zeros(num_classes, dtype=np.float32)
                    class_vector[new_category_id] = 1.0

                    # Normalizar bounding box
                    bbox = np.array(annotation['bbox'], dtype=np.float32)
                    img_width = image_info['width']
                    img_height = image_info['height']

                    # Converter para formato normalizado [x_min, y_min, x_max, y_max]
                    x_min = bbox[0] / img_width
                    y_min = bbox[1] / img_height
                    x_max = (bbox[0] + bbox[2]) / img_width
                    y_max = (bbox[1] + bbox[3]) / img_height

                    # Garantir que estão no intervalo [0,1]
                    x_min = max(0, min(1, x_min))
                    y_min = max(0, min(1, y_min))
                    x_max = max(0, min(1, x_max))
                    y_max = max(0, min(1, y_max))

                    # Criar vetor de bbox (apenas 4 coordenadas)
                    bbox_vector = np.array([y_min, x_min, y_max, x_max], dtype=np.float32)

                images.append(image)
                class_labels.append(class_vector)
                bbox_labels.append(bbox_vector)

            except Exception as e:
                print(f"Erro ao processar imagem {image_path}: {e}")
                continue

        return images, class_labels, bbox_labels

    def data_generator(self, X, y_class, y_bbox, batch_size=32, augment=True):
        """
        Gerador de dados para treinamento com data augmentation

        Args:
            X: Imagens de entrada
            y_class: Labels de classe
            y_bbox: Labels de bounding box
            batch_size: Tamanho do batch
            augment: Se deve aplicar data augmentation

        Yields:
            batch de (imagens, [classes, bboxes])
        """
        num_samples = X.shape[0]
        indices = np.arange(num_samples)

        # Definir transformações para data augmentation
        if augment:
            datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            datagen = ImageDataGenerator()

        while True:
            np.random.shuffle(indices)

            for start_idx in range(0, num_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]

                batch_X = X[batch_indices]
                batch_y_class = y_class[batch_indices]
                batch_y_bbox = y_bbox[batch_indices]

                # Aplicar data augmentation apenas nas imagens
                if augment:
                    for i in range(len(batch_X)):
                        batch_X[i] = datagen.random_transform(batch_X[i])

                # bbox data is already in correct format (batch_size, 4)
                yield batch_X.astype(np.float32), {
                    'class_output': batch_y_class.astype(np.float32),
                    'bbox_output': batch_y_bbox.astype(np.float32)
                }

    def train(self, annotations_file, images_dir, classes, epochs=50, save_path='models'):
        """
        Treina o modelo com o dataset fornecido

        Args:
            annotations_file: Caminho para arquivo de anotações (JSON)
            images_dir: Diretório contendo as imagens
            classes: Lista de classes (incluindo background como primeiro item)
            epochs: Número de épocas para treinar
            save_path: Caminho para salvar o modelo treinado

        Returns:
            Histórico de treinamento
        """
        # Garantir que o diretório de salvamento existe
        os.makedirs(save_path, exist_ok=True)

        # Construir modelo se ainda não existir
        if self.model is None:
            self.build_model(len(classes))

        # Preparar o dataset
        print("Preparando dataset...")
        X_train, X_val, y_train_class, y_train_bbox, y_val_class, y_val_bbox = self._prepare_dataset(
            annotations_file, images_dir
        )
        print(f"Dataset preparado: {X_train.shape[0]} amostras de treino, {X_val.shape[0]} amostras de validação")

        # Initialize progress tracker
        steps_per_epoch = len(X_train) // self.batch_size
        self.progress_tracker = ProgressTracker(epochs, steps_per_epoch)

        # Custom callback for enhanced progress tracking
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, tracker):
                super().__init__()
                self.tracker = tracker

            def on_train_begin(self, logs=None):
                self.tracker.start_training()

            def on_epoch_begin(self, epoch, logs=None):
                self.tracker.on_epoch_begin(epoch)
                print(f"\n🚀 Starting Epoch {epoch + 1}/{self.tracker.total_epochs}")
                
            def on_train_batch_end(self, batch, logs=None):
                if batch % 20 == 0 and logs:  # Show progress every 20 batches
                    current_loss = logs.get('loss', 0)
                    current_acc = logs.get('class_output_accuracy', 0)
                    bbox_loss = logs.get('bbox_output_loss', 0)
                    class_loss = logs.get('class_output_loss', 0)
                    lr = logs.get('lr', logs.get('learning_rate', 0))
                    
                    # Calculate progress
                    progress = (batch + 1) / self.tracker.steps_per_epoch
                    progress_bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
                    
                    print(f"  [{progress_bar}] Step {batch + 1}/{self.tracker.steps_per_epoch} "
                          f"| Loss: {current_loss:.4f} (cls: {class_loss:.4f}, bbox: {bbox_loss:.4f}) "
                          f"| Acc: {current_acc:.4f} | LR: {lr:.2e}")

            def on_epoch_end(self, epoch, logs=None):
                self.tracker.update_epoch(epoch, logs)
                # Optimize memory after each epoch
                optimize_memory()

        # Configurar callbacks
        progress_callback = ProgressCallback(self.progress_tracker)

        checkpoint = ModelCheckpoint(
            os.path.join(save_path, 'model_checkpoint.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )

        callbacks = [progress_callback, checkpoint, early_stopping, reduce_lr]

        # Treinar o modelo
        print("Iniciando treinamento...")
        history = self.model.fit(
            self.data_generator(X_train, y_train_class, y_train_bbox, self.batch_size, augment=True),
            validation_data=self.data_generator(X_val, y_val_class, y_val_bbox, self.batch_size, augment=False),
            steps_per_epoch=len(X_train) // self.batch_size,
            validation_steps=len(X_val) // self.batch_size,
            epochs=epochs,
            callbacks=callbacks
        )

        # Salvar o modelo final
        self.model.save(os.path.join(save_path, 'final_model.h5'))
        print(f"Modelo salvo em {os.path.join(save_path, 'final_model.h5')}")

        # Plotar gráficos de treinamento
        self._plot_training_history(history, save_path)

        return history

    def _plot_training_history(self, history, save_path):
        """
        Plota e salva gráficos de histórico de treinamento

        Args:
            history: Histórico retornado por model.fit()
            save_path: Caminho para salvar os gráficos
        """
        plt.figure(figsize=(12, 4))

        # Gráfico de loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Treino')
        plt.plot(history.history['val_loss'], label='Validação')
        plt.title('Loss Total')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()

        # Gráfico de acurácia
        plt.subplot(1, 2, 2)
        plt.plot(history.history['class_output_accuracy'], label='Treino')
        plt.plot(history.history['val_class_output_accuracy'], label='Validação')
        plt.title('Acurácia de Classificação')
        plt.xlabel('Época')
        plt.ylabel('Acurácia')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_history.png'))
        plt.close()

        print(f"Gráfico de treinamento salvo em {os.path.join(save_path, 'training_history.png')}")

    def evaluate(self, annotations_file, images_dir, classes):
        """
        Avalia o modelo em um conjunto de dados de teste

        Args:
            annotations_file: Caminho para arquivo de anotações (JSON)
            images_dir: Diretório contendo as imagens
            classes: Lista de classes (incluindo background como primeiro item)

        Returns:
            Métricas de avaliação
        """
        if self.model is None:
            raise ValueError("O modelo não foi carregado ou treinado")

        # Preparar o dataset
        print("Preparando dataset de teste...")
        X_test, _, y_test_class, _, y_test_bbox, _ = self._prepare_dataset(
            annotations_file, images_dir
        )

        # Avaliar o modelo
        print("Avaliando modelo...")
        evaluation = self.model.evaluate(
            self.data_generator(X_test, y_test_class, y_test_bbox, self.batch_size, augment=False),
            steps=len(X_test) // self.batch_size,
            verbose=1
        )

        # Imprimir resultados
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = evaluation[i]
            print(f"{metric_name}: {evaluation[i]:.4f}")

        # Calcular métricas adicionais
        y_pred_class, y_pred_bbox = self.model.predict(X_test)

        # Calcular mAP (mean Average Precision)
        map_score = self._calculate_map(y_test_class, y_pred_class, y_test_bbox, y_pred_bbox)
        metrics['mAP'] = map_score
        print(f"mAP: {map_score:.4f}")

        return metrics

    def _calculate_map(self, y_true_class, y_pred_class, y_true_bbox, y_pred_bbox, iou_threshold=0.5):
        """
        Calcula o mAP (mean Average Precision) para avaliação de detecção de objetos

        Args:
            y_true_class: Classes verdadeiras (one-hot)
            y_pred_class: Probabilidades de classe previstas
            y_true_bbox: Bounding boxes verdadeiros
            y_pred_bbox: Bounding boxes previstos
            iou_threshold: Limiar para considerar uma detecção correta

        Returns:
            mAP score
        """
        num_classes = y_true_class.shape[1]
        aps = []

        for class_id in range(1, num_classes):  # Ignorar background (índice 0)
            # Filtrar instâncias da classe atual
            true_instances = np.where(y_true_class[:, class_id] > 0.5)[0]
            if len(true_instances) == 0:
                continue

            pred_scores = y_pred_class[:, class_id]

            # Ordenar predições por confiança
            sorted_indices = np.argsort(-pred_scores)

            tp = np.zeros(len(sorted_indices))
            fp = np.zeros(len(sorted_indices))

            for i, idx in enumerate(sorted_indices):
                if pred_scores[idx] < 0.5:  # Ignorar predições de baixa confiança
                    continue

                # Calcular IoU com todas as ground truths desta classe
                ious = []
                for true_idx in true_instances:
                    true_bbox = y_true_bbox[true_idx, class_id]
                    pred_bbox = y_pred_bbox[idx, class_id]
                    iou = self._calculate_iou(true_bbox, pred_bbox)
                    ious.append(iou)

                # Se não houver ground truths, é um FP
                if len(ious) == 0:
                    fp[i] = 1
                    continue

                # Verificar se é um TP ou FP
                max_iou = max(ious)
                max_iou_idx = np.argmax(ious)

                if max_iou >= iou_threshold:
                    # É um true positive
                    tp[i] = 1
                    # Remover este ground truth para evitar múltiplas detecções
                    true_instances = np.delete(true_instances, max_iou_idx)
                else:
                    # É um false positive
                    fp[i] = 1

            # Calcular precisão e recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            recalls = tp_cumsum / len(true_instances) if len(true_instances) > 0 else np.zeros_like(tp_cumsum)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

            # Calcular AP (área sob a curva precisão-recall)
            ap = 0
            for t in np.arange(0, 1.1, 0.1):  # 11 pontos de interpolação padrão
                if np.sum(recalls >= t) == 0:
                    p = 0
                else:
                    p = np.max(precisions[recalls >= t])
                ap += p / 11

            aps.append(ap)

        # Calcular mAP
        return np.mean(aps) if len(aps) > 0 else 0

    def _calculate_iou(self, bbox1, bbox2):
        """
        Calcula a Intersection over Union (IoU) entre duas bounding boxes
        Formato das bounding boxes: [y_min, x_min, y_max, x_max]

        Args:
            bbox1: Primeira bounding box
            bbox2: Segunda bounding box

        Returns:
            IoU score
        """
        # Extrair coordenadas
        y1_min, x1_min, y1_max, x1_max = bbox1
        y2_min, x2_min, y2_max, x2_max = bbox2

        # Calcular área de cada bounding box
        area1 = (y1_max - y1_min) * (x1_max - x1_min)
        area2 = (y2_max - y2_min) * (x2_max - x2_min)

        # Calcular coordenadas da interseção
        y_min_inter = max(y1_min, y2_min)
        x_min_inter = max(x1_min, x2_min)
        y_max_inter = min(y1_max, y2_max)
        x_max_inter = min(x1_max, x2_max)

        # Calcular área da interseção
        if y_min_inter >= y_max_inter or x_min_inter >= x_max_inter:
            return 0.0  # Não há interseção

        area_inter = (y_max_inter - y_min_inter) * (x_max_inter - x_min_inter)

        # Calcular IoU
        area_union = area1 + area2 - area_inter
        iou = area_inter / area_union if area_union > 0 else 0

        return iou

    def load_model(self, model_path):
        """
        Carrega um modelo salvo anteriormente

        Args:
            model_path: Caminho para o arquivo do modelo (.h5)

        Returns:
            Modelo carregado
        """
        print(f"Carregando modelo de {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        print("Modelo carregado com sucesso!")
        return self.model

class RoboflowDatasetLoader:
    """
    Classe para carregar e processar múltiplos datasets do Roboflow
    Suporta diferentes estruturas de pastas (train/valid/test ou sem divisão)
    """

    def __init__(self, datasets_base_dir="datasets"):
        self.base_dir = Path(datasets_base_dir)
        self.datasets_info = {}

    def discover_datasets(self):
        """
        Descobre automaticamente todos os datasets do Roboflow no diretório base

        Returns:
            Lista de dicionários com informações dos datasets encontrados
        """
        discovered_datasets = []

        if not self.base_dir.exists():
            print(f"Diretório {self.base_dir} não encontrado!")
            return discovered_datasets

        # Procurar por todos os diretórios que parecem ser datasets do Roboflow
        for dataset_dir in self.base_dir.iterdir():
            if dataset_dir.is_dir() and not dataset_dir.name.startswith('.'):
                # Verificar se é um dataset válido (tem arquivos de anotação COCO)
                dataset_info = self._analyze_dataset_structure(dataset_dir)
                if dataset_info:
                    discovered_datasets.append(dataset_info)
                    print(f"Dataset encontrado: {dataset_info['name']}")
                    print(f"  - Estrutura: {dataset_info['structure_type']}")
                    print(f"  - Splits: {list(dataset_info['splits'].keys())}")
                    print(f"  - Total de classes: {len(dataset_info['classes'])}")
                    print()

        return discovered_datasets

    def _analyze_dataset_structure(self, dataset_dir):
        """
        Analisa a estrutura de um dataset individual

        Args:
            dataset_dir: Path para o diretório do dataset

        Returns:
            Dicionário com informações do dataset ou None se inválido
        """
        dataset_info = {
            'name': dataset_dir.name,
            'path': str(dataset_dir),
            'structure_type': None,
            'splits': {},
            'classes': set(),
            'total_images': 0,
            'total_annotations': 0
        }

        # Verificar se há estrutura train/valid/test
        has_splits = False
        possible_splits = ['train', 'valid', 'validation', 'val', 'test']

        for split_name in possible_splits:
            split_dir = dataset_dir / split_name
            if split_dir.exists() and split_dir.is_dir():
                # Procurar arquivo de anotações COCO neste split
                annotations_file = self._find_coco_annotations(split_dir)
                if annotations_file:
                    has_splits = True
                    # Normalizar nome do split
                    normalized_name = 'val' if split_name in ['valid', 'validation'] else split_name

                    split_info = self._analyze_split(split_dir, annotations_file)
                    if split_info:
                        dataset_info['splits'][normalized_name] = split_info
                        dataset_info['classes'].update(split_info['classes'])
                        dataset_info['total_images'] += split_info['num_images']
                        dataset_info['total_annotations'] += split_info['num_annotations']

        # Se não há splits, verificar se há anotações na raiz
        if not has_splits:
            annotations_file = self._find_coco_annotations(dataset_dir)
            if annotations_file:
                split_info = self._analyze_split(dataset_dir, annotations_file)
                if split_info:
                    dataset_info['splits']['all'] = split_info
                    dataset_info['classes'].update(split_info['classes'])
                    dataset_info['total_images'] = split_info['num_images']
                    dataset_info['total_annotations'] = split_info['num_annotations']
                    has_splits = True

        if has_splits:
            dataset_info['structure_type'] = 'split' if len(dataset_info['splits']) > 1 else 'single'
            dataset_info['classes'] = sorted(list(dataset_info['classes']))
            return dataset_info

        return None

    def _find_coco_annotations(self, directory):
        """
        Encontra arquivo de anotações COCO em um diretório

        Args:
            directory: Path para o diretório

        Returns:
            Path para o arquivo de anotações ou None
        """
        # Padrões comuns para arquivos de anotação COCO
        patterns = [
            '_annotations.coco.json',
            'annotations.json',
            '_annotations.json',
            'instances_*.json'
        ]

        for pattern in patterns:
            matches = list(directory.glob(pattern))
            if matches:
                return matches[0]

        # Procurar qualquer arquivo .json que possa ser anotação
        json_files = list(directory.glob('*.json'))
        for json_file in json_files:
            if self._is_coco_format(json_file):
                return json_file

        return None

    def _is_coco_format(self, json_file):
        """
        Verifica se um arquivo JSON está no formato COCO

        Args:
            json_file: Path para o arquivo JSON

        Returns:
            True se for formato COCO, False caso contrário
        """
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Verificar se tem as chaves essenciais do formato COCO
            required_keys = ['images', 'annotations', 'categories']
            return all(key in data for key in required_keys)
        except:
            return False

    def _analyze_split(self, split_dir, annotations_file):
        """
        Analisa um split específico (train, val, test)

        Args:
            split_dir: Path para o diretório do split
            annotations_file: Path para o arquivo de anotações

        Returns:
            Dicionário com informações do split
        """
        try:
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)

            split_info = {
                'images_dir': str(split_dir),
                'annotations_file': str(annotations_file),
                'num_images': len(annotations['images']),
                'num_annotations': len(annotations['annotations']),
                'classes': [cat['name'] for cat in annotations['categories']],
                'categories': annotations['categories']
            }

            return split_info
        except Exception as e:
            print(f"Erro ao analisar split {split_dir}: {e}")
            return None

    def load_combined_dataset(self, dataset_filters=None, combine_classes=True):
        """
        Carrega e combina múltiplos datasets do Roboflow

        Args:
            dataset_filters: Lista de padrões para filtrar datasets (opcional)
            combine_classes: Se deve combinar classes similares entre datasets

        Returns:
            Dicionário com dataset combinado
        """
        datasets = self.discover_datasets()

        if dataset_filters:
            # Filtrar datasets baseado nos padrões fornecidos
            filtered_datasets = []
            for dataset in datasets:
                for filter_pattern in dataset_filters:
                    if filter_pattern.lower() in dataset['name'].lower():
                        filtered_datasets.append(dataset)
                        break
            datasets = filtered_datasets

        if not datasets:
            print("Nenhum dataset encontrado!")
            return None

        print(f"Combinando {len(datasets)} datasets...")

        # Combinar todas as classes
        all_classes = set(['background'])  # Sempre incluir background
        for dataset in datasets:
            all_classes.update(dataset['classes'])

        if combine_classes:
            # Mapear classes similares
            all_classes = self._normalize_class_names(all_classes)

        all_classes = sorted(list(all_classes))

        # Criar mapeamento de classes
        class_mapping = {name: idx for idx, name in enumerate(all_classes)}

        combined_info = {
            'datasets': datasets,
            'classes': all_classes,
            'class_mapping': class_mapping,
            'structure_type': 'combined'
        }

        return combined_info

    def _normalize_class_names(self, class_names):
        """
        Normaliza nomes de classes similares entre diferentes datasets

        Args:
            class_names: Set de nomes de classes

        Returns:
            Set de nomes de classes normalizados
        """
        # Mapeamento de classes similares
        class_synonyms = {
            'pencil': ['lapis', 'lápis', 'pencil'],
            'pen': ['caneta', 'pen', 'ballpoint'],
            'eraser': ['borracha', 'eraser', 'rubber'],
            'ruler': ['régua', 'regua', 'ruler', 'scale'],
            'sharpener': ['apontador', 'sharpener'],
            'notebook': ['caderno', 'notebook'],
            'book': ['livro', 'book'],
            'calculator': ['calculadora', 'calculator'],
            'scissors': ['tesoura', 'scissors'],
            'glue': ['cola', 'glue'],
            'stapler': ['grampeador', 'stapler'],
            'tape': ['fita', 'tape'],
            'clip': ['clipe', 'clip', 'paperclip']
        }

        normalized_classes = set()
        class_names_lower = {name.lower(): name for name in class_names}

        # Primeiro, mapear sinônimos
        used_originals = set()
        for normalized_name, synonyms in class_synonyms.items():
            found_synonym = False
            for synonym in synonyms:
                if synonym.lower() in class_names_lower:
                    normalized_classes.add(normalized_name)
                    used_originals.add(class_names_lower[synonym.lower()])
                    found_synonym = True
                    break

        # Adicionar classes que não foram mapeadas
        for original_name in class_names:
            if original_name not in used_originals and original_name.lower() != 'background':
                normalized_classes.add(original_name.lower())

        return normalized_classes

def prepare_roboflow_datasets_for_training(datasets_base_dir="datasets", dataset_filters=None):
    """
    Função para preparar datasets do Roboflow para treinamento

    Args:
        datasets_base_dir: Diretório base onde estão os datasets do Roboflow
        dataset_filters: Lista de padrões para filtrar datasets específicos (opcional)
                        Ex: ['stationery', 'office', 'school'] para incluir apenas esses tipos

    Returns:
        Dicionário com informações dos datasets combinados do Roboflow
    """
    print("=== Carregando Datasets do Roboflow ===")

    # Inicializar o carregador de datasets
    loader = RoboflowDatasetLoader(datasets_base_dir)

    # Carregar e combinar datasets
    combined_dataset = loader.load_combined_dataset(
        dataset_filters=dataset_filters,
        combine_classes=True
    )

    if combined_dataset is None:
        print("Nenhum dataset válido encontrado!")
        return None

    print(f"\n=== Resumo dos Datasets Carregados ===")
    print(f"Total de datasets: {len(combined_dataset['datasets'])}")
    print(f"Total de classes: {len(combined_dataset['classes'])}")
    print(f"Classes encontradas: {combined_dataset['classes']}")

    total_images = 0
    total_annotations = 0
    for dataset in combined_dataset['datasets']:
        total_images += dataset['total_images']
        total_annotations += dataset['total_annotations']
        print(f"\nDataset: {dataset['name']}")
        print(f"  Imagens: {dataset['total_images']}")
        print(f"  Anotações: {dataset['total_annotations']}")
        print(f"  Splits: {list(dataset['splits'].keys())}")

    print(f"\nTotal de imagens: {total_images}")
    print(f"Total de anotações: {total_annotations}")

    return combined_dataset


def train_with_roboflow_datasets(datasets_base_dir="datasets", dataset_filters=None,
                                epochs=30, save_path='models/roboflow_educational_objects'):
    """
    Função para treinar modelo usando datasets do Roboflow

    Args:
        datasets_base_dir: Diretório base dos datasets
        dataset_filters: Filtros para selecionar datasets específicos
        epochs: Número de épocas para treinar
        save_path: Caminho para salvar o modelo

    Returns:
        Histórico de treinamento e métricas
    """
    print("=== Sistema de IA para Identificação de Objetos Educacionais ===")
    print("Treinamento com Datasets do Roboflow")
    print("Desenvolvido para auxiliar pessoas com deficiência visual")
    print()

    # 1. Preparar datasets do Roboflow
    print("1. Carregando datasets do Roboflow...")
    combined_dataset = prepare_roboflow_datasets_for_training(
        datasets_base_dir=datasets_base_dir,
        dataset_filters=dataset_filters
    )

    if combined_dataset is None:
        print("Erro: Não foi possível carregar os datasets!")
        return None

    # 2. Inicializar treinador
    print("\n2. Inicializando modelo...")
    # Use larger batch size for RTX 4080 Super
    optimal_batch_size = 64 if gpu_available else 16
    trainer = ModelTrainer(input_size=(300, 300), batch_size=optimal_batch_size)

    # 3. Preparar dados para treinamento
    print("\n3. Preparando dados para treinamento...")
    try:
        X_train, X_val, y_train_class, y_train_bbox, y_val_class, y_val_bbox, num_classes = trainer.prepare_roboflow_datasets(combined_dataset)
    except Exception as e:
        print(f"Erro ao preparar datasets: {e}")
        return None

    # 4. Construir modelo
    print("\n4. Construindo modelo...")
    trainer.build_model(num_classes)

    # 5. Treinar modelo
    print("\n5. Iniciando treinamento...")

    # Initialize progress tracker
    steps_per_epoch = len(X_train) // trainer.batch_size
    progress_tracker = ProgressTracker(epochs, steps_per_epoch)

    # Custom callback for progress tracking
    class ProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self, tracker):
            super().__init__()
            self.tracker = tracker

        def on_train_begin(self, logs=None):
            self.tracker.start_training()

        def on_epoch_begin(self, epoch, logs=None):
            self.tracker.on_epoch_begin(epoch)
            print(f"\n🚀 Starting Epoch {epoch + 1}/{self.tracker.total_epochs}")
            
        def on_train_batch_end(self, batch, logs=None):
            if batch % 15 == 0 and logs:  # Show progress every 15 batches
                current_loss = logs.get('loss', 0)
                current_acc = logs.get('class_output_accuracy', 0)
                bbox_loss = logs.get('bbox_output_loss', 0)
                class_loss = logs.get('class_output_loss', 0)
                lr = logs.get('lr', logs.get('learning_rate', 0))
                
                # Calculate progress within epoch
                progress = (batch + 1) / self.tracker.steps_per_epoch
                progress_bar = "█" * int(progress * 25) + "░" * (25 - int(progress * 25))
                
                # Calculate time estimates
                if hasattr(self.tracker, 'epoch_start_time') and self.tracker.epoch_start_time:
                    batch_elapsed = time.time() - self.tracker.epoch_start_time
                    if batch > 0:
                        avg_batch_time = batch_elapsed / (batch + 1)
                        remaining_batches = self.tracker.steps_per_epoch - (batch + 1)
                        eta_epoch = avg_batch_time * remaining_batches
                        eta_str = f"ETA: {int(eta_epoch//60):02d}:{int(eta_epoch%60):02d}"
                    else:
                        eta_str = "ETA: --:--"
                else:
                    eta_str = "ETA: --:--"
                
                print(f"  [{progress_bar}] {batch + 1}/{self.tracker.steps_per_epoch} "
                      f"| Loss: {current_loss:.4f} (cls: {class_loss:.4f}, bbox: {bbox_loss:.4f}) "
                      f"| Acc: {current_acc:.4f} | LR: {lr:.2e} | {eta_str}")

        def on_epoch_end(self, epoch, logs=None):
            self.tracker.update_epoch(epoch, logs)
            # Optimize memory after each epoch
            optimize_memory()

    # Configurar callbacks
    os.makedirs(save_path, exist_ok=True)

    progress_callback = ProgressCallback(progress_tracker)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(save_path, 'model_checkpoint.h5'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,  # Increased patience for better training
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,  # More aggressive LR reduction
        patience=7,
        min_lr=1e-7,
        verbose=1
    )

    callbacks = [progress_callback, checkpoint, early_stopping, reduce_lr]

    # Use direct fit with generators instead of tf.data.Dataset
    train_generator = trainer.data_generator(X_train, y_train_class, y_train_bbox, trainer.batch_size, augment=True)
    val_generator = trainer.data_generator(X_val, y_val_class, y_val_bbox, trainer.batch_size, augment=False)

    # Treinar
    history = trainer.model.fit(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=len(X_val) // trainer.batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1  # Show progress bars
    )

    # 6. Salvar modelo final
    trainer.model.save(os.path.join(save_path, 'final_model.h5'))
    print(f"Modelo salvo em {os.path.join(save_path, 'final_model.h5')}")

    # 7. Plotar gráficos de treinamento
    trainer._plot_training_history(history, save_path)

    # 8. Avaliação básica
    print("\n6. Avaliando modelo...")
    evaluation = trainer.model.evaluate(
        trainer.data_generator(X_val, y_val_class, y_val_bbox, trainer.batch_size, augment=False),
        steps=len(X_val) // trainer.batch_size,
        verbose=1
    )

    metrics = {}
    for i, metric_name in enumerate(trainer.model.metrics_names):
        metrics[metric_name] = evaluation[i]
        print(f"{metric_name}: {evaluation[i]:.4f}")

    # 9. Salvar informações do modelo
    model_info = {
        'classes': combined_dataset['classes'],
        'class_mapping': combined_dataset['class_mapping'],
        'input_size': trainer.input_size,
        'metrics': metrics,
        'datasets_used': [ds['name'] for ds in combined_dataset['datasets']],
        'total_training_samples': len(X_train),
        'total_validation_samples': len(X_val),
        'description': 'Modelo treinado com datasets do Roboflow para identificação de objetos educacionais'
    }

    model_info_file = os.path.join(save_path, 'model_info.json')
    with open(model_info_file, 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"\n=== Treinamento Concluído ===")
    print(f"Modelo salvo em: {save_path}")
    print(f"Classes treinadas: {len(combined_dataset['classes'])}")
    print(f"Datasets utilizados: {len(combined_dataset['datasets'])}")
    print(f"Amostras de treino: {len(X_train)}")
    print(f"Amostras de validação: {len(X_val)}")
    print("\nO modelo está pronto para ser usado no sistema de auxílio visual!")

    return history, metrics, model_info





if __name__ == "__main__":
    """
    Função principal para executar o treinamento com datasets do Roboflow
    """
    import sys

    print("=== Sistema de IA para Identificação de Objetos Educacionais ===")
    print("Desenvolvido para auxiliar pessoas com deficiência visual")
    print("Treinamento usando datasets reais do Roboflow")
    print()

    # Log GPU information
    log_gpu_info()

    # Descobrir datasets disponíveis
    print("Descobrindo datasets disponíveis...")
    loader = RoboflowDatasetLoader("datasets")
    discovered = loader.discover_datasets()

    if not discovered:
        print("❌ Nenhum dataset do Roboflow encontrado no diretório 'datasets'!")
        print("Certifique-se de que há datasets válidos no formato COCO no diretório.")
        sys.exit(1)

    print(f"✅ {len(discovered)} dataset(s) encontrado(s):")
    for dataset in discovered:
        print(f"  - {dataset['name']}: {dataset['total_images']} imagens, {len(dataset['classes'])} classes")

    # Opção para filtrar datasets específicos
    print("\nOpções de filtro (deixe em branco para usar todos):")
    print("Exemplo: stationery,office,school (separado por vírgula)")
    filter_input = input("Filtros (opcional): ").strip()

    dataset_filters = None
    if filter_input:
        dataset_filters = [f.strip() for f in filter_input.split(',')]
        print(f"Usando filtros: {dataset_filters}")
    else:
        print("Usando todos os datasets encontrados")

    # Perguntar número de épocas
    epochs_input = input("\nNúmero de épocas (padrão: 30): ").strip()
    epochs = 30
    if epochs_input.isdigit():
        epochs = int(epochs_input)
    print(f"Treinando por {epochs} épocas")

    # Treinar com datasets do Roboflow
    try:
        print("\n" + "="*50)
        print("INICIANDO TREINAMENTO")
        print("="*50)

        history, metrics, model_info = train_with_roboflow_datasets(
            datasets_base_dir="datasets",
            dataset_filters=dataset_filters,
            epochs=epochs,
            save_path='models/educational_objects'
        )

        if history is not None:
            print("\n" + "="*50)
            print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
            print("="*50)
            print(f"Modelo salvo em: models/educational_objects/")
            print(f"Classes treinadas: {len(model_info['classes'])}")
            print(f"Datasets utilizados: {len(model_info['datasets_used'])}")
            print(f"Amostras de treino: {model_info['total_training_samples']}")
            print(f"Amostras de validação: {model_info['total_validation_samples']}")
            print("\nO modelo está pronto para ser usado no sistema de auxílio visual!")
        else:
            print("\n❌ Erro durante o treinamento")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Erro no treinamento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main_roboflow_only():
    """
    Função para executar apenas com datasets do Roboflow usando filtros específicos
    """
    # Filtros de exemplo - você pode modificar conforme necessário
    dataset_filters = ['stationery', 'office', 'school', 'pen', 'pencil']

    history, metrics, model_info = train_with_roboflow_datasets(
        datasets_base_dir="datasets",
        dataset_filters=dataset_filters,
        epochs=50,  # Mais épocas para datasets reais
        save_path='models/educational_objects'
    )

    return history, metrics, model_info
