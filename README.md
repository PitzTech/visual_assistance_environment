# Sistema de Assistência Visual para Ambientes Educacionais

Um sistema inteligente de assistência visual que utiliza redes neurais convolucionais e reconhecimento de gestos para auxiliar pessoas com deficiência visual em ambientes educacionais.

## 🎯 Sobre o Projeto

Este projeto desenvolve uma plataforma inovadora que combina detecção de objetos, reconhecimento de gestos e síntese de voz para criar uma solução completa de acessibilidade. O sistema detecta objetos educacionais comuns e fornece feedback auditivo sobre suas posições espaciais, permitindo maior autonomia para pessoas com deficiência visual.

### ✨ Características Principais

- **Detecção de Objetos Inteligente**: Utiliza YOLO11m para identificação em tempo real de materiais escolares
- **Reconhecimento de Gestos Personalizado**: Sistema de cadastro de gestos específicos via MediaPipe
- **Feedback Auditivo Natural**: Síntese de voz neural através da API Kokoro
- **Localização Espacial**: Algoritmo de posicionamento que descreve a localização dos objetos
- **Interface Web Responsiva**: Frontend intuitivo otimizado para acessibilidade

## 🏗️ Arquitetura do Sistema

### Frontend
- Interface HTML5 responsiva
- Captura de vídeo via WebRTC
- Processamento de gestos em tempo real
- Comunicação WebSocket bidirecional

### Backend
- Servidor WebSocket Python
- Modelo YOLO11m para detecção de objetos
- Processamento de coordenadas espaciais
- Integração com API de síntese de voz

## 🚀 Performance

| Métrica | YOLO11m (v2) | Melhoria vs v1 |
|---------|--------------|----------------|
| Tempo de Inferência | ~30ms | 6.7x mais rápido |
| Uso de VRAM | 2GB | 75% menos |
| mAP@0.5 | 0.89 | 24% maior |
| FPS | ~33 FPS | 6.6x mais rápido |

## 📋 Pré-requisitos

### Sistema
- Python 3.8+
- NVIDIA GPU (recomendado) ou CPU
- Webcam
- Navegador Chrome/Chromium

### Hardware Testado
- GPU: NVIDIA RTX 4080 Super (16GB VRAM)
- CPU: AMD Ryzen 7 9700X
- RAM: 64GB
- OS: Ubuntu 24.04 LTS

## 🛠️ Instalação

### 1. Clone o Repositório
```bash
git clone <repository-url>
cd visual_assistance
```

### 2. Configurar Backend
```bash
cd backend
pip install -r requirements.txt
```

### 3. Configurar Síntese de Voz (Docker)
```bash
# Para CPU
docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest

# Para GPU NVIDIA
docker run --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu:latest
```

### 4. Baixar Modelos YOLO
```bash
cd backend
# Os modelos yolo11m.pt e yolov8m.pt devem estar na pasta backend/
```

## 🚀 Como Usar

### 1. Iniciar o Backend
```bash
cd backend
python main.py
```

### 2. Abrir o Frontend
```bash
cd frontend
# Abrir main.html no navegador Chrome/Chromium
```

### 3. Funcionalidades

#### Detecção de Objetos
- O sistema detecta automaticamente objetos educacionais
- Fornece descrição auditiva da localização (ex: "caderno na esquerda meio")
- Suporte para: livros, cadernos, canetas, lápis, mochilas, laptops, etc.

#### Cadastro de Gestos
1. Digite o nome do objeto no campo "Nome do Objeto"
2. Clique em "Capturar Sinal"
3. Faça o gesto desejado na frente da câmera
4. Clique em "Cadastrar Sinal" para salvar

#### Busca por Gestos
- Faça um gesto cadastrado para buscar objetos específicos
- O sistema fornecerá feedback auditivo sobre a localização do objeto

## 🔧 Configuração

### Ajustar Confiança de Detecção
No arquivo `backend/main.py`:
```python
detector = ClassroomObjectDetector(
    model_path='yolo11m.pt',
    confidence_threshold=0.7  # Ajustar entre 0.1-0.9
)
```

### Modelos Disponíveis
- `yolo11m.pt`: Balanceado (recomendado)
- `yolov8n.pt`: Mais rápido (hardware limitado)
- `yolov8m.pt`: Maior precisão

## 📊 Objetos Detectáveis

### COCO Dataset (Pré-treinado)
✅ Pessoa, cadeira, mesa, laptop, livro, mochila, mouse, teclado, celular, garrafa, xícara, TV, maçã, banana

### Objetos Personalizados (v1 - Dataset Customizado)
✅ Caneta, lápis, borracha, apontador, estojo, caderno específico, régua, calculadora, tesoura

## 🌐 Compatibilidade

- **Navegadores Testados**: Chrome, Chromium
- **Sistema Operacional**: Linux (testado), Windows, macOS
- **Hardware**: CPU (funcional) ou GPU NVIDIA (recomendado)

## 🎨 Interface

A interface apresenta:
- Visualização simultânea da câmera original e detecções
- Painel para cadastro de novos gestos
- Lista de gestos cadastrados
- Indicadores de status do sistema
- Feedback visual das detecções

## 📈 Algoritmo de Posicionamento

O sistema divide a tela em uma grade 3x3:
```
esquerda_frente | centro_frente | direita_frente
esquerda_meio   | centro_meio   | direita_meio
esquerda_fundo  | centro_fundo  | direita_fundo
```

## 🧪 Casos de Uso Validados

- **Localização de Material Escolar**: Identificação de canetas, lápis, borrachas
- **Organização de Mesa**: Detecção de livros, cadernos, calculadoras
- **Navegação em Sala**: Identificação de laptops, mochilas, garrafas
- **Interação por Gestos**: Busca direcionada através de gestos personalizados

## ⚡ Métricas de Performance

- **Detecção de objetos**: <50ms por frame
- **Reconhecimento de gestos**: <100ms
- **Síntese de voz**: <200ms
- **Tempo total de resposta**: <500ms
- **Precisão de localização**: >90% de sucesso

## 🔬 Dados Técnicos

### Dataset de Treinamento (v1)
- 18 datasets do Roboflow
- 24.794 imagens totais
- 63.912 anotações
- Objetos educacionais diversos

### Evolução do Projeto
- **v1**: MobileNetSSD com treinamento customizado (48h+)
- **v2**: YOLO11m pré-treinado (otimizado)

## 🤝 Contribuidores

- Gabriel Gonçalves ([gonntt](https://github.com/gonntt))
- Marco Antônio ([macostacurta](https://github.com/macostacurta))
- Vinicius Gomes ([vini-gm](https://github.com/vini-gm/))
- Victor Laurentino ([PitzTech](https://github.com/PitzTech))

## 🎓 Contexto Acadêmico

Este projeto foi desenvolvido como parte do trabalho final do curso, representando uma implementação prática de tecnologias de IA para acessibilidade educacional. O sistema demonstra a integração efetiva de múltiplas tecnologias emergentes para resolver desafios reais de inclusão.

## 🔮 Trabalhos Futuros

- Expansão para dispositivos móveis
- Suporte a mais navegadores
- Detecção 3D com informações de profundidade
- Reconhecimento de texto (OCR)
- Navegação indoor avançada
- Aprendizado adaptativo personalizado

## 📝 Limitações Conhecidas

- Dependência de boa iluminação
- Dificuldades com objetos parcialmente obstruídos
- Compatibilidade limitada a navegadores Chromium
- Necessidade de hardware adequado para performance ótima

## 📄 Licença

Este projeto é desenvolvido para fins acadêmicos e de pesquisa em acessibilidade.

---

**Sistema de Assistência Visual** - Promovendo autonomia e inclusão através da tecnologia.
