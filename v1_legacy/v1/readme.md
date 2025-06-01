# CNN Visual Assistance

This project implements a **Convolutional Neural Network (CNN) for Visual Assistance**, specifically designed to help identify objects in educational environments for people with visual impairments.

## 🎯 Purpose

**Main Goal**: Create an AI-powered visual assistance system that helps people with visual impairments navigate educational environments by identifying and describing the location of classroom objects in real-time.

## ✨ Features

- **Real-time object detection** using MobileNetSSD architecture
- **Educational object recognition** (books, pens, calculators, laptops, etc.)
- **Audio feedback** for detected objects with spatial positioning
- **Transfer learning support** (COCO + Educational datasets)
- **Clean dataset management** with automated class filtering
- **GPU acceleration** (RTX 4080 Super optimized)
- **Instant detection** - objects tracked continuously without manual capture

## 🚀 Quick Start

### 1. Environment Setup
```bash
python3.9 -m venv env
```

#### Windows
```bash
.\env\Scripts\activate
```

#### macOS/Linux
```bash
source env/bin/activate
```

```bash
pip install -r requirements.txt
```

#### Additional Requirements (Linux)
```bash
sudo apt-get install protobuf-compiler
```

### 2. Basic Usage (Recommended for beginners)
```bash
# Start with educational object detection
python object_detection_system.py
```

### 3. Advanced Usage (Recommended for best results)
```bash
# Clean your dataset first (removes 60%+ irrelevant classes)
python class_cleaner.py

# Train transfer learning model (COCO + Educational)
python run_transfer_learning.py

# Use advanced detection system
python transfer_learning_detector.py
```

### 4. Environment Cleanup
```bash
deactivate
```

## 🎮 Controls

**During Detection**:
- `q`: Quit application
- `s`: Toggle audio feedback on/off
- **Automatic detection**: Objects are detected and tracked in real-time

## 📁 Project Files Guide

| File | Purpose | When to Use |
|------|---------|-------------|
| `object_detection_system.py` | Basic educational detection | First-time users, educational-only objects |
| `transfer_learning_detector.py` | Advanced detection (COCO + Educational) | Best accuracy, comprehensive detection |
| `trainer_v3.py` | Train educational model | Educational objects only |
| `run_transfer_learning.py` | Train combined model | Best performance, general + educational |
| `class_cleaner.py` | Clean dataset classes | Remove duplicates, optimize performance |
| `remove_metadata_classes.py` | Remove Roboflow metadata | Clean specific unwanted classes |

📖 **[Complete Usage Guide](PROJECT_GUIDE.md)** - Detailed documentation for all files and workflows

## 🧠 Model Architecture

- **Backbone**: MobileNetV2 (ImageNet pre-trained)
- **Detection**: Single Shot MultiBox Detector (SSD)
- **Input Size**: 300×300 pixels
- **Framework**: TensorFlow/Keras
- **Classes**: 109 educational objects (cleaned from 120)

## 📊 Dataset Information

### Current Dataset (After Cleaning)
- **Total Classes**: 109 (reduced from 120)
- **Focus**: Educational and classroom objects
- **Removed**: Metadata, duplicates, irrelevant items

### Supported Objects
- **Stationery**: Books, pens, pencils, erasers, rulers, scissors
- **Electronics**: Laptops, tablets, calculators, phones, chargers
- **Accessories**: Bags, glasses, water bottles, notebooks
- **And more educational items...**

## 💻 Hardware Requirements

| Component | Recommended | Minimum |
|-----------|-------------|---------|
| **GPU** | NVIDIA RTX 4080 Super | Any CUDA-compatible GPU |
| **RAM** | 16GB+ | 8GB |
| **Storage** | 10GB free space | 5GB |
| **CPU** | Modern multi-core | Any (CPU fallback available) |

## 📈 Performance Optimization

1. **Clean Dataset**: Use `class_cleaner.py` (reduces classes by ~63%)
2. **Transfer Learning**: Use `run_transfer_learning.py` for better accuracy
3. **Remove Metadata**: Use `remove_metadata_classes.py` for Roboflow datasets
4. **GPU Acceleration**: Automatically detected and configured

## 🔧 Workflow Recommendations

### For New Users:
1. `python object_detection_system.py` (test basic functionality)
2. `python class_cleaner.py` (optimize dataset)
3. `python run_transfer_learning.py` (train advanced model)
4. `python transfer_learning_detector.py` (use best model)

### For Advanced Users:
1. Analyze dataset with `class_cleaner.py`
2. Clean specific classes with `remove_metadata_classes.py`
3. Train with cleaned data using preferred trainer
4. Deploy with appropriate detector

## 📱 Output Features

The system provides:
- **Visual**: Real-time bounding boxes with confidence scores
- **Audio**: Object names with spatial positions ("pen detected in center")
- **JSON**: Detailed coordinates and metadata
- **Position**: Spatial descriptions (left/right/center, top/middle/bottom)
- **Type**: Object classification (COCO vs Educational)

## 🔍 Troubleshooting

**Common Issues**:
- **No camera**: Check connections and permissions
- **Model not found**: Run training scripts first
- **Low accuracy**: Clean dataset and use transfer learning
- **Too slow**: Ensure GPU acceleration is working

**Solutions**:
- Run `python test_simple.py` for basic verification
- Check `PROJECT_GUIDE.md` for detailed troubleshooting

## 🤝 Contributing

1. Clean datasets before training
2. Test both basic and transfer learning approaches
3. Document new features clearly
4. Follow established naming conventions
5. Consider accessibility impact

## 📚 Documentation

- **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)**: Complete usage documentation
- **[ROBOFLOW_USAGE.md](ROBOFLOW_USAGE.md)**: Dataset management guide
- **Code Comments**: Detailed inline documentation

## 🌟 Key Improvements

This project has been enhanced with:
- ✅ **Real-time detection** (no manual capture needed)
- ✅ **Transfer learning** support (COCO + Educational)
- ✅ **Dataset cleaning** tools (63% class reduction)
- ✅ **Metadata removal** (Roboflow cleanup)
- ✅ **Comprehensive documentation**
- ✅ **Multiple detection modes** (basic vs advanced)
- ✅ **Optimized performance** (GPU acceleration)

## 📞 Support

For detailed usage instructions, troubleshooting, and advanced features, see **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)**.

## 🛠️ Development Commands

```bash
# Freeze current packages
pip freeze > requirements.txt

# System monitoring
htop    # CPU/Memory usage
nvtop   # GPU usage
```

---

**Remember**: This system is designed to improve accessibility and independence for people with visual impairments in educational environments. Every enhancement in accuracy and usability directly impacts the quality of life for users with visual disabilities.