# Roboflow Datasets Integration

This guide explains how to use your Roboflow datasets with the adapted training script.

## Overview

The code has been successfully adapted to work with multiple Roboflow datasets automatically. It can:

- **Auto-discover** all Roboflow datasets in your `datasets/` directory
- **Handle different folder structures** (train/valid/test or single folder)
- **Combine multiple datasets** into a unified training set
- **Normalize class names** across different datasets
- **Support filtering** to use only specific datasets

## Your Datasets

The test found **18 Roboflow datasets** with **24,794 images** and **63,912 annotations**:

1. AI Based Automatic Stationery Billing System Data.v1i.coco (621 images)
2. Stationery.v1i.coco (503 images) 
3. v1i.coco (101 images)
4. obj.v1i.coco (255 images)
5. Pen.v4i.coco (805 images)
6. equipment.v2i.coco (227 images)
7. OD.v1i.coco (933 images)
8. Home Appliances Detection.v1i.coco (1,276 images)
9. Smart School Supply Detector.v1i.coco (2,185 images)
10. Stationery.v1i.coco (1,382 images)
11. Office Tools.v2i.coco (541 images)
12. DaVinci_Resolve_19.1_Windows (1,443 images)
13. yolov8 v1.v1i.coco (1,277 images)
14. school supplies.v1i.coco (139 images)
15. office items 2.1.v2i.coco (3,821 images)
16. object detection stationery.v4i.coco (328 images)
17. Study desk items.v2i.coco (1,208 images)
18. Office Items.v26i.coco (7,749 images)

## Usage Options

### Option 1: Interactive Training (Recommended)

Run the training script and choose your options interactively:

```bash
python3 trainining_v3.py
```

This will:
1. Ask you to choose between Roboflow datasets (option 1) or synthetic datasets (option 2)
2. Show you all discovered datasets
3. Allow you to filter specific datasets if desired
4. Automatically combine and train on your datasets

### Option 2: Direct Roboflow Training

```bash
python3 trainining_v3.py --roboflow
```

### Option 3: Programmatic Usage

```python
from trainining_v3 import train_with_roboflow_datasets

# Train with all datasets
history, metrics, model_info = train_with_roboflow_datasets(
    datasets_base_dir="datasets",
    dataset_filters=None,  # Use all datasets
    epochs=30,
    save_path='models/roboflow_educational_objects'
)

# Train with filtered datasets
history, metrics, model_info = train_with_roboflow_datasets(
    datasets_base_dir="datasets",
    dataset_filters=['stationery', 'office', 'pen', 'pencil'],  # Only these types
    epochs=50,
    save_path='models/filtered_educational_objects'
)
```

## Key Features

### Automatic Class Normalization

The system automatically normalizes similar class names across datasets:

- `pencil`, `lapis`, `lápis` → `pencil`
- `eraser`, `borracha` → `eraser`
- `ruler`, `régua`, `scale` → `ruler`
- `pen`, `caneta` → `pen`
- And many more...

### Flexible Dataset Filtering

You can filter datasets by name patterns:

```python
# Include only datasets with these terms in their names
dataset_filters = ['stationery', 'office', 'school', 'pen']
```

### Smart Train/Validation Splitting

The system automatically handles:
- Datasets with existing train/valid/test splits
- Datasets without splits (creates 80/20 train/validation split)
- Mixed scenarios (combines appropriately)

## Output

The trained model will be saved with:

- **Model files**: `final_model.h5`, `model_checkpoint.h5`
- **Training plots**: `training_history.png`
- **Model info**: `model_info.json` with classes, metrics, and dataset information

## Testing the Setup

Run the test script to verify everything works:

```bash
python3 test_simple.py
```

This will show you all discovered datasets and their statistics without requiring TensorFlow.

## Recommendations

Based on your datasets, I recommend:

1. **Start with filtered training** using the most relevant datasets:
   ```bash
   # When prompted, use filter: stationery,office,pen,pencil,school
   python3 trainining_v3.py --roboflow
   ```

2. **Use higher epochs** for real datasets (30-50 epochs recommended)

3. **Monitor class distribution** - you have 133 unique classes, which might be too many. Consider filtering to focus on the most important educational objects.

## Class Distribution

Your datasets contain educational objects perfect for the visual assistance system:
- **Stationery**: pencil, pen, eraser, ruler, sharpener
- **Office items**: calculator, notebook, scissors, stapler, clips
- **Study materials**: books, markers, correction tape
- **Technology**: laptop, mouse, keyboard, iPad
- **Personal items**: bag, wallet, keys, phone

The system is ready to train and should work excellent for your educational object detection goals!