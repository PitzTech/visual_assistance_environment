'''
Remove Metadata Classes from Educational Dataset
Removes Roboflow metadata and description classes that don't represent actual objects
'''

import json
import os
import shutil

def remove_metadata_classes():
    """
    Remove specific metadata classes from the educational dataset
    """
    model_info_path = "models/educational_objects/model_info.json"
    
    # Classes to remove (exact matches)
    classes_to_remove = [
        "- - v1 2022-09-02 1-50pm",
        "- collaborate with your team on computer vision projects", 
        "- understand and search unstructured image data",
        "- use active learning to improve your dataset over time",
        "4 classes",
        "object",
        "objects", 
        "roboflow is an end-to-end computer vision platform that helps you",
        "the dataset includes 101 images-",
        "total 350 images",
        "visit https-github-com-roboflow-notebooks"
    ]
    
    if not os.path.exists(model_info_path):
        print(f"Model info file not found: {model_info_path}")
        return
    
    # Backup original file
    backup_path = model_info_path + ".backup"
    shutil.copy2(model_info_path, backup_path)
    print(f"Backup created: {backup_path}")
    
    # Load original model info
    with open(model_info_path, 'r', encoding='utf-8') as f:
        model_info = json.load(f)
    
    original_classes = model_info['classes']
    original_class_mapping = model_info['class_mapping']
    
    print(f"Original classes: {len(original_classes)}")
    
    # Remove metadata classes
    cleaned_classes = []
    removed_classes = []
    
    for class_name in original_classes:
        if class_name in classes_to_remove:
            removed_classes.append(class_name)
            print(f"Removing: '{class_name}'")
        else:
            cleaned_classes.append(class_name)
    
    print(f"\nRemoved {len(removed_classes)} metadata classes:")
    for removed in removed_classes:
        print(f"  - '{removed}'")
    
    print(f"\nCleaned classes: {len(cleaned_classes)}")
    
    # Create new class mapping
    new_class_mapping = {class_name: idx for idx, class_name in enumerate(cleaned_classes)}
    
    # Update model info
    model_info['classes'] = cleaned_classes
    model_info['class_mapping'] = new_class_mapping
    
    # Add cleaning info
    if 'cleaning_info' not in model_info:
        model_info['cleaning_info'] = {}
    
    model_info['cleaning_info'].update({
        'metadata_removed': True,
        'removed_metadata_classes': removed_classes,
        'original_class_count': len(original_classes),
        'cleaned_class_count': len(cleaned_classes),
        'classes_removed_count': len(removed_classes)
    })
    
    # Save cleaned model info
    with open(model_info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print(f"\nUpdated model info saved to: {model_info_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("METADATA CLASSES REMOVAL SUMMARY")
    print("="*60)
    print(f"Original classes: {len(original_classes)}")
    print(f"Cleaned classes: {len(cleaned_classes)}")  
    print(f"Removed metadata: {len(removed_classes)}")
    print(f"Reduction: {len(removed_classes)} classes ({len(removed_classes)/len(original_classes)*100:.1f}%)")
    print("="*60)
    
    # Show first 20 remaining classes
    print(f"\nFirst 20 remaining classes:")
    for i, class_name in enumerate(cleaned_classes[:20], 1):
        print(f"{i:2d}. {class_name}")
    
    if len(cleaned_classes) > 20:
        print(f"... and {len(cleaned_classes) - 20} more")

def main():
    """Main function"""
    print("=== REMOVING METADATA CLASSES ===")
    remove_metadata_classes()
    print("\nMetadata classes removed successfully!")
    print("Your model now has only actual object classes.")

if __name__ == "__main__":
    main()