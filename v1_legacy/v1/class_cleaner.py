'''
Class Cleaner for Educational Object Detection
Removes duplicate, irrelevant, and low-quality classes from the dataset
Creates a clean, focused model for educational environments
'''

import json
import os
from collections import Counter

class ClassCleaner:
    """
    Cleans and filters classes for educational object detection
    Removes duplicates, irrelevant classes, and improves model focus
    """
    
    def __init__(self, model_info_path="models/educational_objects/model_info.json"):
        self.model_info_path = model_info_path
        self.original_classes = []
        self.cleaned_classes = []
        self.class_mapping = {}
        self.removed_classes = []
        
    def load_original_classes(self):
        """Load original classes from model info"""
        if os.path.exists(self.model_info_path):
            with open(self.model_info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
                self.original_classes = model_info['classes']
                print(f"Loaded {len(self.original_classes)} original classes")
        else:
            print("Model info file not found!")
            
    def define_educational_relevant_classes(self):
        """
        Define which classes are relevant for educational environments
        Focus on classroom and study materials
        """
        educational_relevant = {
            # Core stationery
            'book', 'notebook', 'pen', 'pencil', 'eraser', 'ruler', 'scissors',
            'stapler', 'calculator', 'markers', 'sharpener',
            
            # Digital devices
            'laptop', 'computer', 'tablet', 'ipad', 'keyboard', 'mouse',
            'cell phone', 'mobile phone', 'charger', 'flash drive',
            
            # Study materials
            'bag', 'backpack', 'glasses', 'water bottle', 'cup',
            'papers', 'envelope', 'clip', 'staples',
            
            # Classroom items
            'chair', 'desk', 'whiteboard', 'projector'
        }
        
        return educational_relevant
    
    def normalize_class_name(self, class_name):
        """Normalize class names for comparison"""
        name = class_name.lower().strip()
        
        # Remove common prefixes/suffixes
        name = name.replace('_', ' ').replace('-', ' ')
        name = ' '.join(name.split())  # Remove extra spaces
        
        # Handle common variations
        variations = {
            'ballpen': 'pen',
            'ballpoint': 'pen',
            'pulpen': 'pen',
            'signpen': 'pen',
            'fudepen': 'pen',
            'pensil': 'pencil',
            'b_pencil': 'pencil',
            'k_pencil': 'pencil',
            'o_pencil': 'pencil',
            'apple-pencil': 'pencil',
            'penghapus': 'eraser',
            'd_eraser': 'eraser',
            'e_eraser': 'eraser',
            'o_eraser': 'eraser',
            'penggaris': 'ruler',
            'scale': 'ruler',
            'pengserut': 'sharpener',
            'sharpner': 'sharpener',
            'g_sharpner': 'sharpener',
            'p_sharpner': 'sharpener',
            'y_sharpner': 'sharpener',
            'waterbottle': 'water bottle',
            'mobile phone': 'cell phone',
            'charging-cable': 'charger',
            'phone charger': 'charger',
            'correction-tape': 'correction tape',
            'correctiontape': 'correction tape',
            'gluestick': 'glue stick',
            'paper clip': 'clip',
            'paper clips': 'clip',
            'stick note': 'sticky note',
            'ipad-air': 'ipad',
            'ipad-pro': 'ipad',
            'computer host': 'computer',
            'pc': 'computer',
            'lecture-notes': 'notebook',
            'copypaper': 'paper'
        }
        
        return variations.get(name, name)
    
    def filter_irrelevant_classes(self):
        """Remove classes that are not relevant for educational environments"""
        irrelevant_patterns = [
            # Metadata and descriptions
            'roboflow', 'github', 'dataset', 'images', 'classes', 'total', 'visit',
            'collaborate', 'understand', 'use active learning', 'v1 2022',
            
            # Random objects not in classrooms
            'dogs', 'butane_gas', 'nail_clipper', 'nail_polish', 'normal_lighter',
            'toiletpapier', 'towel', 'wipe', 'tube', 'mask', 'gum',
            'home-appliances', 'food', 'food_container', 'fork', 'spoon',
            'blade', 'blades', 'box_cutter', 'knife', 'plier',
            
            # Single characters or meaningless
            '-', 'd', 'object', 'objects', 'equipment', 'deskitems',
            'stationery', 'stationery-btwm', 'study_desk',
            
            # Duplicates with better names
            'eraser-', 'scale-', 'blunt'
        ]
        
        filtered_classes = []
        for class_name in self.original_classes:
            normalized = self.normalize_class_name(class_name)
            
            # Skip if matches irrelevant patterns
            skip = False
            for pattern in irrelevant_patterns:
                if pattern in class_name.lower() or pattern == normalized:
                    skip = True
                    break
            
            if not skip and len(normalized) > 1:  # Keep meaningful names
                filtered_classes.append(normalized)
                
        return filtered_classes
    
    def remove_duplicates(self, classes):
        """Remove duplicate classes while preserving order"""
        seen = set()
        unique_classes = []
        
        for class_name in classes:
            if class_name not in seen:
                seen.add(class_name)
                unique_classes.append(class_name)
                
        return unique_classes
    
    def create_clean_classes(self):
        """Create the final clean class list"""
        print("\n=== CLEANING CLASSES ===")
        
        # Step 1: Filter irrelevant classes
        filtered = self.filter_irrelevant_classes()
        print(f"After filtering irrelevant: {len(filtered)} classes")
        
        # Step 2: Remove duplicates
        unique = self.remove_duplicates(filtered)
        print(f"After removing duplicates: {len(unique)} classes")
        
        # Step 3: Sort alphabetically for better organization
        self.cleaned_classes = sorted(unique)
        
        # Track what was removed
        self.removed_classes = [c for c in self.original_classes 
                              if self.normalize_class_name(c) not in self.cleaned_classes]
        
        print(f"Final clean classes: {len(self.cleaned_classes)} classes")
        print(f"Removed classes: {len(self.removed_classes)} classes")
        
        return self.cleaned_classes
    
    def print_comparison(self):
        """Print before/after comparison"""
        print("\n" + "="*60)
        print("CLASS CLEANING RESULTS")
        print("="*60)
        
        print(f"\nORIGINAL: {len(self.original_classes)} classes")
        print(f"CLEANED: {len(self.cleaned_classes)} classes")
        print(f"REMOVED: {len(self.removed_classes)} classes")
        print(f"REDUCTION: {((len(self.original_classes) - len(self.cleaned_classes)) / len(self.original_classes) * 100):.1f}%")
        
        print(f"\n=== FINAL CLEAN CLASSES ({len(self.cleaned_classes)}) ===")
        for i, class_name in enumerate(self.cleaned_classes, 1):
            print(f"{i:2d}. {class_name}")
        
        print(f"\n=== REMOVED CLASSES ({len(self.removed_classes)}) ===")
        for class_name in self.removed_classes[:20]:  # Show first 20
            print(f"  - {class_name}")
        if len(self.removed_classes) > 20:
            print(f"  ... and {len(self.removed_classes) - 20} more")
    
    def save_cleaned_model_info(self, output_path="models/educational_objects_clean/"):
        """Save cleaned model information"""
        os.makedirs(output_path, exist_ok=True)
        
        # Create new class mapping
        cleaned_mapping = {class_name: idx for idx, class_name in enumerate(self.cleaned_classes)}
        
        # Load original model info
        with open(self.model_info_path, 'r', encoding='utf-8') as f:
            original_info = json.load(f)
        
        # Create cleaned model info
        cleaned_info = {
            "classes": self.cleaned_classes,
            "class_mapping": cleaned_mapping,
            "input_size": original_info.get("input_size", [300, 300]),
            "num_classes": len(self.cleaned_classes),
            "original_num_classes": len(self.original_classes),
            "classes_removed": len(self.removed_classes),
            "description": "Cleaned educational objects model - focused on classroom items",
            "cleaning_info": {
                "removed_classes": self.removed_classes,
                "cleaning_date": "2024",
                "focus": "Educational and classroom environments"
            }
        }
        
        # Save cleaned info
        clean_info_path = os.path.join(output_path, "clean_model_info.json")
        with open(clean_info_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_info, f, indent=2, ensure_ascii=False)
        
        print(f"\nCleaned model info saved to: {clean_info_path}")
        return clean_info_path

def main():
    """Main function to clean classes"""
    print("=== EDUCATIONAL OBJECT CLASS CLEANER ===")
    
    # Create cleaner
    cleaner = ClassCleaner()
    
    # Load original classes
    cleaner.load_original_classes()
    
    # Clean classes
    cleaner.create_clean_classes()
    
    # Print comparison
    cleaner.print_comparison()
    
    # Save cleaned model info
    cleaner.save_cleaned_model_info()
    
    print("\n" + "="*60)
    print("RECOMMENDATION: Retrain your model with these clean classes")
    print("Benefits:")
    print("  - Faster training and inference")
    print("  - Better accuracy on relevant objects")  
    print("  - Less confusion between similar classes")
    print("  - More focused for educational use")
    print("="*60)

if __name__ == "__main__":
    main()