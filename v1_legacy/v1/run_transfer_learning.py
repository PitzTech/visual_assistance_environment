'''
Main script to run transfer learning training
Integrates the trainer and data loader for complete training pipeline
'''

import os
import sys
import tensorflow as tf
from transfer_learning_trainer import TransferLearningTrainer, configure_gpu
from data_loader_transfer import TransferLearningDataLoader
import json

def main():
    """
    Main function to run transfer learning training
    """
    print("=" * 60)
    print("TRANSFER LEARNING FOR EDUCATIONAL OBJECT DETECTION")
    print("=" * 60)
    
    # Configure GPU
    configure_gpu()
    
    # Create data loader
    print("\n1. Setting up data loader...")
    data_loader = TransferLearningDataLoader(input_size=(300, 300))
    
    # Load educational classes from existing model
    data_loader.load_educational_classes()
    
    # Create combined class mapping
    combined_mapping = data_loader.create_combined_mapping()
    
    # Save combined class information
    data_loader.save_combined_class_info()
    
    # Create transfer learning trainer
    print("\n2. Setting up transfer learning trainer...")
    trainer = TransferLearningTrainer(
        input_size=(300, 300),
        num_educational_classes=len(data_loader.educational_classes)
    )
    
    # Load educational classes into trainer
    trainer.educational_classes = data_loader.educational_classes
    trainer.total_classes = len(data_loader.combined_classes)
    
    # Create and compile model
    print("\n3. Creating transfer learning model...")
    trainer.load_pretrained_weights()
    trainer.compile_model(learning_rate=0.001)
    
    # Print model summary
    print("\n4. Model architecture:")
    trainer.model.summary()
    
    # Setup data generators
    print("\n5. Setting up data generators...")
    
    # For demonstration, we'll use the generator with dummy data
    # In practice, you would provide actual dataset paths
    train_generator = data_loader.create_training_generator(
        coco_data_path=None,  # Replace with actual COCO path
        educational_data_path="datasets/educational_objects/",
        batch_size=16,
        augment=True
    )
    
    val_generator = data_loader.create_training_generator(
        coco_data_path=None,  # Replace with actual COCO path
        educational_data_path="datasets/educational_objects/",
        batch_size=16,
        augment=False
    )
    
    # Train model
    print("\n6. Starting transfer learning training...")
    print("Phase 1: Training with frozen backbone...")
    
    # First phase: train with frozen backbone
    history1 = trainer.model.fit(
        train_generator,
        steps_per_epoch=50,  # Adjust based on your dataset size
        epochs=20,
        validation_data=val_generator,
        validation_steps=10,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                'models/transfer_learning_checkpoint.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                verbose=1,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ],
        verbose=1
    )
    
    # Second phase: unfreeze and fine-tune
    print("\nPhase 2: Fine-tuning with unfrozen layers...")
    
    # Unfreeze some layers for fine-tuning
    for layer in trainer.model.layers[-10:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    trainer.compile_model(learning_rate=0.0001)
    
    # Fine-tune
    history2 = trainer.model.fit(
        train_generator,
        steps_per_epoch=50,
        epochs=15,
        validation_data=val_generator,
        validation_steps=10,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                'models/transfer_learning_final.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                verbose=1,
                restore_best_weights=True
            )
        ],
        verbose=1
    )
    
    # Save final model
    print("\n7. Saving transfer learning model...")
    trainer.save_transfer_model("models/transfer_learning/")
    
    # Save training history
    training_history = {
        "phase1_history": {
            "loss": history1.history.get("loss", []),
            "val_loss": history1.history.get("val_loss", []),
            "class_output_accuracy": history1.history.get("class_output_accuracy", []),
            "val_class_output_accuracy": history1.history.get("val_class_output_accuracy", [])
        },
        "phase2_history": {
            "loss": history2.history.get("loss", []),
            "val_loss": history2.history.get("val_loss", []),
            "class_output_accuracy": history2.history.get("class_output_accuracy", []),
            "val_class_output_accuracy": history2.history.get("val_class_output_accuracy", [])
        }
    }
    
    with open("models/transfer_learning/training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("TRANSFER LEARNING TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Final model saved to: models/transfer_learning/")
    print(f"Total classes: {trainer.total_classes}")
    print(f"COCO classes: {len(trainer.coco_classes)}")
    print(f"Educational classes: {len(trainer.educational_classes)}")
    print("=" * 60)

if __name__ == "__main__":
    main()