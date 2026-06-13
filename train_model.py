import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# Download FER2013 dataset from Kaggle first
# https://www.kaggle.com/datasets/msambare/fer2013

def load_and_preprocess_data():
    """Load and preprocess the FER2013 dataset"""
    
    # Path to dataset (adjust based on your download location)
    train_path = "fer2013/train"
    test_path = "fer2013/test"
    
    # Emotion labels
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Using Keras ImageDataGenerator for easy loading
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(48, 48),
        batch_size=64,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=True
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(48, 48),
        batch_size=64,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, test_generator, emotions

def create_model(input_shape=(48, 48, 1), num_classes=7):
    """Create a CNN model for emotion classification"""
    
    model = keras.Sequential([
        # First Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model():
    """Train the emotion classification model"""
    
    print("Loading data...")
    train_gen, test_gen, emotions = load_and_preprocess_data()
    
    print(f"Number of classes: {len(emotions)}")
    print(f"Emotions: {emotions}")
    
    # Create model
    model = create_model()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        keras.callbacks.ModelCheckpoint(
            'models/emotion_model.h5',
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        train_gen,
        epochs=50,
        validation_data=test_gen,
        callbacks=callbacks
    )
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(test_gen)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    return model, emotions

if __name__ == "__main__":
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Train the model
    model, emotions = train_model()