import tensorflow as tf
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
import os

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 30
IMG_SIZE = (96, 96)
DATA_DIR = './data'          # Root directory of the dataset, with subfolders for each category 

def create_lightweight_cnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(96,96,3)),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(15, activation='softmax')
    ])
    return model

def main():
    # Data augmentation and normalization
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2          # Use 20% data as validtion set
    )

    # Training generator
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    # Validation generator
    val_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Build model
    model = create_lightweight_cnn()
    model.summary()

    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callback: save the best model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint]
    )

    # Save final model
    model.save('models/final_model.h5')
    print("Model saved to models/final_model.h5")

if __name__ == '__main__':
    main()