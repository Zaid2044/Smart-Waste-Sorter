# train_model.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import config # Import our configuration

# --- 1. Data Preparation ---
# Use ImageDataGenerator for loading and augmenting images.
# Augmentation creates modified versions of images (rotated, zoomed, etc.)
# which makes the model more robust and prevents overfitting.
datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values from [0, 255] to [0, 1]
    validation_split=0.2,    # Reserve 20% of the data for validation
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# --- Training Data Generator ---
train_generator = datagen.flow_from_directory(
    'dataset', # Path to the main dataset folder
    target_size=(config.IMG_WIDTH, config.IMG_HEIGHT),
    batch_size=32,
    class_mode='binary', # Because we have two classes
    subset='training'    # Specify this is the training set
)

# --- Validation Data Generator ---
validation_generator = datagen.flow_from_directory(
    'dataset',
    target_size=(config.IMG_WIDTH, config.IMG_HEIGHT),
    batch_size=32,
    class_mode='binary',
    subset='validation'  # Specify this is the validation set
)

# Crucial: Check the class indices. This tells us which folder corresponds to 0 and 1.
# Write this down! For example: {'non_recyclable': 0, 'recyclable': 1}
print("Class Indices Found:", train_generator.class_indices)

# --- 2. Model Building (Using Transfer Learning) ---
# Load a pre-trained model (MobileNetV2) without its top classification layer.
# MobileNetV2 is lightweight and great for mobile/edge devices.
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(config.IMG_WIDTH, config.IMG_HEIGHT, 3))

# Freeze the layers of the base model. We don't want to change the learned features.
base_model.trainable = False

# Add our custom classification layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x) # Averages the spatial features
x = Dense(128, activation='relu')(x) # A new fully-connected layer for our specific task
x = Dropout(0.5)(x) # Dropout helps prevent overfitting
# The final prediction layer. A single neuron with a 'sigmoid' activation function
# will output a value between 0 and 1, perfect for binary classification.
predictions = Dense(1, activation='sigmoid')(x)

# Combine the base model with our new layers
model = Model(inputs=base_model.input, outputs=predictions)

# --- 3. Compile the Model ---
# We configure the model for training.
model.compile(
    optimizer=Adam(learning_rate=0.0001), # Adam is a popular, effective optimizer
    loss='binary_crossentropy', # A standard loss function for binary classification
    metrics=['accuracy'] # We want to monitor the accuracy during training
)

# --- 4. Train the Model ---
print("Starting model training...")
history = model.fit(
    train_generator,
    epochs=10, # An epoch is one full pass through the entire training dataset
    validation_data=validation_generator
)

# --- 5. Save the Trained Model ---
# The trained model is saved so we can use it later without retraining.
model.save('waste_sorter_model.h5')
print("Training complete! Model saved as 'waste_sorter_model.h5'")