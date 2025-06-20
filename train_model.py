import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import config

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    'dataset',
    target_size=(config.IMG_WIDTH, config.IMG_HEIGHT),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    'dataset',
    target_size=(config.IMG_WIDTH, config.IMG_HEIGHT),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

print("Class Indices Found:", train_generator.class_indices)

base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=(config.IMG_WIDTH, config.IMG_HEIGHT, 3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Starting model training...")

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

model.save('waste_sorter_model.keras')
print("Training complete! Model saved as 'waste_sorter_model.keras'")