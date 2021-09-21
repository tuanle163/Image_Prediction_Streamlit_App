import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications import InceptionV3

img_augmentation = tf.keras.Sequential([
    # tf.keras.layers.experimental.preprocessing.Resizing(224, 224),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2, fill_mode='nearest'),
    tf.keras.layers.experimental.preprocessing.RandomZoom(.6, .6)
])

def create_model():
    base_model_InV3 = InceptionV3(weights="imagenet",
                       input_shape=(224, 224, 3), 
                       include_top=False)

    base_model_InV3.trainable=True

    # Fine-tune from 
    fine_tune_at = 280

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model_InV3.layers[:fine_tune_at]:
        layer.trainable =  False

    inputs_2 = keras.Input(shape=(224, 224, 3))
    x_2 = keras.applications.inception_v3.preprocess_input(inputs_2)
    x_2 = img_augmentation(x_2)
    x_2 = base_model_InV3(x_2, training=False)
    x_2 = keras.layers.GlobalAveragePooling2D()(x_2)
    x_2 = keras.layers.Dense(2048,activation='relu')(x_2)
    x_2 = keras.layers.Dropout(0.5)(x_2)
    x_2 = keras.layers.Dense(2048,activation='relu')(x_2)
    outputs_2 = keras.layers.Dense(9, activation='softmax')(x_2)
    model_InV3 = keras.Model(inputs_2, outputs_2)

    model_InV3.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model_summary = model_InV3.summary()

    return model_InV3, model_summary

def test_single(image, model):
    label_to_index = {'1000': 0, '10000': 1, '100000': 2, 
    '2000': 3, '20000': 4, '200000': 5, 
    '5000': 6, '50000': 7, '500000': 8}

    
    image = tf.image.resize(image, [224, 224])
    img_array  = tf.keras.preprocessing.image.img_to_array(image)
    img_array  = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    for key, value in label_to_index.items():
        if value == prediction[0].argmax():
            pred = key

    return pred