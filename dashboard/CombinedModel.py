import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Model, Sequential
from keras.layers import Activation, Add, Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, LSTM, concatenate, Bidirectional, SpatialDropout1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report , confusion_matrix
import cv2
import os


#récupération des données
def get_data(data_dir, labels, img_size):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Redimensionner les images à la taille spécifiée
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    # Convertir les données en DataFrame
    df = pd.DataFrame(data, columns=['image', 'label'])
    return df


def load_and_resize_images(df, img_size, channels=1):
    x = []
    y = []

    for _, row in df.iterrows():
        img = cv2.resize(row['image'], (img_size, img_size))
        if channels == 1:
            img = np.expand_dims(img, axis=-1)  # Ajouter une dimension pour les canaux
        x.append(img)
        y.append(row['label'])

    return np.array(x), np.array(y)


def preprocess_data(train_df, test_df, val_df, img_size):
    # Charger et redimensionner les images
    x_train, y_train = load_and_resize_images(train_df, img_size)
    x_val, y_val = load_and_resize_images(val_df, img_size)
    x_test, y_test = load_and_resize_images(test_df, img_size)

    # Normaliser les données
    x_train = x_train / 255.0
    x_val = x_val / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_val, y_val, x_test, y_test


def create_datagen(rotation_range=30, zoom_range=0.2, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, vertical_flip=False):
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip
    )
    return datagen


def build_cnn_model(input_shape=(150, 150, 1)):
    input_layer_cnn = Input(shape=input_shape)

    # Première couche de convolution avec 64 filtres de taille 3x3
    x = Conv2D(64, (3, 3), strides=1, padding='same')(input_layer_cnn)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2), strides=2, padding='same')(x)

    # Deuxième couche de convolution avec 128 filtres de taille 3x3
    x = Conv2D(128, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2), strides=2, padding='same')(x)

    # Troisième couche de convolution avec 256 filtres de taille 3x3
    x = Conv2D(256, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2), strides=2, padding='same')(x)

    # Quatrième couche de convolution avec 512 filtres de taille 3x3
    x = Conv2D(512, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2), strides=2, padding='same')(x)

    # Cinquième couche de convolution avec 512 filtres de taille 3x3
    x = Conv2D(512, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2), strides=2, padding='same')(x)

    # Ajout d'un bloc résiduel avec deux couches de convolution de 512 filtres
    shortcut = x
    x = Conv2D(512, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = Activation('relu')(x)

    x = Flatten()(x)

    cnn_model = Model(inputs=input_layer_cnn, outputs=x)
    return cnn_model

def build_rnn_model(input_shape):
    input_layer_rnn = Input(shape=(None, cnn_model.output_shape[1]))

    rnn_layer = Bidirectional(LSTM(units=256, return_sequences=True))(input_layer_rnn)
    rnn_layer = SpatialDropout1D(0.3)(rnn_layer)  # Ajout de dropout spatial pour la régularisation
    rnn_layer = Bidirectional(LSTM(units=128, return_sequences=True))(rnn_layer)
    rnn_layer = SpatialDropout1D(0.3)(rnn_layer)  # Ajout de dropout spatial pour la régularisation
    rnn_layer = LSTM(units=64, return_sequences=True)(rnn_layer)

    # Ajout de couches de pooling pour agréger les informations temporelles
    max_pool = GlobalMaxPooling1D()(rnn_layer)
    avg_pool = GlobalAveragePooling1D()(rnn_layer)

    # Concaténation des couches de pooling pour former un vecteur de caractéristiques
    pooled_features = concatenate([max_pool, avg_pool])

    # Ajout d'une couche dense pour extraire des caractéristiques supplémentaires
    dense_layer = Dense(units=64, activation='relu')(pooled_features)
    dense_layer = Dropout(0.3)(dense_layer)

    concatenated_outputs = concatenate([pooled_features, dense_layer])

    return concatenated_outputs

def build_combined_model(cnn_model, rnn_output_shape):
    input_layer_rnn = Input(shape=(None, cnn_model.output_shape[1]))

    combined_features = Dense(units=32, activation='relu')(input_layer_rnn)
    combined_features = Dense(units=32, activation='relu')(combined_features)

    output_layer = Dense(units=128, activation='relu')(combined_features)
    output_layer = Dropout(0.2)(output_layer)
    output_layer = Dense(units=1, activation='sigmoid')(output_layer)

    combined_model = Model(inputs=[cnn_model.input, input_layer_rnn], outputs=output_layer)
    combined_model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])

    return combined_model

def train_model(model, x_train, rnn_x_train, y_train, x_val, rnn_x_val, y_val, epochs, callbacks):
    history = model.fit([x_train, rnn_x_train], y_train, epochs=epochs, validation_data=([x_val, rnn_x_val], y_val), callbacks=callbacks)
    return history

