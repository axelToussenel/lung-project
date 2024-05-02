import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Model, Sequential
from keras.layers import Activation, Add, Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, LSTM, concatenate, Bidirectional, SpatialDropout1D, GlobalMaxPooling1D, GlobalAveragePooling1D, GRU
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report , confusion_matrix
import cv2
import os


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


def preprocess_data(data_df, img_size):
    # Charger et redimensionner les images
    x_data, y_data = load_and_resize_images(data_df, img_size)

    # Normaliser les données
    x_data = x_data / 255.0

    return x_data, y_data


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

    rnn_layer = Bidirectional(GRU(units=256, return_sequences=True))(input_layer_rnn)  # Remplacement de LSTM par GRU
    rnn_layer = SpatialDropout1D(0.3)(rnn_layer)  # Ajout de dropout spatial pour la régularisation
    rnn_layer = Bidirectional(GRU(units=128, return_sequences=True))(rnn_layer)
    rnn_layer = SpatialDropout1D(0.3)(rnn_layer)  # Ajout de dropout spatial pour la régularisation
    rnn_layer = GRU(units=64, return_sequences=True)(rnn_layer)

    max_pool = GlobalMaxPooling1D()(rnn_layer)
    avg_pool = GlobalAveragePooling1D()(rnn_layer)

    pooled_features = concatenate([max_pool, avg_pool])

    dense_layer = Dense(units=64, activation='relu')(pooled_features)
    dense_layer = Dropout(0.3)(dense_layer)

    concatenated_outputs = concatenate([pooled_features, dense_layer])

    return concatenated_outputs

def build_combined_model(cnn_model, rnn_output_shape):
    input_layer_rnn = Input(shape=(None, cnn_model.output_shape[1]))

    combined_features = Dense(units=32, activation='relu')(input_layer_rnn)
    combined_features = Dense(units=32, activation='relu')(combined_features)

    # Ajout de couches supplémentaires
    combined_features = Dense(units=64, activation='relu')(combined_features)
    combined_features = Dropout(0.3)(combined_features)

    output_layer = Dense(units=128, activation='relu')(combined_features)
    output_layer = Dropout(0.2)(output_layer)
    output_layer = Dense(units=1, activation='sigmoid')(output_layer)

    combined_model = Model(inputs=[cnn_model.input, input_layer_rnn], outputs=output_layer)
    combined_model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])

    return combined_model


def train_model(model, x_train, rnn_x_train, y_train, x_val, rnn_x_val, y_val, epochs, callbacks):
    history = model.fit([x_train, rnn_x_train], y_train, epochs=epochs, validation_data=([x_val, rnn_x_val], y_val), callbacks=callbacks)
    return history

# Example usage:

labels = ['PNEUMONIA', 'NORMAL']
img_size = 150

train_df = get_data('./radio_pictures/chest_xray/train', labels, img_size)
test_df = get_data('./radio_pictures/chest_xray/test', labels, img_size)
val_df = get_data('./radio_pictures/chest_xray/val', labels, img_size)



# Utilisation de la fonction preprocess_data pour les ensembles d'entraînement, de validation et de test
x_train, y_train = preprocess_data(train_df, img_size)
x_val, y_val = preprocess_data(val_df, img_size)
x_test, y_test = preprocess_data(test_df, img_size)

# Utilisation de la fonction pour créer le générateur de données
datagen = create_datagen(rotation_range=30, zoom_range=0.2, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, vertical_flip=False)

datagen.fit(x_train)

cnn_model = build_cnn_model()
rnn_output = build_rnn_model(cnn_model.output_shape[1:])
combined_model = build_combined_model(cnn_model, rnn_output.shape[1])



# Display model summary
print(combined_model.summary())

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.3, min_lr=0.000001)

# Reshape les targets
y_train_reshaped = y_train.reshape(-1, 1)
y_val_reshaped = y_val.reshape(-1, 1)

# Preprocess your data to get CNN-extracted spatial features for each image
cnn_train_features = cnn_model.predict(x_train)
cnn_val_features = cnn_model.predict(x_val)

rnn_x_train = np.expand_dims(cnn_train_features, axis=1)
rnn_x_val = np.expand_dims(cnn_val_features, axis=1) 

# Train the combined model
history = train_model(combined_model, x_train, rnn_x_train, y_train_reshaped, x_val, rnn_x_val, y_val_reshaped, epochs=12, callbacks=[learning_rate_reduction])

# Evaluation du modèle

y_test_reshaped = y_test.reshape(-1, 1)

# Préprocesser vos données de test pour obtenir les caractéristiques spatiales extraites par CNN pour chaque image
cnn_test_features = cnn_model.predict(x_test)

# Répéter les caractéristiques spatiales pour correspondre à la longueur de la séquence d'images
rnn_x_test = np.expand_dims(cnn_test_features, axis=1)

# Évaluer le modèle sur l'ensemble de test
loss, accuracy = combined_model.evaluate([x_test, rnn_x_test], y_test_reshaped)

# Imprimer la perte et l'accuracy
print("La perte du modèle est :", loss)
print("L'accuracy du modèle est :", accuracy * 100, "%")


# Prédiction 

# Make predictions on the test set
predictions = combined_model.predict([x_test, rnn_x_test])

# Round the predictions to get binary values (0 or 1)
predictions = np.round(predictions)

# Convertir les prédictions en valeurs binaires
predictions_binary = [1 if pred > 0.5 else 0 for pred in predictions]

# Métriques d'évaluation

# Imprimer le rapport de classification
print(classification_report(y_test, predictions_binary, target_names=['Pneumonia (Class 0)', 'Normal (Class 1)']))

# Calculer la matrice de confusion
cm1 = confusion_matrix(y_test_reshaped, predictions_binary)
print(cm1)
# Créer un DataFrame pour la matrice de confusion
cm1_df = pd.DataFrame(cm1, index=['0', '1'], columns=['0', '1'])

def extract_performance_metrics(self):
        # Logic to extract performance metrics
        # This could involve accessing attributes of the model, evaluating metrics, etc.
        history = self.history
        loss = self.loss
        accuracy = self.accuracy
        classification_rep = self.classification_rep
        cm1 = self.cm1
        label_counts = self.label_counts
        sample_images = self.sample_images

        return history, loss, accuracy, classification_rep, cm1, label_counts, sample_images

        
# Tracer la heatmap de la matrice de confusion
plt.figure(figsize=(8, 8))
sns.heatmap(cm1_df, cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()