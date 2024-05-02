import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, LSTM, concatenate, Bidirectional
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

    x = Conv2D(32, (3, 3), strides=1, padding='same', activation='relu')(input_layer_cnn)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2), strides=2, padding='same')(x)
    x = Conv2D(64, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2), strides=2, padding='same')(x)
    x = Conv2D(64, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2), strides=2, padding='same')(x)
    x = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2), strides=2, padding='same')(x)
    x = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 2), strides=2, padding='same')(x)
    x = Flatten()(x)

    cnn_model = Model(inputs=input_layer_cnn, outputs=x)
    return cnn_model

def build_rnn_model(input_shape):
    input_layer_rnn = Input(shape=(None, cnn_model.output_shape[1]))

    rnn_layer = Bidirectional(LSTM(units=256, return_sequences=True))(input_layer_rnn)
    rnn_layer = Bidirectional(LSTM(units=128, return_sequences=True))(rnn_layer)
    rnn_layer = LSTM(units=64, return_sequences=True)(rnn_layer)

    concatenated_outputs = concatenate([rnn_layer, rnn_layer])
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

# Example usage:

labels = ['PNEUMONIA', 'NORMAL']
img_size = 150

train_df = get_data('./../../radio_pictures/chest_xray/train', labels, img_size)
test_df = get_data('./../../radio_pictures/chest_xray/test', labels, img_size)
val_df = get_data('./../../radio_pictures/chest_xray/val', labels, img_size)

# Exploratrion des données.


# Compter les occurrences de chaque étiquette
label_counts = train_df['label'].value_counts()

# Tracer le countplot avec des couleurs différentes
sns.set_style('darkgrid')
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=train_df, palette=['blue', 'green'])
plt.title('Nombre d\'images pneumonia (pneumonie) et normal dans l\'ensemble de données d\'entraînement')
plt.xlabel('Étiquette')
plt.ylabel('Nombre d\'images')
plt.xticks(ticks=[0, 1], labels=['Pneumonia', 'Normal'])
plt.show()

# Affichage d'une image de Pneumonia
plt.figure(figsize=(5, 5))
plt.imshow(train_df[train_df['label'] == 0]['image'].iloc[0], cmap='gray')
plt.title('Pneumonia')

# Affichage d'une image de Normal
plt.figure(figsize=(5, 5))
plt.imshow(train_df[train_df['label'] == 1]['image'].iloc[0], cmap='gray')
plt.title('Normal')

plt.show()


x_train, y_train, x_val, y_val, x_test, y_test = preprocess_data(train_df, test_df, val_df, img_size)

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

# Imprimer la perte et l'accuracy à chaque epoch
for epoch in range(1, 13):
    train_loss = history.history['loss'][epoch - 1]
    train_accuracy = history.history['accuracy'][epoch - 1]
    val_loss = history.history['val_loss'][epoch - 1]
    val_accuracy = history.history['val_accuracy'][epoch - 1]
    print(f"Epoch {epoch}/12 - Loss: {train_loss:.4f} - Accuracy: {train_accuracy:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")

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

# Créer un DataFrame pour la matrice de confusion
cm1_df = pd.DataFrame(cm1, index=['0', '1'], columns=['0', '1'])

# Tracer la heatmap de la matrice de confusion
plt.figure(figsize=(8, 8))
sns.heatmap(cm1_df, cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()