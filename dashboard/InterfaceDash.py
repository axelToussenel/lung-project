import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import datetime
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import base64
import cv2
import os

# Importer les fonctions du modèle combiné
from RNN import get_data, build_combined_model, preprocess_data, create_datagen, build_cnn_model, build_rnn_model, train_model

# Initialisation de l'application Dash
app = dash.Dash(__name__)

# Entrainement de l'ensemble de données
@app.callback(Output('summary', 'figure'),
              Output('confusion-matrix', 'figure'),
              Output('loss', 'children'),
              Output('accuracy', 'children'),
              Output('classification-rep', 'children'),
              Input('train-button', 'n_clicks')) 
def init_dashboard(n_clicks):
    print('init_dashboard')
    if n_clicks is None:
        print('error')
        raise dash.exceptions.PreventUpdate
    else:
        labels = ['PNEUMONIA', 'NORMAL']
        train_df = get_data('.\\..\\radio_pictures\\chest_xray\\train', labels, 150)
        test_df = get_data('.\\..\\radio_pictures\\chest_xray\\test', labels, 150)
        val_df = get_data('.\\..\\radio_pictures\\chest_xray\\val', labels, 150)

        x_train, y_train, x_val, y_val, x_test, y_test = preprocess_data(train_df, test_df, val_df, 150)

        true_labels = np.random.randint(0, 2, size=100)  # Étiquettes de test générées aléatoirement
        predictions = np.random.randint(0, 2, size=100)  # Prédictions de test générées aléatoirement

        # Calcul de la matrice de confusion
        cm = confusion_matrix(true_labels, predictions)

        # Utilisation de la fonction pour créer le générateur de données
        datagen = create_datagen(rotation_range=30, zoom_range=0.2, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, vertical_flip=False)

        datagen.fit(x_train)

        cnn_model = build_cnn_model()
        rnn_output = build_rnn_model(cnn_model.output_shape[1:])
        combined_model = build_combined_model(cnn_model, rnn_output.shape[1])



        # Display model summary
        summary = combined_model.summary()

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
        loss = ("La perte du modèle est :", loss)
        accuracy = ("L'accuracy du modèle est :", accuracy * 100, "%")

        predictions = combined_model.predict([x_test, rnn_x_test])

        # Round the predictions to get binary values (0 or 1)
        predictions = np.round(predictions)
        
        # Convertir les prédictions en valeurs binaires
        predictions_binary = [1 if pred > 0.5 else 0 for pred in predictions]

        # Métriques d'évaluation

        # Imprimer le rapport de classification
        classification_rep = classification_report(y_test, predictions_binary, target_names=['Pneumonia (Class 0)', 'Normal (Class 1)'])

        # Calculer la matrice de confusion
        cm1 = confusion_matrix(y_test_reshaped, predictions_binary)

        # Créer un DataFrame pour la matrice de confusion
        cm1_df = pd.DataFrame(cm1, index=['0', '1'], columns=['0', '1'])

        # Tracer la heatmap de la matrice de confusion
        fig_confusion_matrix = go.Figure(data=go.Heatmap(z=cm, x=['PNEUMONIA', 'NORMAL'], y=['PNEUMONIA', 'NORMAL'],
                                                        colorscale='Blues'))
        fig_confusion_matrix.update_layout(title="Matrice de confusion",
                                        xaxis_title="Prédiction",
                                        yaxis_title="Vraie valeur")
        return summary, fig_confusion_matrix, loss, accuracy, classification_rep


# Entrainement du modèle pour des images sélectionnées
@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            html.Div([
                html.H5(filename),
                html.Img(src=contents, style={'width': '50%'}),
                html.Hr(),
                html.H5(datetime.datetime.fromtimestamp(date))
            ]) for contents, filename, date in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


# Définir les styles CSS
styles = {
    'dashboard': {
        'textAlign': 'center'
    },
    'dashboard-logo': {
        'height': '60px', 'marginRight': '15px'
    },
    'dashboard-header': {
        'backgroundColor': '#282c34',
        'minHeight': '10vh',
        'display': 'flex',
        'flexDirection': 'row',
        'alignItems': 'center',
        'justifyContent': 'space-between',
        'fontSize': '20px',
        'color': 'white',
        'paddingLeft': '10px',
        'borderBottom': '3px solid #61dafb'
    },
    'title-container': {
        'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center'
    },
    'dashboard-title': {
        'color': '#61dafb', 'textTransform': 'uppercase', 'letterSpacing': '2px'
    },
    'dashboard-button': {
        'padding': '10px 20px',
        'fontSize': '20px',
        'color': '#fff',
        'backgroundColor': '#61dafb',
        'border': 'none',
        'borderRadius': '5px',
        'cursor': 'pointer',
        'marginRight': '20px',
        'transition': 'background-color 0.3s ease'
    },
    'dashboard-button:hover': {
        'backgroundColor': '#4098da'
    },
    'dashboard-layout': {
        'display': 'flex',
        'flexWrap': 'wrap',
        'justifyContent': 'space-around',
        'gridTemplateColumns': 'repeat(auto-fill, minmax(200px, 1fr))',
        'gap': '20px',
        'padding': '20px'
    },
    'dashboard-chart': {
        'width': '45%', 'height': '400px', 'border': '1px solid #ccc', 'margin': '10px'
    },
    'dashboard-table': {
        'width': '45%', 'height': '300px', 'border': '1px solid #ccc', 'margin': '10px'
    }
}

# Mise en page de l'application Dash
app.layout = html.Div(
    style=styles['dashboard'],
    children=[
        html.Header(
            style=styles['dashboard-header'],
            children=[
                html.Div(
                    style=styles['title-container'],
                    children=[
                        html.Img(
                            style=styles['dashboard-logo'],
                            src="https://img.icons8.com/color/48/lungs.png",
                            alt="logo"
                        ),
                        html.H1(
                            style=styles['dashboard-title'],
                            children="Pneumonia Detection"
                        )
                    ]
                ),
                html.Button(
                    id='train-button',
                    style=styles['dashboard-button'],
                    children="Lancer l'entrainement du modèle",
                    n_clicks=1),
            ]
        ),
        html.Div([
            html.H1("Analyse des performances du modèle", style={'textAlign': 'center'}),
        ]),
        html.Div(
            style=styles['dashboard-layout'],
            children=[
                dcc.Graph(id='summary', style=styles['dashboard-table']),
                dcc.Graph(id='confusion-matrix', style=styles['dashboard-chart']),
                html.Pre(id='loss', style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-all'}),
                html.Pre(id='accuracy', style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-all'}),
            ]
        ),
        html.Div(
            children=[
                html.H2("Matrice de confusion :", style={'textAlign': 'center', 'margin': 'auto', 'alignItems': 'center'}),
                html.Pre(id='classification-rep', style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-all'}),
        ]),
        html.Div([
            html.H2("Tester des radios pulmonaires", style={'textAlign': 'center', 'marginTop': '80px'}),
            dcc.Upload(
                id='upload-image',
                children=html.Div([
                    'Glisser-déposer ou ',
                    html.A('Sélectionner une image', 
                    style={'color': 'blue', 'textDecoration': 'none'},
                    className='upload-link')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Autoriser plusieurs fichiers à être téléchargés
                multiple=True
            ),
            html.Div(id='output-image-upload'),
        ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
