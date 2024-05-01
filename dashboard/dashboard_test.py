import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report



# Initialisation de l'application Dash
app = dash.Dash(__name__)

#entrainement de l'ensemble de données
@app.callback(Output('features-table', 'figure'),
              Output('performance-comparison-table', 'figure'),
              Output('confusion-matrix', 'figure'),
              Output('performance-graph', 'figure'),
              Output('confusion-matrix-text', 'children'),
    Input('train-button', 'n_clicks'))
def init_dashboard(n_clicks):
    print("entree dans init_dashboard")
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    else:
        # Exemple de données fictives pour les graphiques
        true_labels = np.random.randint(0, 2, size=100)  # Étiquettes de test générées aléatoirement
        predictions = np.random.randint(0, 2, size=100)  # Prédictions de test générées aléatoirement

        # Calcul de la matrice de confusion
        cm = confusion_matrix(true_labels, predictions)
        # Calcul des métriques de classification
        classification_rep = classification_report(true_labels, predictions, target_names=['PNEUMONIA', 'NORMAL'])
        # Création de la matrice de confusion sous forme de heatmap
        fig_confusion_matrix = go.Figure(data=go.Heatmap(z=cm, x=['PNEUMONIA', 'NORMAL'], y=['PNEUMONIA', 'NORMAL'],
                                                        colorscale='Blues'))
        fig_confusion_matrix.update_layout(title="Matrice de confusion",
                                        xaxis_title="Prédiction",
                                        yaxis_title="Vraie valeur")

        # Création de la visualisation des performances par classe
        data_performance = {
            'Classe': ['PNEUMONIA', 'NORMAL'],
            'Précision': [round(random.uniform(0, 1), 2), round(random.uniform(0, 1), 2)],  # Exemple de précision
            'Rappel': [round(random.uniform(0, 1), 2), round(random.uniform(0, 1), 2)],  # Exemple de rappel
            'F-mesure': [round(random.uniform(0, 1), 2), round(random.uniform(0, 1), 2)]  # Exemple de F-mesure
        }
        df_performance = pd.DataFrame(data_performance)
        fig_performance = go.Figure(data=[
            go.Bar(name='Précision', x=df_performance['Classe'], y=df_performance['Précision']),
            go.Bar(name='Rappel', x=df_performance['Classe'], y=df_performance['Rappel']),
            go.Bar(name='F-mesure', x=df_performance['Classe'], y=df_performance['F-mesure'])
        ])
        fig_performance.update_layout(barmode='group', title="Analyse des performances par classe")

        # Ajout des données factices pour les graphiques 3 et 4
        training_performance = {'Précision': 0.85, 'Rappel': 0.78, 'F-mesure': 0.81}
        test_performance = {'Précision': 0.82, 'Rappel': 0.75, 'F-mesure': 0.78}

        # Création du graphique 3 : Visualisation des caractéristiques extraites (tableau)
        fig_features = go.Figure(data=[go.Table(
            header=dict(values=['Couche', 'Activation moyenne'],
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[['Conv1', 'Conv2', 'Conv3', 'Conv4'],  # Noms des couches
                            [0.75, 0.82, 0.68, 0.79]],  # Activations moyennes
                    fill_color='lavender',
                    align='left'))
        ])
        fig_features.update_layout(title="Visualisation des caractéristiques extraites")

        # Création du graphique 4 : Comparaison des résultats d'entraînement et de test
        fig_performance_comparison = go.Figure(data=[go.Table(
            header=dict(values=['Métrique', 'Entraînement', 'Test'],
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[['Précision', 'Rappel', 'F-mesure'],
                            [0.85, 0.78, 0.81],  # Performances d'entraînement
                            [0.82, 0.75, 0.78]],  # Performances de test
                    fill_color='lavender',
                    align='left'))
        ])
        fig_performance_comparison.update_layout(title="Comparaison des performances d'entraînement et de test")
        return fig_features, fig_performance_comparison, fig_confusion_matrix, fig_performance, classification_rep


#entrainement du modèle pour des images sélectionnées
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


# Define CSS styles
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
                dcc.Graph(id='features-table', style=styles['dashboard-table']),
                dcc.Graph(id='performance-comparison-table', style=styles['dashboard-table']),
                dcc.Graph(id='confusion-matrix', style=styles['dashboard-chart']),
                dcc.Graph(id='performance-graph', style=styles['dashboard-chart']),
            ]
        ),
        html.Div(
            children=[
                html.H2("Matrice de confusion :", style={'textAlign': 'center', 'margin': 'auto', 'alignItems': 'center'}),
                html.Pre(id='confusion-matrix-text', style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-all'}),
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