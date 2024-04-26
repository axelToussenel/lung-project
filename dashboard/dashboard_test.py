import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Exemple de données fictives pour les graphiques
true_labels = np.random.randint(0, 2, size=100)  # Étiquettes de test générées aléatoirement
predictions = np.random.randint(0, 2, size=100)  # Prédictions de test générées aléatoirement

# Calcul de la matrice de confusion
cm = confusion_matrix(true_labels, predictions)

# Calcul des métriques de classification
classification_rep = classification_report(true_labels, predictions, target_names=['PNEUMONIA', 'NORMAL'])

# Initialisation de l'application Dash
app = dash.Dash(__name__)

# Création de la matrice de confusion sous forme de heatmap
fig_confusion_matrix = go.Figure(data=go.Heatmap(z=cm, x=['PNEUMONIA', 'NORMAL'], y=['PNEUMONIA', 'NORMAL'],
                                                 colorscale='Blues'))
fig_confusion_matrix.update_layout(title="Matrice de confusion",
                                   xaxis_title="Prédiction",
                                   yaxis_title="Vraie valeur")

# Création de la visualisation des performances par classe
data_performance = {
    'Classe': ['PNEUMONIA', 'NORMAL'],
    'Précision': [0.85, 0.92],  # Exemple de précision
    'Rappel': [0.78, 0.95],  # Exemple de rappel
    'F-mesure': [0.81, 0.93]  # Exemple de F-mesure
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
                    n_clicks=0
                )
            ]
        ),
        html.Div([
            html.H1("Analyse des performances du modèle", style={'textAlign': 'center'}),
        ]),
        html.Div(
            style=styles['dashboard-layout'],
            children=[
                dcc.Graph(id='features-table', figure=fig_features, style=styles['dashboard-table']),
                dcc.Graph(id='performance-comparison-table', figure=fig_performance_comparison, style=styles['dashboard-table']),
                dcc.Graph(id='confusion-matrix', figure=fig_confusion_matrix, style=styles['dashboard-chart']),
                dcc.Graph(id='performance-graph', figure=fig_performance, style=styles['dashboard-chart']),
                html.H2("Matrice de confusion :", style={'textAlign': 'center', 'margin': 'auto', 'alignItems': 'center'}),
                html.Pre(classification_rep, style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-all'}),
            ]
        )
])

if __name__ == '__main__':
    app.run_server(debug=True)