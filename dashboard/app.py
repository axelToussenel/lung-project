import dash
from dash import dcc, html
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


app = dash.Dash(__name__)
app.run_server(debug=True)

class Dashboard:
    def __init__(self, true_labels, predictions, cm):
        self.true_labels = true_labels
        self.predictions = predictions
        self.cm = cm

    def matriceDeConfusion(self):
        # Création de la matrice de confusion sous forme de heatmap
        fig_confusion_matrix = go.Figure(data=go.Heatmap(z=self.cm, x=['PNEUMONIA', 'NORMAL'], y=['PNEUMONIA', 'NORMAL'],
            colorscale='Blues'))
        return fig_confusion_matrix.update_layout(title="Matrice de confusion",
            xaxis_title="Prédiction",
            yaxis_title="Vraie valeur")
        
    def figDePerf():
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
        return fig_performance.update_layout(barmode='group', title="Analyse des performances par classe")
        # Ajout des données factices pour les graphiques 3 et 4
        training_performance = {'Précision': 0.85, 'Rappel': 0.78, 'F-mesure': 0.81}
        test_performance = {'Précision': 0.82, 'Rappel': 0.75, 'F-mesure': 0.78}

    def tableauCaracteristiques():
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
        return fig_features.update_layout(title="Visualisation des caractéristiques extraites")

    def tableauComparaison():
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
        return fig_performance_comparison.update_layout(title="Comparaison des performances d'entraînement et de test")

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
        'width': '45%', 'height': '200px', 'border': '1px solid #ccc', 'margin': '10px'
    },
    'dashboard-table': {
        'width': '45%', 'height': '200px', 'border': '1px solid #ccc', 'margin': '10px'
    }
}

true_labels = np.random.randint(0, 2, size=100)  # Étiquettes de test générées aléatoirement
predictions = np.random.randint(0, 2, size=100)  # Prédictions de test générées aléatoirement
# Calcul de la matrice de confusion
cm = confusion_matrix(true_labels, predictions)
# Calcul des métriques de classification
classification_rep = classification_report(true_labels, predictions, target_names=['PNEUMONIA', 'NORMAL'])
dashboard = Dashboard(true_labels, predictions, cm)

# App layout
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
        html.Div(
            style=styles['dashboard-layout'],
            children=[
                html.Div('ca marche'),
                html.H1("Analyse des performances du modèle"),
                html.Pre(classification_rep, style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-all'}),
                dcc.Graph(id='confusion-matrix', figure=dashboard.matriceDeConfusion(), style=styles['dashboard-chart']),
                dcc.Graph(id='performance-graph', figure=dashboard.figDePerf, style=styles['dashboard-chart']),
                dcc.Graph(id='features-table', figure=dashboard.tableauCaracteristiques, style=styles['dashboard-chart']),
                dcc.Graph(id='performance-comparison-table', figure=dashboard.tableauComparaison, style=styles['dashboard-chart']),
                html.H3("Analyse des performances par classe"),
            ]
        )
    ]
)


if __name__ == '__main__':
    app.run_server(debug=True)