import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA

# Importez les fonctions de votre modèle DeepP
from process.CNN import get_training_data, CNNModel

# Définir la taille des images
img_size = 150

# Définir les transformations pour le prétraitement des données
transform = transforms.Compose([
    transforms.ToTensor(),  # Convertir l'image en tenseur
    transforms.Resize((img_size, img_size)),  # Redimensionner l'image
])

# Définir la classe CustomDataset
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img = self.dataframe.iloc[idx]['image']
        label = self.dataframe.iloc[idx]['label']
        if self.transform:
            img = self.transform(img)
        return img, label

# Charger les données d'entraînement, de validation et de test
labels = ['PNEUMONIA', 'NORMAL']
train_df = get_training_data(r'C:/Users/yonim/OneDrive - Ynov/Deep learning/chest_xray/train', labels)
test_df = get_training_data(r'C:\Users/yonim/OneDrive - Ynov/Deep learning/chest_xray/test', labels)
val_df = get_training_data(r'C:/Users/yonim/OneDrive - Ynov/Deep learning/chest_xray/val', labels)

# Créer les ensembles de données d'entraînement, de validation et de test
train_dataset = CustomDataset(train_df, transform=transform)
val_dataset = CustomDataset(val_df, transform=transform)
test_dataset = CustomDataset(test_df, transform=transform)

# Initialiser le DataLoader
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=20)
test_loader = DataLoader(test_dataset, batch_size=20)

# Initialiser le modèle en utilisant la taille des images définie précédemment
model = CNNModel(img_size)

# Charger les poids pré-entraînés si nécessaire
# model.load_state_dict(torch.load("path_to_pretrained_weights"))

# Calculer les prédictions, la matrice de confusion et le rapport de classification
def get_predictions_and_metrics(model, test_loader):
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Pas besoin d'ajouter une dimension supplémentaire avec unsqueeze(1)
            outputs = model(inputs.float())  
            predictions.extend((outputs > 0.5).int().tolist())
            true_labels.extend(labels.int().tolist())

    cm = confusion_matrix(true_labels, predictions)
    classification_rep = classification_report(true_labels, predictions, labels=[0, 1])


    
    return predictions, true_labels, cm, classification_rep

predictions, true_labels, cm, classification_rep = get_predictions_and_metrics(model, test_loader)

# Créer la matrice de confusion sous forme de heatmap
fig_confusion_matrix = go.Figure(data=go.Heatmap(z=cm, x=labels, y=labels,
                                                 colorscale='Blues'))
fig_confusion_matrix.update_layout(title="Matrice de confusion",
                                   xaxis_title="Prédiction",
                                   yaxis_title="Vraie valeur")

# Réduire la dimensionnalité des caractéristiques extraites à 2 pour la visualisation
pca = PCA(n_components=2)
features = []
with torch.no_grad():
    for inputs, _ in test_loader:
        # Obtenir les caractéristiques extraites du modèle
        outputs = model.conv4(inputs.float())
        features.extend(outputs.numpy())

# Appliquer PCA aux caractéristiques extraites
pca_features = pca.fit_transform(features)
pca_df = pd.DataFrame(pca_features, columns=['Component 1', 'Component 2'])

# Créer le scatter plot pour visualiser les caractéristiques extraites
fig_features = px.scatter(pca_df, x='Component 1', y='Component 2', title='Visualisation des caractéristiques extraites')

# Ajouter les imports nécessaires pour la comparaison des performances d'entraînement et de test
import plotly.express as px

# Supposons que vous avez les listes train_losses et val_losses contenant les pertes d'entraînement et de validation
num_epochs = 12
train_losses = np.random.rand(num_epochs)  # Exemple de liste de pertes d'entraînement
val_losses = np.random.rand(num_epochs)  # Exemple de liste de pertes de validation

# Créer le trace pour la comparaison des performances d'entraînement et de test
losses_df = pd.DataFrame({'Training Loss': train_losses, 'Validation Loss': val_losses})
fig_losses = go.Figure()
fig_losses.add_trace(go.Scatter(x=np.arange(1, num_epochs + 1), y=losses_df['Training Loss'], mode='lines', name='Training Loss'))
fig_losses.add_trace(go.Scatter(x=np.arange(1, num_epochs + 1), y=losses_df['Validation Loss'], mode='lines', name='Validation Loss'))
fig_losses.update_layout(title='Comparaison des performances d\'entraînement et de test',
                         xaxis_title='Epoch',
                         yaxis_title='Loss')

# Initialisation de l'application Dash
app = dash.Dash(__name__)

# Mise en page de l'application Dash
app.layout = html.Div([
    html.H1("Analyse des performances du modèle"),
    dcc.Graph(id='confusion-matrix', figure=fig_confusion_matrix),
    html.H3("Analyse des performances par classe"),
    html.Pre(classification_rep, style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-all'}),
    dcc.Graph(id='features-visualization', figure=fig_features),
    dcc.Graph(id='losses-comparison', figure=fig_losses)
])

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
