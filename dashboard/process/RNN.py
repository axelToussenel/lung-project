import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from sklearn.metrics import confusion_matrix

img_size = 150
sequence_length = 1  # Longueur de la séquence pour chaque entrée dans le RNN

def get_training_data(data_dir, labels):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # Charger l'image en niveaux de gris
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Redimensionner les images à la taille préférée
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    # Convertir les données en DataFrame
    df = pd.DataFrame(data, columns=['image', 'label'])
    return df

# Exemple d'utilisation:
labels = ['PNEUMONIA', 'NORMAL']
train_df = get_training_data(r'C:/Users/yonim/OneDrive - Ynov/Deep learning/chest_xray/train', labels)
test_df = get_training_data(r'C:\Users/yonim/OneDrive - Ynov/Deep learning/chest_xray/test', labels)
val_df = get_training_data(r'C:/Users/yonim/OneDrive - Ynov/Deep learning/chest_xray/val', labels)

# Définir les transformations pour le prétraitement des données
transform = transforms.Compose([
    transforms.ToTensor(),  # Convertir l'image en tenseur
    transforms.Resize((img_size, img_size)),  # Redimensionner l'image
])

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

# Créer les ensembles de données d'entraînement, de validation et de test
train_dataset = CustomDataset(train_df, transform=transform)
val_dataset = CustomDataset(val_df, transform=transform)
test_dataset = CustomDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=20)
test_loader = DataLoader(test_dataset, batch_size=20)

class RNNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialiser l'état caché initial à zéro
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        # Passer les données à travers le RNN
        out, _ = self.rnn(x, h0)
        # Obtenir la sortie de la dernière étape temporelle
        out = self.fc(out[:, -1, :])
        return out

# Initialiser le modèle en utilisant la taille des images définie précédemment
model = RNNModel(img_size, 128, 2, 1)  # Utilisation de 128 unités cachées et 2 couches dans le RNN

criterion = torch.nn.BCEWithLogitsLoss()  # Utilisation de BCEWithLogitsLoss pour la classification binaire
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)

# Supposons que vos images aient 3 canaux (RGB)
num_channels = 3

# Entraînement du modèle
num_epochs = 12
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Réorganiser les dimensions des données d'entrée
        inputs = inputs.view(inputs.size(0), -1)  # Aplatir chaque image individuellement
        # Remettre à zéro les gradients
        optimizer.zero_grad()
        # Transmettre les données au modèle
        outputs = model(inputs)
        # Calculer la perte
        loss = criterion(outputs.squeeze(), labels.float())
        # Rétropropagation et mise à jour des poids
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Évaluation du modèle
model.eval()
test_losses = []
test_accs = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.view(inputs.size(0), -1)  # Aplatir chaque image individuellement
        outputs = model(inputs)
        test_loss = criterion(outputs.squeeze(), labels.float())
        test_losses.append(test_loss.item())
        test_acc = ((outputs > 0.5).float() == labels.unsqueeze(1).float()).float().mean()
        test_accs.append(test_acc.item())
mean_test_loss = np.mean(test_losses)
print(f'Test Loss: {mean_test_loss:.4f}, Test Accuracy: {mean_test_acc:.4f}')

# Génération de la matrice de confusion
predictions = []
true_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.view(-1, img_size * img_size)
        outputs = model(inputs)
        predictions.extend((outputs > 0.5).int().tolist())
        true_labels.extend(labels.int().tolist())

cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, cmap='Blues', linecolor='black', linewidth=1, annot=True, fmt='', xticklabels=['PNEUMONIA', 'NORMAL'], yticklabels=['PNEUMONIA', 'NORMAL'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()