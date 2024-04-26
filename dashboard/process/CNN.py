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

class CNNModel(torch.nn.Module):
    def __init__(self, img_size):
        super(CNNModel, self).__init__()
        # Modifier la première couche de convolution pour accepter un seul canal
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        # Les autres couches restent inchangées
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = torch.nn.Conv2d(128, 1, kernel_size=3)

        # Calculer la taille de sortie des couches de convolution
        self.fc_input_size = self._get_conv_output_size((1, img_size, img_size))
        self.fc1 = torch.nn.Linear(self.fc_input_size, 512)
        self.fc2 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv4(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(-1, self.fc_input_size)  # Redimensionner les données en fonction de la taille calculée
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

    def _get_conv_output_size(self, shape):
        x = torch.rand(1, *shape)
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv4(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        return x.size(1) * x.size(2) * x.size(3)

# Initialiser le modèle en utilisant la taille des images définie précédemment
model = CNNModel(img_size)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)

num_epochs = 12
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        # Transmettre les données au modèle sans utiliser unsqueeze
        outputs = model(inputs)  # Supprimer unsqueeze(1)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

test_losses = []
test_accs = []
with torch.no_grad():
    for inputs, labels in test_loader:
        # Redimensionner les données d'entrée pour correspondre aux attentes du modèle
        # Pas besoin de convertir les données en float ici
        outputs = model(inputs)
        test_loss = criterion(outputs.squeeze(), labels.float())
        test_losses.append(test_loss.item())
        test_acc = ((outputs > 0.5).float() == labels.unsqueeze(1).float()).float().mean()
        test_accs.append(test_acc.item())
mean_test_loss = np.mean(test_losses)
mean_test_acc = np.mean(test_accs)
print(f'Test Loss: {mean_test_loss:.4f}, Test Accuracy: {mean_test_acc:.4f}')

if __name__ == "__main__":
    # Votre code d'exemple pour le test du modèle et la génération de graphiques...
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Pas besoin d'ajouter une dimension supplémentaire avec unsqueeze(1)
            outputs = model(inputs.float())  
            predictions.extend((outputs > 0.5).int().tolist())
            true_labels.extend(labels.int().tolist())

    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, cmap='Blues', linecolor='black', linewidth=1, annot=True, fmt='', xticklabels=['PNEUMONIA', 'NORMAL'], yticklabels=['PNEUMONIA', 'NORMAL'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
