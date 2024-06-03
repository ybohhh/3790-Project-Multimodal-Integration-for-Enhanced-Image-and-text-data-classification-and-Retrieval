import os
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.manifold import TSNE
import seaborn as sns

# Set the data paths
train_folder = '3790 dataset/seg_train'
test_folder = '3790 dataset/seg_test'

# Data preprocessing and augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(10),      # Random rotation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the datasets
train_dataset = ImageFolder(root=train_folder, transform=train_transform)
test_dataset = ImageFolder(root=test_folder, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Define the model
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_dataset.classes))  # Adjusting the output layer
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Store the loss and accuracy during training
train_losses = []
test_accuracies = []

# Store all predicted labels and true labels
all_preds = []
all_labels = []

# Train the model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    test_accuracies.append(accuracy)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')

    # Update the learning rate
    scheduler.step()

# Visualize the changes in loss and accuracy during training
plt.figure(figsize=(12, 5))

# Plot the loss curve
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)

# Plot the accuracy curve
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy', marker='o', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy over Epochs')
plt.legend()
plt.grid(True)

plt.show()

# Confusion matrix
def visualize_confusion_matrix(all_labels, all_preds, class_names):
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='viridis')
    plt.title('Confusion Matrix')
    plt.show()

# Classification report
def print_classification_report(all_labels, all_preds, class_names):
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)

# t-SNE visualization
def visualize_tsne(model, data_loader, class_names, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, label in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.extend(outputs.cpu().numpy())
            labels.extend(label.cpu().numpy())

    tsne = TSNE(n_components=2, random_state=42)
    tsne_features = tsne.fit_transform(features)
    tsne_df = pd.DataFrame(tsne_features, columns=['Component 1', 'Component 2'])
    tsne_df['label'] = labels

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='Component 1', y='Component 2', hue='label', palette='viridis', data=tsne_df, legend='full')
    plt.title('t-SNE Visualization of Features')
    plt.legend(class_names)
    plt.show()

# Visualization
class_names = train_dataset.classes
visualize_confusion_matrix(all_labels, all_preds, class_names)
print_classification_report(all_labels, all_preds, class_names)
visualize_tsne(model, test_loader, class_names, device)
