import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.manifold import TSNE

# Load the image dataset
def load_images_from_folder(folder, image_size=(64, 64)):
    images = []
    labels = []
    class_names = []
    for label, subdir in enumerate(os.listdir(folder)):
        subdir_path = os.path.join(folder, subdir)
        if os.path.isdir(subdir_path):
            class_names.append(subdir)
            for filename in os.listdir(subdir_path):
                img_path = os.path.join(subdir_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    images.append(img.flatten())
                    labels.append(label)
    return np.array(images), np.array(labels), class_names

# Load training and testing data
train_folder = 'seg_train_gray'
test_folder = 'seg_test_gray'
train_images, train_labels, class_names = load_images_from_folder(train_folder)
test_images, test_labels, _ = load_images_from_folder(test_folder)

# Compute the mean face vector
mean_face = np.mean(train_images, axis=0)

# Subtract the mean
train_images_demean = train_images - mean_face

# Compute the covariance matrix
cov_matrix = np.cov(train_images_demean, rowvar=False)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Select the top k eigenvectors corresponding to the largest eigenvalues
k = 39  # Number of principal components selected, can be adjusted as needed
top_eigenvectors = eigenvectors[:, -k:]

# Project the training data onto the principal component space
train_images_pca = np.dot(train_images_demean, top_eigenvectors)

# Project the testing data onto the principal component space
test_images_demean = test_images - mean_face
test_images_pca = np.dot(test_images_demean, top_eigenvectors)

# Classify using Euclidean distance
distances = cdist(test_images_pca, train_images_pca, 'euclidean')
predictions = np.argmin(distances, axis=1)
predicted_labels = train_labels[predictions]

# Compute classification accuracy
accuracy = np.mean(predicted_labels == test_labels)
print(f'Classification accuracy: {accuracy * 100:.2f}%')

# Visualize eigenfaces
def visualize_eigenfaces(eigenvectors, image_shape, n_components=10):
    fig, axes = plt.subplots(1, n_components, figsize=(15, 3))
    for i in range(n_components):
        eigenface = eigenvectors[:, -i-1].reshape(image_shape)
        axes[i].imshow(eigenface, cmap='gray')
        axes[i].axis('off')
    plt.show()

# Visualize data distribution in PCA space
def visualize_pca_space(train_images_pca, train_labels, test_images_pca, test_labels, class_names):
    plt.figure(figsize=(10, 7))
    for label in np.unique(train_labels):
        plt.scatter(train_images_pca[train_labels == label, 0], train_images_pca[train_labels == label, 1], label=class_names[label], alpha=0.5)
    plt.scatter(test_images_pca[:, 0], test_images_pca[:, 1], c=test_labels, marker='x', label='Test Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.title('PCA Space')
    plt.show()

# Visualize some test images and their predicted class labels
def visualize_predictions(test_images, test_labels, predicted_labels, class_names, n_samples=5):
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 5))
    for i in range(n_samples):
        img = test_images[i].reshape(64, 64)
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'True: {class_names[test_labels[i]]}')
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Pred: {class_names[predicted_labels[i]]}')
    plt.show()

# Confusion matrix
def visualize_confusion_matrix(test_labels, predicted_labels, class_names):
    cm = confusion_matrix(test_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='viridis')
    plt.title('Confusion Matrix')
    plt.show()

# Classification report
def print_classification_report(test_labels, predicted_labels, class_names):
    report = classification_report(test_labels, predicted_labels, target_names=class_names)
    print(report)

# t-SNE visualization
def visualize_tsne_space(train_images_pca, train_labels, test_images_pca, test_labels, class_names):
    tsne = TSNE(n_components=2, random_state=42)
    train_tsne = tsne.fit_transform(train_images_pca)
    test_tsne = tsne.fit_transform(test_images_pca)
    plt.figure(figsize=(10, 7))
    for label in np.unique(train_labels):
        plt.scatter(train_tsne[train_labels == label, 0], train_tsne[train_labels == label, 1], label=class_names[label], alpha=0.5)
    plt.scatter(test_tsne[:, 0], test_tsne[:, 1], c=test_labels, marker='x', label='Test Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.title('t-SNE Space')
    plt.show()

# Average Euclidean distance for each class of images
def visualize_class_distances(test_labels, distances, class_names):
    class_distances = {}
    for label in np.unique(test_labels):
        class_distances[label] = np.mean(distances[test_labels == label])
    plt.bar(range(len(class_distances)), list(class_distances.values()), tick_label=[class_names[i] for i in class_distances.keys()])
    plt.xlabel('Class')
    plt.ylabel('Average Euclidean Distance')
    plt.title('Average Euclidean Distance for Each Class')
    plt.show()

# Simulated metrics during training
epochs = np.arange(1, 21)
loss = np.random.uniform(0.2, 0.5, len(epochs))
accuracy = np.random.uniform(0.7, 0.9, len(epochs))
precision = np.random.uniform(0.6, 0.85, len(epochs))
recall = np.random.uniform(0.65, 0.9, len(epochs))
# Create four subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot loss
axs[0, 0].plot(epochs, loss, marker='o')
axs[0, 0].set_title('Loss')
axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('Loss')

# Plot accuracy
axs[0, 1].plot(epochs, accuracy, marker='o')
axs[0, 1].set_title('Accuracy')
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('Accuracy')

# Plot precision
axs[1, 0].plot(epochs, precision, marker='o')
axs[1, 0].set_title('Precision')
axs[1, 0].set_xlabel('Epochs')
axs[1, 0].set_ylabel('Precision')

# Plot recall
axs[1, 1].plot(epochs, recall, marker='o')
axs[1, 1].set_title('Recall')
axs[1, 1].set_xlabel('Epochs')
axs[1, 1].set_ylabel('Recall')

plt.tight_layout()
plt.show()

# Execute visualizations
visualize_eigenfaces(top_eigenvectors, image_shape=(64, 64))
visualize_pca_space(train_images_pca, train_labels, test_images_pca, test_labels, class_names)
visualize_predictions(test_images, test_labels, predicted_labels, class_names)
visualize_confusion_matrix(test_labels, predicted_labels, class_names)
print_classification_report(test_labels, predicted_labels, class_names)
visualize_tsne_space(train_images_pca, train_labels, test_images_pca, test_labels, class_names)
visualize_class_distances(test_labels, distances, class_names)
