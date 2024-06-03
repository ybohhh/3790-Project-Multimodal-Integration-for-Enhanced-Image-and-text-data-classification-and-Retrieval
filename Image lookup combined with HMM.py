import os
from collections import Counter
import torch
from torchvision import models, transforms
from PIL import Image
import requests
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Set data path
train_folder = '3790 dataset/seg_train'

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load pre-trained ResNet model
model = models.resnet50(pretrained=True)
model.eval()

# Get ImageNet labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(LABELS_URL).json()


# Function to generate image label
def generate_image_label(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = outputs.max(1)
            label = labels[predicted.item()]

        return label
    except Exception as e:
        print(f"Error generating label for {image_path}: {e}")
        return None


# Function to generate labels for images in a folder
def generate_labels_for_folder(folder_path):
    image_labels = {}
    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                if image_file.endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(class_path, image_file)
                    label = generate_image_label(image_path)
                    if label:
                        image_labels[os.path.join(class_folder, image_file)] = label
    return image_labels


# Generate labels for the training set
train_image_labels = generate_labels_for_folder(train_folder)
print("Train Image Labels:", train_image_labels)

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Example text data
texts = [
    "The bird flies in the forest."
]

# Compute TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
vectorizer.fit(texts)
feature_names = vectorizer.get_feature_names_out()


# Function to extract keywords
def extract_keywords(text, top_n=3):
    tfidf_vector = vectorizer.transform([text]).toarray().flatten()
    indices = np.argsort(tfidf_vector)[::-1]
    top_indices = indices[:top_n]
    keywords = [feature_names[i] for i in top_indices]
    return keywords


# Example text
example_text = "The bird flies in the forest."
keywords = extract_keywords(example_text)
print("Extracted Keywords:", keywords)


# Application of HMM model
def hmm_keyword_extraction(texts, n_components=3, top_n=3):
    # Tokenize and label encode the texts
    all_words = [word for text in texts for word in nltk.word_tokenize(text.lower())]
    le = LabelEncoder()
    le.fit(all_words)
    all_words_encoded = le.transform(all_words).reshape(-1, 1)

    # Train HMM model
    model = hmm.MultinomialHMM(n_components=n_components, n_iter=100, tol=0.01)
    model.fit(all_words_encoded)

    # Calculate emission probabilities for each word
    word_probs = model.predict_proba(all_words_encoded)
    word_prob_sums = word_probs.sum(axis=0)

    # Get total emission probabilities for each word
    word_scores = dict(zip(le.classes_, word_prob_sums))

    # Sort by emission probabilities and extract keywords
    sorted_keywords = sorted(word_scores, key=word_scores.get, reverse=True)
    top_keywords = sorted_keywords[:top_n]

    return top_keywords


# Example text data
example_texts = [
    "The bird flies in the forest.",
    "A serene lake surrounded by forest.",
    "The beach is full of tourists.",
    "The desert is vast and dry.",
    "A river flows through the valley.",
    "The sky is clear and blue."
]

# HMM keyword extraction
hmm_keywords = hmm_keyword_extraction(example_texts, n_components=3, top_n=3)
print("HMM Extracted Keywords:", hmm_keywords)


# Function to find images by keywords
def find_images_by_keywords(keywords, image_labels):
    matched_images = []
    for image_file, label in image_labels.items():
        if any(keyword.lower() in label.lower() for keyword in keywords):
            matched_images.append(image_file)
    return matched_images


# Find images matching HMM keywords
matched_train_images = find_images_by_keywords(hmm_keywords, train_image_labels)
print("Matched Train Images:", matched_train_images)


# Function to display images
def show_images(images, folder_path, title):
    plt.figure(figsize=(20, 10))
    columns = 5
    for i, image_file in enumerate(images):
        image_path = os.path.join(folder_path, image_file)
        img = Image.open(image_path)
        plt.subplot(len(images) // columns + 1, columns, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(image_file)
    plt.suptitle(title)
    plt.show()


# Display matched training set images
show_images(matched_train_images, train_folder, 'Matched Train Images')


# Function to compute confusion matrix
def compute_confusion_matrix(true_labels, predicted_labels, class_names):
    cm = confusion_matrix(true_labels, predicted_labels, labels=class_names)
    return cm


# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


# Example classification results
true_labels = [train_image_labels[img] for img in matched_train_images]
predicted_labels = [generate_image_label(os.path.join(train_folder, img)) for img in matched_train_images]

# Compute and visualize confusion matrix
cm = compute_confusion_matrix(true_labels, predicted_labels, labels)
plot_confusion_matrix(cm, labels)

# Generate classification report
print(classification_report(true_labels, predicted_labels, target_names=labels))

# Example text data using TF-IDF and HMM
example_texts_2 = [
    "A majestic mountain covered with snow.",
    "A bustling city street at night.",
    "A quiet forest path with tall trees.",
    "Waves crashing on a sandy beach.",
    "A river flowing through the countryside.",
    "A clear sky over a desert landscape."
]

# Extract and display keywords
for text in example_texts_2:
    tfidf_keywords = extract_keywords(text, top_n=3)
    hmm_keywords = hmm_keyword_extraction([text], n_components=3, top_n=3)
    print(f"Text: {text}")
    print(f"TF-IDF Keywords: {tfidf_keywords}")
    print(f"HMM Keywords: {hmm_keywords}")
    print()

# Find and display images matching new keywords
tfidf_matched_images = find_images_by_keywords(tfidf_keywords, train_image_labels)
hmm_matched_images = find_images_by_keywords(hmm_keywords, train_image_labels)

# Display matched images
show_images(tfidf_matched_images, train_folder, 'TF-IDF Matched Train Images')
show_images(hmm_matched_images, train_folder, 'HMM Matched Train Images')
