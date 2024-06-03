import os
import shutil
import nltk
import random
import numpy as np
from hmmlearn import hmm
from nltk import FreqDist, bigrams, word_tokenize
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from nltk.corpus import stopwords
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Assuming Flickr30k text descriptions are extracted into the captions.txt file
captions_file = '3790 dataset/captions.txt/captions.txt'
image_folder = '3790 dataset/flickr30k_images'

with open(captions_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Preprocess text
texts = [text.strip() for text in content.split('\n') if text.strip()]
stop_words = set(stopwords.words('english'))

# Processed text list
processed_texts = [' '.join(word for word in nltk.word_tokenize(text.lower())
                            if word.isalpha() and word not in stop_words)
                   for text in texts]

# TF-IDF calculation
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(processed_texts)
tfidf_feature_names = vectorizer.get_feature_names_out()

# Assumed labels
Y = [1 if 'landscape' in text.lower() or 'scenery' in text.lower() else 0 for text in texts]

# Ensure the length of Y matches the number of rows in tfidf_matrix
assert len(Y) == tfidf_matrix.shape[0]

# Chi-square test
chi2_scores, p_values = chi2(tfidf_matrix, Y)
chi2_scores_dict = dict(zip(tfidf_feature_names, chi2_scores))

# Extract TF-IDF and chi-square test results
tfidf_scores = dict(zip(tfidf_feature_names, tfidf_matrix.sum(axis=0).A1))
sorted_tfidf = sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)
sorted_chi2 = sorted(chi2_scores_dict.items(), key=lambda item: item[1], reverse=True)

# Select top 500 keywords
top_tfidf_keywords = [word for word, score in sorted_tfidf[:500]]
top_chi2_keywords = [word for word, score in sorted_chi2[:500]]

# PMI calculation
bigram_measures = BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(word_tokenize(' '.join(processed_texts)))
finder.apply_freq_filter(2)
bigrams_with_pmi = finder.score_ngrams(bigram_measures.pmi)
bigrams_with_pmi_dict = {bigram: score for bigram, score in bigrams_with_pmi}

# Keyword category mapping
landscape_keywords = {
    'building': ['building', 'house', 'cabin', 'church'],
    'forest': ['forest', 'woods', 'trees', 'jungle'],
    'glacier': ['glacier', 'ice', 'snow', 'arctic'],
    'mountain': ['mountain', 'hill', 'peak', 'summit'],
    'sea': ['sea', 'ocean', 'beach', 'shore'],
    'street': ['street', 'road', 'path', 'alley'],
    'river': ['river', 'stream', 'creek', 'waterfall']
}

# Map keywords to categories
def map_keyword_to_category(keyword, keyword_map):
    for category, keywords in keyword_map.items():
        if keyword in keywords:
            return category
    return None

# Extract landscape keywords and map to categories
def extract_landscape_keywords(texts, top_keywords, keyword_map):
    extracted_keywords = []
    for text in texts:
        text_keywords = [word for word in nltk.word_tokenize(text.lower()) if word in top_keywords]
        for word in text_keywords:
            category = map_keyword_to_category(word, keyword_map)
            if category:
                extracted_keywords.append((word, category))
    return extracted_keywords

landscape_keywords_extracted = extract_landscape_keywords(texts, top_tfidf_keywords + top_chi2_keywords,
                                                          landscape_keywords)

# Display extracted keywords and categories
print("Extracted Landscape Keywords and Categories:", landscape_keywords_extracted)

# Build HMM model
class HMMModel:
    def __init__(self, n_components=5):
        self.model = hmm.MultinomialHMM(n_components=n_components)

    def fit(self, sequences):
        lengths = [len(seq) for seq in sequences]
        X = np.concatenate(sequences)
        self.model.fit(X.reshape(-1, 1), lengths)

    def generate(self, length=10):
        _, states = self.model.sample(length)
        return states

# Build and train HMM model
sequences = [np.array([ord(c) for c in text]) for text in texts]
hmm_model = HMMModel()
hmm_model.fit(sequences)

# Load caption file
with open(captions_file, 'r', encoding='utf-8') as f:
    captions = f.readlines()

# Assuming each line format is "index,caption"
captions = [(line.strip().split(',', 1)[0], line.strip().split(',', 1)[1]) for line in captions if ',' in line]

# Print some parsed data for debugging
print("Sample Data:")
for i in range(5):
    print(captions[i])

# Filter image IDs and descriptions containing landscape keywords
landscape_captions = defaultdict(list)
for image_id, caption in captions:
    for word, category in landscape_keywords_extracted:
        if word in caption.lower():
            landscape_captions[category].append((image_id, caption))
            break

print(f"Found {sum(len(captions) for captions in landscape_captions.values())} landscape images and captions.")
print(landscape_captions)

# Define original image folder path and new folder paths
original_image_folder = '3790 dataset/flickr30k_images'
new_image_folder = '3790 dataset/flickr30k_landscape_images'
new_caption_folder = '3790 dataset/flickr30k_landscape_captions'

# Create new folders
os.makedirs(new_image_folder, exist_ok=True)
os.makedirs(new_caption_folder, exist_ok=True)

# Copy filtered images and text descriptions
for category, items in landscape_captions.items():
    category_image_folder = os.path.join(new_image_folder, category)
    category_caption_folder = os.path.join(new_caption_folder, category)
    os.makedirs(category_image_folder, exist_ok=True)
    os.makedirs(category_caption_folder, exist_ok=True)

    for image_id, caption in items:
        image_path = os.path.join(original_image_folder, f"{image_id}.jpg")
        if os.path.exists(image_path):
            print(f"Copying image: {image_path}")  # Adding debugging information
            # Copy image file
            shutil.copy(image_path, category_image_folder)

            # Save text description
            caption_file_path = os.path.join(category_caption_folder, f"{image_id}.txt")
            with open(caption_file_path, 'w', encoding='utf-8') as caption_file:
                caption_file.write(caption)
        else:
            print(f"Image not found: {image_path}")

print(f"Copied {sum(len(captions) for captions in landscape_captions.values())} images and captions to new folders.")

# Visualization

# Histogram - Distribution of keywords per category
category_counts = Counter([category for word, category in landscape_keywords_extracted])
plt.figure(figsize=(10, 6))
plt.bar(category_counts.keys(), category_counts.values(), color='skyblue')
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Number of Keywords per Landscape Category')
plt.show()

# Wordcloud - Display top 500 TF-IDF keywords
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(top_tfidf_keywords))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top 500 TF-IDF Keywords')
plt.show()

# Wordcloud - Display top 500 chi-square keywords
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(top_chi2_keywords))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top 500 Chi-Square Keywords')
plt.show()

# Sample Display - Show some images and descriptions
for category, items in landscape_captions.items():
    print(f"\nCategory: {category}")
    for i, (image_id, caption) in enumerate(items[:3]):  # Show first three images and descriptions in each category
        image_path = os.path.join(new_image_folder, category, f"{image_id}.jpg")
        print(f"Image ID: {image_id}")
        print(f"Caption: {caption}")
        plt.figure()
        plt.imshow(plt.imread(image_path))
        plt.title(caption)
        plt.axis('off')
        plt.show()

# HMM text sequence generation example
generated_sequences = hmm_model.generate(length=50)
print("Generated HMM Text Sequences:", ''.join([chr(c) for c in generated_sequences]))
