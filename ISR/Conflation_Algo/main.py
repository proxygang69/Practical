# main.py
import re
import os
from collections import Counter
from nltk.stem import PorterStemmer

# Function to load stop words from file
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        stopwords = set(word.strip().lower() for word in f)
    return stopwords

# Function to preprocess document: tokenize, remove stopwords, apply stemming
def preprocess_document(doc_path, stopwords):
    with open(doc_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()

    # Tokenize (extract only words)
    words = re.findall(r'\b[a-z]+\b', text)

    # Remove stop words
    filtered_words = [w for w in words if w not in stopwords]

    # Apply stemming (Conflation step)
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(w) for w in filtered_words]

    return stemmed_words

# Function to generate frequency index
def generate_frequency_index(words):
    return Counter(words)

# Function to save output to file
def save_output(counter, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Word\tFrequency\n")
        f.write("=====================\n")
        for word, freq in counter.most_common():
            f.write(f"{word}\t{freq}\n")
    print(f"\n✅ Output saved to {output_path}")

# ---- MAIN EXECUTION ----
if __name__ == "__main__":
    stopwords_file = "stopwords.txt"  # your file name
    document_file = "doc.txt"        # your file name
    output_file = "output.txt"

    # Check if files exist
    if not os.path.exists(stopwords_file):
        print(f"❌ Error: Stopword file not found at {stopwords_file}")
        exit()

    if not os.path.exists(document_file):
        print(f"❌ Error: Document file not found at {document_file}")
        exit()

    # Load stopwords and process document
    stopwords = load_stopwords(stopwords_file)
    stemmed_words = preprocess_document(document_file, stopwords)
    freq_index = generate_frequency_index(stemmed_words)

    # Save and display results
    save_output(freq_index, output_file)

    print("\nTop frequent conflated (stemmed) words:")
    for word, freq in freq_index.most_common(10):
        print(f"{word}: {freq}")
