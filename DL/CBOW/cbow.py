import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

# =====================
# 1. Load Corpus from Text File
# =====================
with open("cbow.txt", "r", encoding="utf-8") as file:
    corpus = file.readlines()

# Remove any leading/trailing spaces or newlines
corpus = [line.strip() for line in corpus if line.strip()]

print("Number of sentences in corpus:", len(corpus))
print("Sample sentence:", corpus[0])

# =====================
# 2. Tokenization
# =====================
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
word2id = tokenizer.word_index
id2word = {v: k for k, v in word2id.items()}
vocab_size = len(word2id) + 1

print("\nVocabulary:", word2id)

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(corpus)
print(sequences)

# =====================
# 3. Generate CBOW Data
# =====================
window_size = 2
X, y = [], []

for seq in sequences:
    for idx, target_word in enumerate(seq):
        context_words = []
        for offset in range(-window_size, window_size + 1):
            if offset == 0:
                continue
            pos = idx + offset
            if 0 <= pos < len(seq):
                context_words.append(seq[pos])
        if context_words:
            X.append(context_words)
            y.append(target_word)

# For simplicity, take the mean index of context words
X = np.array([np.mean(context, dtype=int) for context in X]).reshape(-1, 1)
y = to_categorical(y, vocab_size)

# =====================
# 4. CBOW Model
# =====================
embedding_dim = 50
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=1),
    Lambda(lambda x: tf.reduce_mean(x, axis=1)),
    Dense(vocab_size, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# =====================
# 5. Train Model
# =====================
model.fit(X, y, epochs=200, verbose=0)

# =====================
# 6. Extract Embeddings
# =====================
weights = model.get_weights()[0]
print("\nEmbedding matrix shape:", weights.shape)

# =====================
# 7. Find Most Similar Words
# =====================
def most_similar(word, top_n=3):
    if word not in word2id:
        return []
    idx = word2id[word]
    vec = weights[idx].reshape(1, -1)
    sims = cosine_similarity(vec, weights)[0]
    similar_ids = sims.argsort()[-top_n-1:][::-1]
    return [id2word[i] for i in similar_ids if i in id2word and i != idx]

print("\nMost similar to 'learning':", most_similar("learning"))
print("Most similar to 'model':", most_similar("model"))
