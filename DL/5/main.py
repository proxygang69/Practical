import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

# =====================
# 1. Sample Corpus
# =====================
corpus = [
    "Deep learning is a subfield of machine learning",
    "Word embeddings capture semantic meaning of words",
    "CBOW model predicts target word using context words"
]

# =====================
# 2. Tokenization
# =====================
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
word2id = tokenizer.word_index
id2word = {v: k for k, v in word2id.items()}
vocab_size = len(word2id) + 1

print("Vocabulary:", word2id)

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(corpus)

# =====================
# 3. Generate CBOW Data
# =====================
window_size = 2
X, y = [], []

for seq in sequences:
    for idx, target_word in enumerate(seq):
        context_words = []
        for offset in range(-window_size, window_size + 1):
            if offset == 0:  # skip target word
                continue
            pos = idx + offset
            if 0 <= pos < len(seq):
                context_words.append(seq[pos])
        if context_words:
            X.append(context_words)
            y.append(target_word)

# Pad or take average of context words (simple averaging embeddings)
X = np.array([np.mean(context, dtype=int) for context in X]).reshape(-1, 1)
y = to_categorical(y, vocab_size)

# =====================
# 4. CBOW Model
# =====================
embedding_dim = 50
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=1),
    Lambda(lambda x: tf.reduce_mean(x, axis=1)),  # average embeddings if multiple context
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
print("Embedding matrix shape:", weights.shape)

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

print("Most similar to 'learning':", most_similar("learning"))
print("Most similar to 'model':", most_similar("model"))
