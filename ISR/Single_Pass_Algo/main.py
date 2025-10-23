import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_documents(num_docs=5, folder="."):
    documents = []
    for i in range(1, num_docs + 1):
        filename = os.path.join(folder, f"doc{i}.txt")
        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}")
            continue
        with open(filename, 'r', encoding='utf-8') as f:
            documents.append(f.read())
    return documents

def single_pass_clustering(docs, threshold=0.15):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(docs)

    clusters = []

    for doc_idx in range(len(docs)):
        doc_vector = tfidf_matrix[doc_idx]

        if not clusters:
            clusters.append([doc_idx])
            continue

        cluster_similarities = []
        for cluster in clusters:
            cluster_vectors = tfidf_matrix[cluster]
            centroid = cluster_vectors.mean(axis=0).A1  # Convert matrix to 1D array
            similarity = cosine_similarity(doc_vector, centroid.reshape(1, -1))[0][0]
            cluster_similarities.append(similarity)

        max_sim = max(cluster_similarities)
        max_index = cluster_similarities.index(max_sim)

        if max_sim >= threshold:
            clusters[max_index].append(doc_idx)
        else:
            clusters.append([doc_idx])

    return clusters

if __name__ == "__main__":
    folder = "."  # Assuming your docs are in the same folder as the script
    documents = load_documents(num_docs=5, folder=folder)

    if len(documents) < 1:
        print("No documents loaded. Please check file paths and filenames.")
        exit()

    clusters = single_pass_clustering(documents, threshold=0.15)  # lowered threshold

    print("\n--- Clustering Results ---")
    for i, cluster in enumerate(clusters):
        print(f"\nCluster {i + 1}:")
        for doc_idx in cluster:
            print(f"  Doc {doc_idx + 1}: {documents[doc_idx]}")
