# Inverted Index Implementation in Python

# Step 1: Build the inverted index
def build_inverted_index(filename):
    inverted_index = {}
    with open(filename, 'r') as file:
        text = file.read().lower()  # convert to lowercase

    # Split text into words (basic tokenization)
    words = text.replace('.', '').replace(',', '').split()

    # Build index with word positions
    for position, word in enumerate(words, start=1):
        if word not in inverted_index:
            inverted_index[word] = []
        inverted_index[word].append(position)
    
    return inverted_index


# Step 2: Search query word(s)
def search_query(inverted_index, query):
    query = query.lower()
    if query in inverted_index:
        return inverted_index[query]
    else:
        return []


# Step 3: Main program
def main():
    filename = r"E:\Practical\ISR\InvertedFile\input.txt"

    inverted_index = build_inverted_index(filename)

    print("\n--- Inverted Index ---")
    for word, positions in inverted_index.items():
        print(f"{word:10s} -> {positions}")

    # Query input
    query = input("\nEnter a word to search: ")
    result = search_query(inverted_index, query)

    if result:
        print(f"\nWord '{query}' found at positions: {result}")
    else:
        print(f"\nWord '{query}' not found in the text.")


if __name__ == "__main__":
    main()
