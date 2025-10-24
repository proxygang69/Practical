# Precision and Recall Calculation in Python

def read_input(filename):
    retrieved = set()
    relevant = set()

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("A:"):
                retrieved = set(line.replace("A:", "").strip().split())
            elif line.startswith("Rq1:"):
                relevant = set(line.replace("Rq1:", "").strip().split())

    return retrieved, relevant


def calculate_precision_recall(retrieved, relevant):
    # Intersection: relevant and retrieved
    true_positive = retrieved.intersection(relevant)

    # Precision and Recall formulas
    precision = len(true_positive) / len(retrieved) if retrieved else 0
    recall = len(true_positive) / len(relevant) if relevant else 0

    return precision, recall, true_positive


def main():
    filename = "data.txt"
    retrieved, relevant = read_input(filename)

    precision, recall, true_positive = calculate_precision_recall(retrieved, relevant)

    print("\nRetrieved Documents (A):", retrieved)
    print("Relevant Documents (Rq1):", relevant)
    print("Relevant & Retrieved (Intersection):", true_positive)

    print(f"\nPrecision = {len(true_positive)}/{len(retrieved)} = {precision:.2f}")
    print(f"Recall    = {len(true_positive)}/{len(relevant)} = {recall:.2f}")


if __name__ == "__main__":
    main()
