# F-measure (Harmonic Mean) and E-measure Calculation

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
    true_positive = retrieved.intersection(relevant)
    precision = len(true_positive) / len(retrieved) if retrieved else 0
    recall = len(true_positive) / len(relevant) if relevant else 0
    return precision, recall


def calculate_f_and_e_measure(precision, recall, beta=1):
    # Avoid division by zero
    if (precision + recall) == 0:
        f_measure = 0
    else:
        f_measure = (2 * precision * recall) / (precision + recall)

    # E-measure (for beta=1, E = 1 - F)
    e_measure = 1 - f_measure
    return f_measure, e_measure


def main():
    filename = "data.txt"
    retrieved, relevant = read_input(filename)

    precision, recall = calculate_precision_recall(retrieved, relevant)
    f_measure, e_measure = calculate_f_and_e_measure(precision, recall)

    print("\nRetrieved Documents (A):", retrieved)
    print("Relevant Documents (Rq1):", relevant)
    print(f"\nPrecision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F-Measure (F1 Score): {f_measure:.2f}")
    print(f"E-Measure: {e_measure:.2f}")


if __name__ == "__main__":
    main()
