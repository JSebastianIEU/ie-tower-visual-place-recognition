def top_k_accuracy(
    retrieved_labels: list[list[str]],
    ground_truth_labels: list[str],
    k: int,
) -> float:
    if len(retrieved_labels) != len(ground_truth_labels):
        raise ValueError("retrieved_labels and ground_truth_labels must have the same length")

    if not ground_truth_labels:
        return 0.0

    hits = sum(
        ground_truth in predicted_labels[:k]
        for predicted_labels, ground_truth in zip(retrieved_labels, ground_truth_labels)
    )
    return hits / len(ground_truth_labels)


def average_precision(relevance: list[int]) -> float:
    num_relevant = 0
    precision_sum = 0.0

    for rank, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            num_relevant += 1
            precision_sum += num_relevant / rank

    if num_relevant == 0:
        return 0.0

    return precision_sum / num_relevant


def mean_average_precision(relevance_lists: list[list[int]]) -> float:
    if not relevance_lists:
        return 0.0

    return sum(average_precision(relevance) for relevance in relevance_lists) / len(
        relevance_lists
    )
