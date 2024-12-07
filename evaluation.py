from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import re
from data_loader import LABEL_MAPPINGS

def parse_prediction_to_number(prediction, dataset_name):
    """
    Parse model predictions to numeric labels using regex and global label mappings.

    Args:
        prediction (str): The model's output text.
        dataset_name (str): The dataset being used, e.g., "dbpedia_14", "tweet_eval", "imdb".

    Returns:
        int: The numeric label corresponding to the parsed prediction.
    """
    label_mapping = LABEL_MAPPINGS.get(dataset_name)
    if not label_mapping:
        raise ValueError(f"Label mapping not found for dataset '{dataset_name}'.")

    # Try to match the prediction with the label keys
    for label, number in label_mapping.items():
        if re.search(rf"\b{re.escape(label)}\b", prediction, re.IGNORECASE):
            return number

    # If no match, return -1 (indicates invalid prediction)
    return -1

def evaluate_model(predictions, true_labels, dataset_name):
    """
    Evaluate model performance using parsed predictions.

    Args:
        predictions (list): List of raw predictions from the model.
        true_labels (list): List of ground truth labels.
        dataset_name (str): The dataset being used, e.g., "dbpedia_14", "tweet_eval", "imdb".

    Returns:
        dict: A dictionary with evaluation metrics (accuracy, precision, recall, F1).
    """
    # Parse predictions to numeric labels
    parsed_predictions = [
        parse_prediction_to_number(prediction, dataset_name) for prediction in predictions
    ]

    # Filter out invalid predictions (-1)
    valid_indices = [i for i, p in enumerate(parsed_predictions) if p != -1]
    valid_predictions = [parsed_predictions[i] for i in valid_indices]
    valid_true_labels = [true_labels[i] for i in valid_indices]

    # Compute metrics
    accuracy = accuracy_score(valid_true_labels, valid_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_true_labels, valid_predictions, average="weighted"
    )
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def visualize_metrics(results):
    """Visualizes metrics for each dataset."""
    for dataset_name, metrics in results.items():
        labels = list(metrics.keys())
        values = list(metrics.values())

        plt.figure(figsize=(8, 5))
        plt.bar(labels, values)
        plt.title(f"Performance Metrics for {dataset_name}")
        plt.ylabel("Score")
        plt.xlabel("Metric")
        plt.show()
