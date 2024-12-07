import yaml
import os
import json
from data_loader import *
from prompt import create_prompt_with_custom_labels
from model import initialize_llm, create_chain
from evaluation import evaluate_model

# Load configuration from YAML
def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

# Load dataset by name
def load_dataset_by_name(name):
    if name == "imdb":
        return load_imdb()
    elif name == "dbpedia_14":
        return load_dbpedia()
    elif name == "tweet_eval":
        return load_tweet_eval()
    else:
        raise ValueError(f"Dataset '{name}' not supported.")

# Generate few-shot examples dynamically
def generate_few_shot_examples(train_data, n_shot, label_column, text_column):
    examples = []
    sampled_data = train_data.sample(n=n_shot, random_state=42)
    for _, row in sampled_data.iterrows():
        examples.append({"input": row[text_column], "label": row[label_column]})
    return examples

# Save predictions to a JSON file
def save_predictions_to_json(predictions, dataset_name, results_dir):
    json_path = os.path.join(results_dir, f"{dataset_name}_predictions.json")
    with open(json_path, "w") as json_file:
        json.dump(predictions, json_file, indent=4)
    print(f"Predictions saved to {json_path}")

# Save metrics to a TXT file
def save_metrics_to_txt(metrics, dataset_name, results_dir):
    txt_path = os.path.join(results_dir, f"{dataset_name}_metrics.txt")
    with open(txt_path, "w") as txt_file:
        for key, value in metrics.items():
            txt_file.write(f"{key}: {value:.4f}\n")
    print(f"Metrics saved to {txt_path}")

def perform_offline_evaluation(config, llm):
    # Create results directory if not exists
    results_dir = config["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # Load datasets
    datasets_config = config["datasets"]
    for _, dataset_details in datasets_config.items():
        dataset_name = dataset_details["name"]
        print(f"Evaluating dataset: {dataset_name}...")

        # Load dataset
        train, test = load_dataset_by_name(dataset_name)

        # Limit test examples
        test_limit = dataset_details["test_limit"]
        test = test.head(test_limit)

        # Generate few-shot examples
        n_shot = config["n_shot"]
        few_shot_examples = generate_few_shot_examples(
            train_data=train,
            n_shot=n_shot,
            label_column=dataset_details["label_column"],
            text_column=dataset_details["text_column"]
        )

        # Dynamically retrieve labels and descriptions
        labels = dataset_details.get("labels")
        if not labels:
            raise ValueError(f"No labels found for dataset '{dataset_name}'.")

        # Create prompt
        prompt = create_prompt_with_custom_labels(
            task_type=dataset_details["task_type"],
            labels=labels,
            few_shot_examples=few_shot_examples
        )

        # Create chain
        chain = create_chain(llm, prompt)

        # Run predictions
        predictions = []
        for idx, row in test.iterrows():
            input_text = row[dataset_details["text_column"]]
            true_label = row[dataset_details["label_column"]]
            prediction = chain.run({"input": input_text}).strip()
            predictions.append({
                "key": idx,
                "input": input_text,
                "prediction": prediction,
                "label": true_label
            })

        # Save predictions to JSON
        save_predictions_to_json(predictions, dataset_name, results_dir)

        # Evaluate metrics
        true_labels = [LABEL_MAPPINGS[dataset_name][item["label"]] for item in predictions]
        raw_predictions = [item["prediction"] for item in predictions]
        metrics = evaluate_model(raw_predictions, true_labels, dataset_name)

        # Save metrics to TXT
        save_metrics_to_txt(metrics, dataset_name, results_dir)

def perform_custom_prediction(config, llm):
    # User-defined labels and task type
    task_type = input("Enter task type (Binary or Multiclass): ").strip()
    labels = config["custom_labels"].get(task_type)
    if not labels:
        raise ValueError(f"No custom labels found for task type '{task_type}'.")

    # Create prompt
    prompt = create_prompt_with_custom_labels(
        task_type=task_type,
        labels=labels,
        few_shot_examples=[]
    )

    # Create chain
    chain = create_chain(llm, prompt)

    # Perform prediction
    while True:
        input_text = input("\nEnter text to classify (or type 'exit' to quit): ").strip()
        if input_text.lower() == "exit":
            break
        prediction = chain.run({"input": input_text}).strip()
        print(f"Prediction: {prediction}")

def main():
    # Load configuration
    config = load_config("config.yaml")

    # Initialize LLM
    llm = initialize_llm(model_name=config["model"]["name"], temperature=config["model"]["temperature"])

    # Determine mode
    mode = config.get("mode")
    if mode == "offline_evaluation":
        perform_offline_evaluation(config, llm)
    elif mode == "custom_prediction":
        perform_custom_prediction(config, llm)
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose between 'offline_evaluation' and 'custom_prediction'.")

if __name__ == "__main__":
    main()
