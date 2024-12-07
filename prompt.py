from langchain.prompts import ChatPromptTemplate

def create_prompt_with_custom_labels(task_type: str, labels: dict, few_shot_examples: list = []):
    """
    Creates a classification prompt with user-defined labels and descriptions.

    Args:
        task_type (str): Classification task type (e.g., Binary or Multiclass).
        labels (dict): Labels and their descriptions as key-value pairs.
        few_shot_examples (list): Few-shot examples in the form of a list of dicts.

    Returns:
        ChatPromptTemplate: A dynamic prompt for the classification task.
    """
    label_description = "\n".join([f"{label}: {desc}" for label, desc in labels.items()])
    few_shots = "\n".join([f"Input: {example['input']}\nLabel: {example['label']}" for example in few_shot_examples])
    
    template = f"""
    Task Type: {task_type}
    Labels:
    {label_description}
    Few-shot Examples:
    {few_shots}
    
    Input: {{input}}
    Predict the correct label for the input based on the labels above.
    """
    return ChatPromptTemplate.from_template(template)
