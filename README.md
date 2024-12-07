
# **LLM Classification**

This project implements a LLM-based classification framework that supports two modes of operation:
1. **Custom Prediction:** Classify user-provided text using custom labels and descriptions.
2. **Offline Evaluation:** Evaluate the model on pre-configured datasets and generate predictions and metrics.

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone <repository_url>
cd <repository_directory>
```

### **2. Set Up a Virtual Environment**
```bash
# Create a virtual environment
conda create -n llmclass python=3.8

# Activate the virtual environment
conda activate llmclass
```

### **3. Install Requirements**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## **Project Overview**

### **Framework Components**
1. **Datasets:** Supports datasets such as IMDB, DBpedia, and TweetEval.
   - These are preprocessed dynamically during offline evaluation.
   - Each dataset includes labels and descriptions defined in `config.yaml`.

2. **Modes:**
   - **Offline Evaluation:**
     - Evaluates the model on pre-configured datasets.
     - Generates predictions, computes metrics (accuracy, precision, recall, F1-score), and saves them to files.
   - **Custom Prediction:**
     - Allows classification of user-provided input text.
     - Uses labels and descriptions defined in `config.yaml`.

3. **Configuration File:**
   - `config.yaml` contains all settings, including datasets, labels, number of few-shot examples, and mode of operation.

4. **Key Files:**
   - `data_loader.py`: Handles dataset loading and preprocessing.
   - `prompt_engineer.py`: Constructs prompts for the LLM.
   - `model_chain.py`: Initializes the language model and constructs classification chains.
   - `evaluation.py`: Computes evaluation metrics.
   - `main.py`: Orchestrates the overall workflow based on `config.yaml`.

---

## **Usage**

### **1. Run in `custom_prediction` Mode**

In this mode, you can classify custom text inputs using predefined labels and descriptions.

1. Edit `config.yaml`:
   ```yaml
   mode: "custom_prediction"
   ```
   Example custom labels:
   ```yaml
   custom_labels:
     Binary:
       Positive: "Positive sentiment."
       Negative: "Negative sentiment."
     Multiclass:
       Critical: "Critical issue that needs immediate attention."
       High: "High priority issue."
       Medium: "Medium priority issue."
       Low: "Low priority issue."
   ```

2. Run the script:
   ```bash
   python main.py
   ```

3. Example Interaction:
   ```bash
   Enter task type (Binary or Multiclass): Binary
   Enter text to classify (or type 'exit' to quit): The movie was amazing!
   Prediction: Positive

   Enter text to classify (or type 'exit' to quit): The product was a waste of money.
   Prediction: Negative

   Enter text to classify (or type 'exit' to quit): exit
   ```

---

### **2. Run in `offline_evaluation` Mode**

In this mode, the framework evaluates the model on pre-configured datasets and saves predictions and metrics.

1. Edit `config.yaml`:
   ```yaml
   mode: "offline_evaluation"
   ```

2. Configure datasets:
   - Pre-configured datasets include:
     - IMDB (binary sentiment classification)
     - DBpedia (multiclass classification of Wikipedia topics)
     - TweetEval (multiclass sentiment classification of tweets)

   Example dataset configuration:
   ```yaml
   datasets:
     imdb:
       name: "imdb"
       label_column: "sentiment"
       text_column: "review"
       task_type: "Binary"
       test_limit: 100
       labels:
         Positive: "Positive sentiment."
         Negative: "Negative sentiment."
     dbpedia:
       name: "dbpedia_14"
       label_column: "label"
       text_column: "content"
       task_type: "Multiclass"
       test_limit: 50
       labels:
         Company: "A company or business organization."
         Educational Institution: "An institution focused on education."
         ...
   ```

3. Run the script:
   ```bash
   python main.py
   ```

4. Output Files:
   - Predictions:
     - `results/imdb_predictions.json`
     - `results/dbpedia_predictions.json`
     - `results/tweet_eval_predictions.json`
   - Metrics:
     - `results/imdb_metrics.txt`
     - `results/dbpedia_metrics.txt`
     - `results/tweet_eval_metrics.txt`

---

## **Example Outputs**

### **Predictions (JSON)**
**File:** `results/imdb_predictions.json`
```json
[
    {
        "key": 0,
        "input": "This movie was fantastic!",
        "prediction": "Positive",
        "label": "Positive"
    },
    {
        "key": 1,
        "input": "The plot was dull and uninspired.",
        "prediction": "Negative",
        "label": "Negative"
    }
]
```

### **Metrics (TXT)**
**File:** `results/imdb_metrics.txt`
```
accuracy: 0.8700
precision: 0.8750
recall: 0.8650
f1: 0.8700
```

---

## **Project Dependencies**

### **Python Packages**
- `langchain`: Framework for constructing LLM-powered workflows.
- `datasets`: Handles dataset loading and preprocessing.
- `scikit-learn`: Computes evaluation metrics.
- `pandas`: Manages data preprocessing and tabular operations.
- `regex`: Processes and matches model predictions.

### **Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## **Project Extension**
This framework is designed for extensibility:
- Add new datasets by extending `data_loader.py` and updating `config.yaml`.
- Add new task types or labels dynamically in `custom_prediction` mode.

Feel free to contribute enhancements or report issues.

---

## **License**
This project is licensed under the MIT License.

---

Let me know if you'd like any additional sections, such as example datasets or contributing guidelines!