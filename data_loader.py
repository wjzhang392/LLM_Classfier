import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset

# Global label mappings for datasets
LABEL_MAPPINGS = {
    "dbpedia_14": {
        "Company": 0,
        "Educational Institution": 1,
        "Artist": 2,
        "Athlete": 3,
        "Office Holder": 4,
        "Mean of Transportation": 5,
        "Building": 6,
        "Natural Place": 7,
        "Village": 8,
        "Animal": 9,
        "Plant": 10,
        "Album": 11,
        "Film": 12,
        "Written Work": 13
    },
    "tweet_eval": {
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2
    },
    "imdb": {  # Binary labels
        "Negative": 0,
        "Positive": 1
    }
}

def load_imdb():
    """Loads IMDB dataset for binary sentiment classification."""
    dataset = load_dataset("imdb")
    train = dataset["train"].to_pandas()
    test = dataset["test"].to_pandas()

    # Rename columns for consistency
    train = train.rename(columns={"text": "review", "label": "sentiment"})
    test = test.rename(columns={"text": "review", "label": "sentiment"})

    # Map sentiment labels to numeric values
    label_mapping_reverse = {v: k for k, v in LABEL_MAPPINGS["imdb"].items()}
    train["sentiment"] = train["sentiment"].map(label_mapping_reverse)
    test["sentiment"] = test["sentiment"].map(label_mapping_reverse)
    return train, test

def load_dbpedia():
    """Loads DBpedia dataset for multiclass classification."""
    dataset = load_dataset("dbpedia_14")
    train = dataset["train"].to_pandas()
    test = dataset["test"].to_pandas()
    label_mapping_reverse = {v: k for k, v in LABEL_MAPPINGS["dbpedia_14"].items()}
    train["label"] = train["label"].map(label_mapping_reverse)
    test["label"] = test["label"].map(label_mapping_reverse)
    return train, test

def load_tweet_eval():
    """Loads TweetEval dataset for sentiment classification."""
    dataset = load_dataset("tweet_eval", "sentiment")
    train = dataset["train"].to_pandas()
    test = dataset["test"].to_pandas()
    label_mapping_reverse = {v: k for k, v in LABEL_MAPPINGS["tweet_eval"].items()}
    train["label"] = train["label"].map(label_mapping_reverse)
    test["label"] = test["label"].map(label_mapping_reverse)
    return train, test
