import unittest
from unittest.mock import patch
import pandas as pd
from main import main
from io import StringIO

# Mock small dataset samples
MOCK_IMDB_DATA = StringIO("""review,sentiment
The product is amazing!,Positive
Terrible experience.,Negative
Not worth the money.,Negative
Excellent service!,Positive
Horrible quality.,Negative
Really loved it.,Positive
Awful,Negative
Perfect experience.,Positive
Waste of time.,Negative
Superb performance!,Positive
""")

MOCK_AG_NEWS_DATA = StringIO("""text,label
Stocks rally as market recovers.,Business
Local team wins championship.,Sports
Government announces new policies.,World
Breakthrough in AI research.,Science/Tech
Unemployment rate falls.,Business
Player breaks world record.,Sports
Country signs peace agreement.,World
Scientists discover new element.,Science/Tech
Central bank raises interest rates.,Business
Team prepares for final match.,Sports
""")

MOCK_YELP_DATA = StringIO("""text,sentiment
"Great food and excellent service!",Positive
"Terrible experience, the waiter was rude.",Negative
"The ambiance was amazing, will come again.",Positive
"The food was cold and flavorless.",Negative
"Perfect evening at a lovely restaurant.",Positive
"Waited for an hour to be served, very disappointed.",Negative
"Loved the dessert, it was heavenly.",Positive
"The worst dining experience of my life.",Negative
"Nice place but overpriced.",Negative
"Absolutely wonderful food and staff.",Positive
""")

class TestMain(unittest.TestCase):
    @patch("main.load_dataset_by_name")
    @patch("main.evaluate_model")
    @patch("main.visualize_metrics")
    def test_main_with_mock_data(self, mock_visualize, mock_evaluate, mock_load_dataset):
        # Prepare mock data
        imdb_df = pd.read_csv(MOCK_IMDB_DATA)
        ag_news_df = pd.read_csv(MOCK_AG_NEWS_DATA)
        yelp_df = pd.read_csv(MOCK_YELP_DATA)

        # Mock dataset loading function
        def mock_load(name):
            if name == "imdb":
                return imdb_df.iloc[:5], imdb_df.iloc[5:]  # Split into train and test
            elif name == "ag_news":
                return ag_news_df.iloc[:5], ag_news_df.iloc[5:]  # Split into train and test
            elif name == "yelp":
                return yelp_df.iloc[:5], yelp_df.iloc[5:]  # Split into train and test
            else:
                raise ValueError(f"Dataset '{name}' not supported.")

        mock_load_dataset.side_effect = mock_load

        # Mock evaluation and visualization functions
        mock_evaluate.return_value = {
            "accuracy": 0.8,
            "precision": 0.75,
            "recall": 0.8,
            "f1": 0.77
        }
        mock_visualize.return_value = None

        # Run the main function
        with patch("builtins.print") as mock_print:
            main()

        # Verify that the mocked methods were called
        self.assertTrue(mock_load_dataset.called, "Dataset loading was not called.")
        self.assertTrue(mock_evaluate.called, "Evaluation function was not called.")
        self.assertTrue(mock_visualize.called, "Visualization function was not called.")

        # Check printed output
        printed_output = "\n".join([str(arg[0]) for arg in mock_print.call_args_list])
        self.assertIn("Evaluating dataset: imdb...", printed_output, "IMDB evaluation not logged.")
        self.assertIn("Evaluating dataset: ag_news...", printed_output, "AG News evaluation not logged.")
        self.assertIn("Evaluating dataset: yelp...", printed_output, "Yelp evaluation not logged.")
        self.assertIn("Metrics for imdb", printed_output, "Metrics for IMDB not displayed.")
        self.assertIn("Metrics for ag_news", printed_output, "Metrics for AG News not displayed.")
        self.assertIn("Metrics for yelp", printed_output, "Metrics for Yelp not displayed.")

if __name__ == "__main__":
    unittest.main()