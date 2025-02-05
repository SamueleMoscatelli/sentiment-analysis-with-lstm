import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.data_preprocessing import load_test_data

def evaluate_model(model_path, max_length=100):
    """
    Load the trained LSTM model and evaluate its performance on the test dataset.
    """
    # Load the trained model
    model = load_model(model_path)

    # Load the test dataset
    (X_test, y_test), tokenizer = load_test_data()

    # Pad sequences to match training input shape
    X_test = pad_sequences(X_test, maxlen=max_length, padding='post')

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    return loss, accuracy

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Evaluate the trained LSTM sentiment analysis model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model file (.keras format).")
    args = parser.parse_args()

    # Run evaluation
    evaluate_model(args.model_path)

if __name__ == "__main__":
    main()
