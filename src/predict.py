import tensorflow as tf
import numpy as np
import argparse
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Load the trained model
model = tf.keras.models.load_model("models/sentiment_model.keras")

# Load IMDB word index
word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3
reverse_word_index = {v: k for k, v in word_index.items()}

def encode_text(text):
    """
    Encode input text into numerical sequences using IMDB word index.
    """
    words = text.lower().split()
    encoded = [word_index.get(w, 2) for w in words]  # 2 is <UNK> for unknown words
    return pad_sequences([encoded], maxlen=200)

def predict_sentiment(text):
    """
    Predict sentiment of a given text input.
    """
    processed_text = encode_text(text)
    prediction = model.predict(processed_text)[0][0]
    sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"
    print(f"Sentiment: {sentiment} ({prediction:.2f})")

def main():
    parser = argparse.ArgumentParser(description="Predict sentiment of a given text")
    parser.add_argument("--text", type=str, required=True, help="Input text for sentiment analysis")
    args = parser.parse_args()
    
    predict_sentiment(args.text)

if __name__ == "__main__":
    main()
