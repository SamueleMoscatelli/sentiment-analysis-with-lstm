import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import create_model

def train_model():
    """
    Load IMDB dataset, preprocess the data, and train an LSTM model.
    """
    vocab_size = 10000  # Top 10,000 words in vocabulary
    max_length = 200    # Max review length

    # Load IMDB dataset
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    x_train, x_test = pad_sequences(x_train, maxlen=max_length), pad_sequences(x_test, maxlen=max_length)

    # Create and train the model
    model = create_model(vocab_size)
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

    # Save the trained model
    model.save("models/sentiment_model.keras")
    print("✅ Model saved successfully!")

def main():
    train_model()

if __name__ == "__main__":
    main()
