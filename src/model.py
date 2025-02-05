import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

def create_model(vocab_size, embedding_dim=128, lstm_units=64):
    """
    Create an LSTM model for sentiment analysis.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        LSTM(lstm_units, return_sequences=True),
        Dropout(0.3),
        LSTM(lstm_units),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
