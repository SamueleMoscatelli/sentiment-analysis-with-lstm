# Sentiment Analysis with LSTMs

This project implements a **sentiment analysis model** using **LSTMs (Long Short-Term Memory Networks)**. The model is trained on the **IMDB movie review dataset** and predicts whether a given review has a **positive or negative sentiment**.

## 🚀 Features
✅ **Trains an LSTM model for sentiment analysis**  
✅ **Uses TensorFlow/Keras for deep learning**  
✅ **Loads preprocessed IMDB reviews dataset**  
✅ **Accepts custom user input for sentiment prediction**  

## 📂 Project Structure
sentiment-analysis-with-lstm/ <br/>
│── notebooks/ # Jupyter notebooks for experiments <br/>
│── src/ # Source code for training and evaluation <br/>
│   │── train.py # Main script to train the model <br/>
│   │── predict.py # Script to make sentiment predictions <br/>
│   │── model.py # LSTM model definition <br/>
│── models/ # Saved trained models <br/>
│   │── sentiment_model.keras <br/>
│── requirements.txt # Dependencies <br/>
│── README.md # Project description <br/>
│── .gitignore # Ignore unnecessary files<br/>


## 🛠 Setup Instructions
### 1️⃣ Clone the repository  
```bash
git clone https://github.com/SamueleMoscatelli/sentiment-analysis-with-lstm.git
cd sentiment-analysis-lstm
```

### 2️⃣ Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

## 📊 Train the Model
To train the model on the IMDB dataset:
```bash
python src/train.py
```

This will train the LSTM model and save it to models/sentiment_model.keras.

## 📝 Predict Sentiment
To predict sentiment of a custom text input:
```bash
python src/predict.py --text "This movie was absolutely amazing!"
```

🔹 Output:
```bash
Sentiment: Positive (0.91)
```