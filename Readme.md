## RNN Movie Sentiment Analysis

This project uses a **Simple RNN model** to classify movie reviews as **Positive** or **Negative** using the **IMDB dataset**.
A small **Streamlit app** is included so you can enter your own movie reviews and see the model’s prediction.

## Tech Stack

* Python
* TensorFlow / Keras
* Streamlit
* IMDB Dataset


## Running the App

1. Install the required packages:

```
pip install -r requirements.txt
```

2. Start the Streamlit app:

```
streamlit run main.py
```

3. Enter a movie review and the model will predict whether the sentiment is **positive or negative**.

## Model

The model is a **Simple RNN network** trained on the IMDB movie review dataset with a vocabulary size of **10,000 words**.
Reviews are padded to a length of **500 tokens** before being passed to the network.

## Note

This is a basic RNN implementation for learning purposes, so predictions may not always be perfect.
