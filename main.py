import numpy as np
import tensorflow as tf
import streamlit as st
import re
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# PAGE CONFIG 

st.set_page_config(page_title="IMDB Sentiment AI", page_icon="🎬")


st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#020617,#0f172a);
color:white;
}

.title{
font-size:48px;
font-weight:700;
text-align:center;
}

.subtitle{
text-align:center;
color:#cbd5f5;
margin-bottom:40px;
}

.result-box{
padding:25px;
border-radius:14px;
font-size:24px;
font-weight:600;
text-align:center;
margin-top:15px;
}

.positive{
background:#065f46;
}

.negative{
background:#7f1d1d;
}

</style>
""", unsafe_allow_html=True)

#  MODEL SETTINGS

max_features = 10000
max_len = 500

# Load IMDB word index
word_index = imdb.get_word_index()

#  LOAD MODEL (CACHED) 

@st.cache_resource
def load_my_model():
    return load_model("simple_rnn_imdb.h5", compile=False)

model = load_my_model()

# PREPROCESS FUNCTION 

def preprocess_text(text):

    # Clean text
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text.lower())

    words = text.split()

    encoded_review = []

    for word in words:

        index = word_index.get(word)

        if index is not None:
            index = index + 3

            # prevent embedding crash
            if index < max_features:
                encoded_review.append(index)
            else:
                encoded_review.append(2)
        else:
            encoded_review.append(2)

    # prevent empty input
    if len(encoded_review) == 0:
        encoded_review = [2]

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=max_len
    )

    return padded_review


# TITLE 

st.markdown("<div class='title'>🎬 IMDB Movie Review Sentiment AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Enter a movie review and AI will classify it.</div>", unsafe_allow_html=True)

# SESSION STATE 

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

# EXAMPLE BUTTONS 
st.subheader("Try Example Reviews")

col1, col2, col3 = st.columns(3)

if col1.button("😊 Positive Example"):
    st.session_state.review_text = "This movie was amazing and I loved every moment"

if col2.button("😞 Negative Example"):
    st.session_state.review_text = "This was the worst movie"

if col3.button("🤔 Mixed Example"):
    st.session_state.review_text = "The movie started great but the ending was terrible"

#  TEXT AREA

st.session_state.review_text = st.text_area(
    "Movie Review",
    value=st.session_state.review_text,
    height=150
)

#  PREDICTION

if st.button("🚀 Classify Sentiment"):

    if st.session_state.review_text.strip() == "":
        st.warning("Please enter a movie review.")

    else:

        with st.spinner("Analyzing review..."):

            processed_input = preprocess_text(st.session_state.review_text)

            prediction = model.predict(processed_input, verbose=0)

            score = float(prediction[0][0])

        sentiment = "Positive 😊" if score > 0.5 else "Negative 😞"

        # RESULT 

        st.subheader("Result")

        if score > 0.5:
            st.markdown(
                f"<div class='result-box positive'>{sentiment}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-box negative'>{sentiment}</div>",
                unsafe_allow_html=True
            )

        st.write(f"Prediction Score: **{score:.4f}**")

        #  CONFIDENCE 

        positive_prob = score
        negative_prob = 1 - score

        st.subheader("Confidence")

        st.write(f"Positive: {positive_prob:.2f}")
        st.progress(float(positive_prob))

        st.write(f"Negative: {negative_prob:.2f}")
        st.progress(float(negative_prob))
