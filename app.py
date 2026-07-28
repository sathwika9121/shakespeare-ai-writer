# app.py
import os
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

MODEL_PATH = "shakespeare_rnn.keras"
VOCAB_PATH = "vocabulary.pkl"
WINDOW_SIZE = 50

st.set_page_config(
    page_title="Shakespeare AI Writer",
    page_icon="🖋️",
    layout="centered"
)

# Custom Styling
st.markdown("""
<style>
.stApp { background:#0b1220; color:white; }
.header { text-align:center; padding:10px; }
.header h1 { font-size:42px; color:#60a5fa; }
.output { background:#020617; padding:20px; border-radius:12px; border-left:5px solid #38bdf8; font-family:Georgia; font-size:18px; line-height:1.8; white-space: pre-wrap; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>🖋️ Shakespeare AI Writer</h1>
    <p>Character Level Text Generation using SimpleRNN</p>
</div>
""", unsafe_allow_html=True)

# Load cached resources
@st.cache_resource
def load_rnn_model():
    return load_model(MODEL_PATH)

@st.cache_data
def load_vocabulary():
    with open(VOCAB_PATH, "rb") as file:
        return pickle.load(file)

# Check if model exists
if not os.path.exists(MODEL_PATH) or not os.path.exists(VOCAB_PATH):
    st.error("Model files not found! Please run 'python train.py' in your terminal first.")
    st.stop()

model = load_rnn_model()
char_to_num, num_to_char = load_vocabulary()

# UI Inputs
seed_text = st.text_input("Enter starting text:", value="To")
text_length = st.slider("Generated Characters:", 100, 800, 300, 50)
temperature = st.slider("Creativity Level:", 0.2, 1.5, 0.7, 0.1)

if st.button("✨ Generate Poem"):
    with st.spinner("Writing..."):
        result = seed_text
        while len(result) < WINDOW_SIZE:
            result = " " + result

        for _ in range(text_length):
            sequence = [char_to_num.get(c, 0) for c in result[-WINDOW_SIZE:]]
            sequence = np.array(sequence).reshape(1, WINDOW_SIZE)
            
            preds = model.predict(sequence, verbose=0)[0]
            preds = np.log(preds + 1e-8) / temperature
            preds = np.exp(preds) / np.sum(np.exp(preds))
            
            next_idx = np.random.choice(len(preds), p=preds)
            result += num_to_char[next_idx]

    st.success("Generation Complete!")
    st.markdown(f'<div class="output">{result}</div>', unsafe_allow_html=True)