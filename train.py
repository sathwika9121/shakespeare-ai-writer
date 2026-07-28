import os
import pickle
import urllib.request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

DATA_FILE = "tiny-shakespeare.txt"
MODEL_PATH = "shakespeare_rnn.keras"
VOCAB_PATH = "vocabulary.pkl"
WINDOW_SIZE = 50
EPOCHS = 5

# 1. Download dataset automatically if missing
if not os.path.exists(DATA_FILE):
    print("Downloading tiny-shakespeare.txt dataset...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, DATA_FILE)
    print("Dataset downloaded successfully!")

# 2. Read dataset
print("Reading text data...")
with open(DATA_FILE, "r", encoding="utf-8") as file:
    text = file.read()[:100000]

characters = sorted(list(set(text)))
char_to_num = {char: idx for idx, char in enumerate(characters)}
num_to_char = {idx: char for idx, char in enumerate(characters)}

print("Saving vocabulary.pkl...")
with open(VOCAB_PATH, "wb") as file:
    pickle.dump((char_to_num, num_to_char), file)

# 3. Prepare sequence data
print("Preparing training dataset sequences...")
inputs, outputs = [], []
for i in range(len(text) - WINDOW_SIZE):
    inputs.append([char_to_num[c] for c in text[i : i + WINDOW_SIZE]])
    outputs.append(char_to_num[text[i + WINDOW_SIZE]])

X = np.array(inputs)
y = tf.keras.utils.to_categorical(outputs, num_classes=len(characters))

# 4. Build model
print("Building neural network model...")
model = Sequential([
    Embedding(len(characters), 64, input_shape=(WINDOW_SIZE,)),
    SimpleRNN(128, return_sequences=False),
    Dense(128, activation="relu"),
    Dense(len(characters), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 5. Train and save
print("Training model (this will take 1–2 minutes)...")
model.fit(X, y, epochs=EPOCHS, batch_size=256)

print("Saving shakespeare_rnn.keras...")
model.save(MODEL_PATH)

print("\nSUCCESS: Both 'shakespeare_rnn.keras' and 'vocabulary.pkl' are created!")