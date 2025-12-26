import tkinter as tk
from tkinter import scrolledtext
import json
import random
import numpy as np
import re
import spacy
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load intents
with open("intents.json") as file:
    data = json.load(file)

# Load trained model
model = tf.keras.models.load_model("chatbot_model.h5")

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_punct])

# Prepare tokenizer
patterns = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(preprocess(pattern))
        labels.append(intent["tag"])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns)

label_index = list(set(labels))

# Chatbot response
def chatbot_response(msg):
    msg = preprocess(msg)
    seq = tokenizer.texts_to_sequences([msg])
    pad = pad_sequences(seq, maxlen=model.input_shape[1])
    pred = model.predict(pad, verbose=0)
    tag = label_index[np.argmax(pred)]

    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

# GUI functions
def send_message():
    user_msg = user_entry.get()
    if user_msg.strip() == "":
        return

    chat_box.insert(tk.END, "You: " + user_msg + "\n")
    response = chatbot_response(user_msg)
    chat_box.insert(tk.END, "Bot: " + response + "\n\n")
    user_entry.delete(0, tk.END)

# GUI setup
root = tk.Tk()
root.title("IT Helpdesk Chatbot")
root.geometry("500x500")

chat_box = scrolledtext.ScrolledText(root, wrap=tk.WORD)
chat_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

user_entry = tk.Entry(root)
user_entry.pack(padx=10, pady=5, fill=tk.X)

send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(pady=5)

root.mainloop()
