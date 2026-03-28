import json
import numpy as np
import random
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load files
model = load_model("model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("intents.json") as file:
    data = json.load(file)

max_len = 20

def predict_class(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    
    prediction = model.predict(padded)
    tag = label_encoder.inverse_transform([np.argmax(prediction)])
    
    return tag[0]

def get_response(tag):
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

print("Chatbot is running! (type 'quit' to exit)")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "quit":
        break
    
    tag = predict_class(user_input)
    response = get_response(tag)
    
    print("Bot:", response)