# NLP Chatbot using LSTM

This project is a simple AI chatbot built using Natural Language Processing and an LSTM deep learning model. It can understand user input, classify the intent, and respond with predefined answers.

## Tech Stack

Python, TensorFlow/Keras, NLTK, Scikit-learn

## Features

* Text preprocessing (tokenization, padding)
* Intent classification using LSTM
* Dynamic responses based on user input
* Easily customizable dataset (intents.json)

## Project Structure

* train.py – trains the LSTM model
* chatbot.py – runs the chatbot
* intents.json – dataset with intents and responses
* model.h5 – trained model
* tokenizer.pkl – tokenizer file
* label_encoder.pkl – label encoder

## How to Run

Install dependencies using `pip install -r requirements.txt`
Run `python train.py` to train the model
Run `python chatbot.py` to start the chatbot

## Future Improvements

* Add web interface using Flask
* Improve dataset for better accuracy
* Integrate APIs for real-time responses

## Author

Mohammed Abzel
