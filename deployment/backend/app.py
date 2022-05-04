from flask import Flask, jsonify, request
import json
import string
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

app = Flask(__name__)

# Define stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Define stopword
factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()

# Load json file
f = open('intent/intent.json', 'r')
intent_json = json.load(f)

# Define slang dictionary
slang = pd.read_csv('lexicon/slang ke semi baku.csv')
slang_replace = {}
for i, row in enumerate(slang['slang']):
    slang_replace[row] = slang['formal'].iloc[i]

# Define std word dictionary
baku = pd.read_csv('lexicon/slang ke baku.csv')
std_word_replace = {}
for i, row in enumerate(baku['slang']):
    std_word_replace[row] = baku['baku'].iloc[i]

# Load TF model
model = tf.keras.models.load_model('saved_model/model')

# Load OHE for intent
with open("saved_model/encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load text vectorization layer
from_disk = pickle.load(open("saved_model/textvect.pkl", "rb"))
textvect = tf.keras.layers.TextVectorization.from_config(from_disk['config'])
# Adapt to dummy data
textvect.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
textvect.set_weights(from_disk['weights'])


# Create text cleaning function
def clean_text(text):
    new_text = []

    text = text.lower() # Lowercase

    # Remove punctuations
    text = text.translate(
        str.maketrans(
            '',
            '',
            string.punctuation
        )
    )

    # Split text into words then loop through each word
    for kata in text.split(): 
        # Keep word not in slang or standard word
        if kata not in (slang_replace|std_word_replace): 
            new_text.append(kata) 
        # Replace non-formal word with standard word
        elif kata in std_word_replace:
            new_text+=std_word_replace[kata].split() 
        # Replace slang with standard word
        elif kata in slang_replace:
            new_text+=slang_replace[kata].split() 

    # Join words without stopwords
    new_text = ' '.join(
        stemmer.stem(
            std_word_replace.get(
                word,
                word
            )
        ) for word in new_text if word not in stopwords 
    )
    
    return new_text


def bot_response(text):
    """Take text as function input then predict using model. Return response based on highest probability using numpy argmax    
    """
    text = clean_text(text)
    pred = model.predict([text])
    res = le.classes_[pred.argmax()]
    if textvect(text).numpy().max() > 1:
        for label_pred in intent_json['intents']:
            if label_pred['intent'] == res:
                response = label_pred['response']
    else:
        response = ['Maaf, saya tidak mengerti']
    
    dict_temp = []
    for i in range(len(pred[0])):
        temp = {le.classes_[i]: pred[0][i]}
        dict_temp.append(temp)
    return response


@app.route("/", methods=['GET','POST'])
def model_prediction():
    if request.method == "POST":
        content = request.json
        try:
            response = {"code": 200, "status":"OK", 
                        "result":bot_response(content['text'])}
            return jsonify(response)
        except Exception as e:
            response = {"code":500, "status":"ERROR", 
                        "result":{"error_msg":str(e)}}
            return jsonify(response)
    return "<p>Please insert your data in FE side.</p>"