from flask import Flask, request
import json
import string
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import telegram
from telegram.ext.updater import Updater
from telegram.update import Update
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.commandhandler import CommandHandler
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.filters import Filters
from sklearn.preprocessing import LabelEncoder

global bot
global TOKEN
global URL

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

# Remove certain words from stopwords
stopwords.remove('ok')
stopwords.remove('tidak')


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
    try:
        if textvect(text).numpy().max() > 1:
            for label_pred in intent_json['intents']:
                if label_pred['intent'] == res:
                    response = label_pred['response']
        else:
            response = ['Maaf, saya tidak mengerti']
    except:
        response = ['Maaf, saya tidak mengerti']

    dict_temp = []
    for i in range(len(pred[0])):
        temp = {le.classes_[i]: pred[0][i]}
        dict_temp.append(temp)
    print(dict_temp)
    print(le.classes_[pred.argmax()])
    return np.random.choice(response)


# URL = 'https://fiktifid-bot.herokuapp.com/'
TOKEN = 'Insert Token Here'
# bot = telegram.Bot(token=TOKEN)

# @app.route('/setwebhook', methods=['GET', 'POST'])
# def set_webhook():
#     # we use the bot object to link the bot to our app which live
#     # in the link provided by URL
#     s = bot.setWebhook(URL)
#     # something to let us know things work
#     if s:
#         return "webhook setup ok"
#     else:
#         return "webhook setup failed"
   
def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Halo, selamat datang. Ada yang bisa dibantu seputar rekrutmen PT Fiktif?")

def help(update: Update, context: CallbackContext):
    update.message.reply_text("""Hubungi developer untuk bantuan lebih lanjut.\n\rhttps://github.com/Riezn/.""")


def reply(update, context):
    user_input = str(update.message.text)
    update.message.reply_text(bot_response(user_input))


def error(update: Update, context: CallbackContext):
    print(f"update{update} caused error {context.error}")


def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))
    dp.add_handler(MessageHandler(Filters.text, reply))
    dp.add_error_handler(error)
    
    updater.start_polling()
    updater.idle(10)

    
main()
