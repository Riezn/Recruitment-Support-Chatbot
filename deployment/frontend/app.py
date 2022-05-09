from flask import Flask
import os
import requests
import json
import string
import pickle
import numpy as np
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from telegram.ext.updater import Updater
from telegram.update import Update
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.commandhandler import CommandHandler
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.filters import Filters
import spacy

global TOKEN
global URL
global URL_backend
global PORT

app = Flask(__name__)


# Define stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()


# Define stopword
factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()
stopwords.remove('ok')
stopwords.remove('oh')
stopwords.remove('tidak')
stopwords.remove('ya')


# Load json file
f = open('intent/intent.json', 'r')
intent_json = json.load(f)


baku = pd.read_csv('lexicon/baku.csv')
std_word_replace = {}
for i, row in enumerate(baku['slang']):
    std_word_replace[row] = baku['baku'].iloc[i]


# Create text cleaning function
def clean_text(text):
    new_text = []
    text = text.lower() # Lowercase
    # Loop each word in a sentence
    for kata in text.split(): 
        # Keep word not in slang or standard word
        if kata not in std_word_replace: 
            new_text.append(kata) 
        # Replace non-formal word with standard word
        elif kata in std_word_replace:
            new_text+=std_word_replace[kata].split() 
    # Join words without stopwords after stemming
    new_text = ' '.join(
        stemmer.stem(word) for word in new_text if word not in stopwords
    )
    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    return new_text


# Load OHE for intent
with open("saved_model/encoder.pkl", "rb") as f:
    le = pickle.load(f)


# Load ner model
ner = spacy.load('model-best')


URL_backend = ''
URL = ''
TOKEN = ''


@app.route('/')
def index():
    return '.'


def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Halo, selamat datang. Ada yang bisa dibantu seputar rekrutmen PT Fiktif?")


def help(update: Update, context: CallbackContext):
    update.message.reply_text("""Hubungi developer untuk bantuan lebih lanjut.\n\rhttps://github.com/Riezn/""")


def reply(update, context):
    user_input = str(update.message.text)
    clean_input = clean_text(user_input)
    data = {'user_input': clean_input}
    r = requests.post(URL_backend, json=data)
    resp = r.json()
    label_idx = int(resp['prediction'])
    try:
        try:
            dok = ner(data['user_input'])
            dok.ents[0].label_.lower()
        except:
            dok = ner('ds')
        if label_idx != 1000:
            i = 0
            while i < len(intent_json['intents']):
                if le.classes_[label_idx] == intent_json['intents'][i]['intent'] and (le.classes_[label_idx] != 'responsibilities' or le.classes_[label_idx] != 'qualification'):
                    response = intent_json['intents'][i]['response']
                    break
                elif le.classes_[label_idx] == 'responsibilities':
                    if dok.ents[0].label_.lower() == "scientist":
                        response = ["Berikut adalah tanggung jawab yang akan diberikan untuk posisi Data Scientist:\n- Merancang dan mengembangkan berbagai solusi Machine Learning dan Deep Learning untuk meningkatkan pengalaman pengguna bagi konsumen\n- Berkolaborasi dengan seluruh elemen bisnis dan bertanggung jawab untuk merencanakan solusi end-to-end berbasis data untuk menyelesaikan permasalahan bisnis\n- Menjadi thinking partner bagi stakeholder lain untuk memperbaiki alur perjalanan data dan penggunaannya dalam operasional perusahaan, misal merancang proses feedback-loop atau human-in-the-loop untuk meningkatkan performa model secara berkelanjutan"]
                        break
                    elif dok.ents[0].label_.lower()=='engineer':
                        response = ["Berikut adalah tanggung jawab yang akan diberikan untuk posisi Data Engineer:\n- Membangun dan menjaga end-to-end data pipeline dari input dan output heterogen\n- Menangani dan mengelola data warehouse\n- Membantu tim mentransformasikan data (ETL) dan mengembangkan proses ETL dari beberapa sumber\n- Menganalisis dan mengorganisir data mentah \n- Memastikan kualitas data dan integrasi data"]
                        break
                    elif dok.ents[0].label_.lower()=='analis':
                        response = ["Berikut adalah tanggung jawab yang akan diberikan untuk posisi Data Analyst:\n- Mengumpulkan dan menyediakan data untuk membantu stakeholder lain meningkatkan metrik bisnis perusahaan dan retensi pelanggan\n- Menganalisis data untuk menemukan insight yang dapat ditindaklanjuti seperti membuat funnel conversion analysis, cohort analysis, long-term trends, user segmentation, dan dapat membantu meningkatkan kinerja perusahaan dan mendukung pengambilan keputusan yang lebih baik\n - Mengidentifikasi kebutuhan dan peluang bisnis berdasarkan data yang tersedia"]
                        break
                elif le.classes_[label_idx] == 'qualification':
                    if dok.ents[0].label_.lower()=='scientist':
                        response = ["Untuk posisi Data Scientist ada beberapa kualifikasi yang harus dipenuhi:\n1. Memiliki gelar sarjana di bidang informatika, ilmu komputer, statistika, matematika, atau bidang lain yang berhubungan\n2. Memiliki pemahaman mendasar tentang Statistika Analitik, Machine Learning, Deep Learning untuk menyelesaikan permasalahan bisnis\n3. Memiliki pengalaman kerja di bidang Data Science selama 1-3 tahun\n4. Memiliki pemahaman dan pengalaman tentang Big Data\n5. Memiliki kemampuan bekerja sama, kepemimpinan, dan problem solving yang baik"]
                        break
                    elif dok.ents[0].label_.lower()=='engineer':
                        response = ["Untuk posisi Data Engineer ada beberapa kualifikasi yang harus dipenuhi:\n1. Memiliki gelar sarjana di bidang informatika, ilmu komputer, statistika, matematika, atau bidang lain yang berhubungan\n2. Memiliki pengalaman bekerja dengan tools untuk ETL seperti AWS Glue, SSIS, Informatica, dll.\n3. Memiliki pengalaman kerja di bidang Data Engineer selama 1-3 tahun\n4. Memiliki pemahaman yang baik tentang ETL, SQL, dan noSQL\n5. Memiliki kemampuan bekerja sama, kepemimpinan, dan problem solving yang baik"]
                        break
                    elif dok.ents[0].label_.lower()=='analis':
                        response = ["Untuk posisi Data Analyst ada beberapa kualifikasi yang harus dipenuhi:\n1. Memiliki gelar sarjana di bidang informatika, ilmu komputer, statistika, matematika, atau bidang lain yang berhubungan\n2. Memiliki pemahaman mendasar tentang Statistika Analitik dan Inferensial untuk mencari peluang bisnis\n3. Memiliki pengalaman kerja di bidang Data Analyst selama 1-3 tahun\n4. Memiliki pemahaman dan pengalaman tentang Big Data serta visualisasi dengan tools seperti Power BI, Tableau, dll.\n5. Memiliki kemampuan bekerja sama, kepemimpinan, dan problem solving yang baik"]
                        break
                else:
                    i+=1
        else:
            response = ['Maaf, Kak. Aku tidak mengerti chatnya...']
    except:
        print("error")
        response = ['Maaf, Kak. Aku tidak mengerti chatnya...\n\n\rTerjadi error']
    update.message.reply_text(np.random.choice(response))



def error(update: Update, context: CallbackContext):
    print(f"update{update} caused error {context.error}")


PORT = int(os.environ.get('PORT', '443'))


def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))
    dp.add_handler(MessageHandler(Filters.text, reply))
    dp.add_error_handler(error)
    
    # updater.start_polling()
    # Add handlers
    updater.start_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=TOKEN,
        webhook_url=URL+TOKEN
        )
    updater.idle()

  
main()