from flask import Flask, render_template, url_for, request
from util import *
from sklearn.externals import joblib
import pandas as pd
from tensorflow.keras import models, layers, preprocessing as kprocessing
import transformers
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

def load_model():
    TFIDF = joblib.load('models/TFIDF.mnb')
    BERT = models.load_model('models/Bert')
    DistilBERT = models.load_model('models/DistilBERT')
    return TFIDF, BERT, DistilBERT


def predict(feature_matrix, model):
    propability = model.predict(feature_matrix, batch_size= 64)
    predicted = []
    i = 0
    while (i < len(propability)):
        try:
            result = {0: 'BUSINESS', 1: 'ENTERTAINMENT', 2: 'POLITICS & WORLDS', 3: 'SPORT', 4: 'TECH'}[np.argmax(propability[i])]
            predicted.append(result)
        except:
            predicted.append('UNKNOWN')
        i = i + 1
    return predicted

def initiate(content, TFIDF_model, BERT_model, DistilBERT_result):
    table = {'text': [content]}
    input_dtf = pd.DataFrame(table)
    lst_stopwords = nltk.corpus.stopwords.words("english")
    input_dtf["text_clean"] = input_dtf["text"].apply(lambda x: 
          utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, 
          lst_stopwords=lst_stopwords))
    BERT_features = create_feature_matrix(input_dtf['text_clean'], transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True))
    DistilBERT_features = create_feature_matrix(input_dtf['text_clean'], transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True))
    TFIDF_result = TFIDF_model.predict(input_dtf['text_clean'])
    BERT_result = predict(BERT_features,BERT_model)
    DistilBERT_result = predict(DistilBERT_features,DistilBERT_model)
    return TFIDF_result[0], BERT_result[0], DistilBERT_result[0]

app = Flask(__name__)

TFIDF_model, BERT_model, DistilBERT_model = load_model()

@app.route('/', methods=['POST','GET'])
def index():
    return render_template('index.html', TFIDF_result = '', BERT_result = '', DistilBERT_result = '' )

@app.route('/onClick', methods=['POST','GET'])
def onClick():
    if request.method == 'POST':
        content = request.form['content']
        TFIDF_result, BERT_result, DistilBERT_result = initiate(content, TFIDF_model, BERT_model, DistilBERT_model)
        return render_template('index.html', TFIDF_result = TFIDF_result, BERT_result = BERT_result, DistilBERT_result = DistilBERT_result)
    return render_template('index.html', TFIDF_result = '', BERT_result = '', DistilBERT_result = '')

if __name__ == "__main__":
    app.run(debug=True)