from logging import debug
from flask import Flask, render_template, request
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer
import spacy
app = Flask(__name__)

import joblib
# load the model
vector = joblib.load('vector.pkl')
model = joblib.load('kdtree_model.pkl')
nlp = spacy.load('en_core_web_sm')

@app.route('/')
def hello():
    return render_template('base.html')

@app.route('/classify',methods = ['POST'])
def classify():
    ip = request.form.get('ip')
    doc = nlp(ip)
    list1=[]
    for token in doc:
        if not token.is_punct:
            if not token.is_stop:
                if not token.is_digit:
                    list1.append(str(token.lemma_))

    embedding = vector.transform(list1).toarray()

    distance , idx = model.query(embedding , k = 3)
    res= []
    df = pd.read_csv('famous_people.csv')
    for i, value in list(enumerate(idx[0])):
        # print(f"Name : {person['Name'][value]}")
        # print(f"Distance : {distance[0][i]}")
        # print(f"URL : {person['URI'][value]}")  
        res.append(df['Name'][value])
        res.append(df['URI'][value])
    return render_template('base.html',prediction_text='{res}')

if __name__=='__main__':
    app.run(debug=True)