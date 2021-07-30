import numpy as np
import pandas as pd
import joblib
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer
from sklearn.neighbors import KDTree
from nltk.stem import WordNetLemmatizer
pd.options.mode.chained_assignment = None

df = pd.read_csv('famous_people.csv')

nlp = spacy.load('en_core_web_sm')

# Reading the files
k=0
pd.options.mode.chained_assignment = None

df["new_Text"]=""
for i in df["Text"]:
    list1=[]
    doc = nlp(i)
    for token in doc:
        if not token.is_punct:
            if not token.is_stop:
                if not token.is_digit:
                    list1.append(str(token.lemma_))
                
    df["new_Text"][k] = " ".join(list1)
    k+=1

tfidf = TfidfVectorizer(stop_words='english')
train_tfidf = tfidf.fit_transform(df.new_Text).toarray()

df["tfidf"] = list(train_tfidf)

kdtree = KDTree(train_tfidf)


# saving the vector
joblib.dump(tfidf,'vector.pkl')
# saving the model
joblib.dump(kdtree,'kdtree_model.pkl')