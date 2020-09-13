#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 19:27:19 2020

@author: shwetabhat
"""

def my_preprocessor(books):
    wl = WordNetLemmatizer()
    new_sent = ""
    tokens = word_tokenize(books)

# convert to lower case
    tokens = [w.lower() for w in tokens]
#Removing words whose length is less than 3
    words = [word for word in tokens if len(word)>3]
# remove remaining tokens that are not alphabetic
    words = [word for word in words if word.isalpha()]
# filter out stop words
    #words = [w for w in words if not w in stopWords]
#Performing lemmatisation on the words
    words = [wl.lemmatize(w) for w in words]
    for w in words:
        new_sent = new_sent +' '+ w
    return new_sent


from sklearn.model_selection import train_test_split 
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
import os
stopWords=stopwords.words('english')
stopWords+=punctuation
stopWords+=['a','b','c'
           'd','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

# splitting the data 
data=pd.read_csv(r"/Users/shwetabhat/Desktop/ATML/ProjectTask/Gutenberg_English_Fiction_1k/master996.csv",delimiter=';',encoding='latin-1') 
data["id"]= data["book_id"].str.split(".", n = 1, expand = True)[0]
data.reset_index(inplace=True)
data.drop(["index"], axis=1, inplace=True)

    
url = r"/Users/shwetabhat/Desktop/ATML/ProjectTask/Gutenberg_English_Fiction_1k/Gutenberg_19th_century_English_Fiction/"
book_id=[]
content=[]

for file in os.listdir(url):
    f=os.path.join(url, file)
    with open(f) as fp:
        soup = BeautifulSoup(fp)
        text = soup.get_text()
        #print (text)
        content.append(text)
        book_id.append(file.split("-")[0])
        #print(file.split("-")[0])
        
dataCorpus=pd.DataFrame({"book_id":book_id,"Content":content})
dataCorpus.reset_index(inplace=True)
dataCorpus.drop(["index"], axis=1, inplace=True)

df_mergedDF = pd.merge(dataCorpus,data, left_on="book_id", right_on="id", how="outer")
df_mergedDF.isna().sum(axis=0)
#df_mergedDF[df_mergedDF['id']!=df_mergedDF['book_id_x']]
df1 = df_mergedDF[df_mergedDF['id'].notna()]

features = df1['Content']#, 'Book_Name','Author_Name', 'id']]
labels = df1['guten_genre'] 
feature_names = df1['guten_genre'].unique()
train, test, train_labels, test_labels = train_test_split(features, labels, test_size = 0.35, random_state = 42,stratify=labels)
vectorizer = CountVectorizer(stop_words=stopWords,preprocessor=my_preprocessor)
trainVect=vectorizer.fit_transform(train)
trainFeatures=vectorizer.get_feature_names()
testVect=vectorizer.fit_transform(test)
testFeatures=vectorizer.get_feature_names()
trainfeatureDF = pd.DataFrame({'TrainFeatures': trainFeatures}) 
testFeaturesDF = pd.DataFrame({'TestFeatures': testFeatures}) 
df_mergedDF.to_csv(r"/Users/shwetabhat/Desktop/ATML/ProjectTask/Preprocessed.csv", header=False, index=False)
trainfeatureDF.to_csv(r"/Users/shwetabhat/Desktop/ATML/ProjectTask/trainfeature.csv", header=False, index=False)
testFeaturesDF.to_csv(r"/Users/shwetabhat/Desktop/ATML/ProjectTask/testFeatures.csv", header=False, index=False)
