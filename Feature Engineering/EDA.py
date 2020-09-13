#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 14:03:35 2020

@author: shwetabhat
"""
# =============================================================================

import pandas as pd
import string
import nltk

from string import punctuation
import matplotlib.pyplot as plt
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
# =============================================================================

data1=pd.read_csv(r"/Users/shwetabhat/Desktop/ATML/ProjectTask/DataPrepAndFeatureExt2/Preprocessed.csv") 
data1.columns.values
data1 = data1[data1['id'].notna()]
data1 = data1[data1['Content'].notna()]

data1.to_csv(r"/Users/shwetabhat/Desktop/ATML/ProjectTask/DataPrepAndFeatureExt2/PreprocessedwithoutNA.csv", header=True, index=False)
data=pd.read_csv(r"/Users/shwetabhat/Desktop/ATML/ProjectTask/DataPrepAndFeatureExt2/PreprocessedwithoutNA.csv") 
stopWords=stopwords.words('english')
stopWords+=punctuation
stopWords+=['a','b','c'
           'd','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

# =============================================================================
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
    words = [w for w in words if not w in stopWords]
#Performing lemmatisation on the words
    words = [wl.lemmatize(w) for w in words]
    for w in words:
        new_sent = new_sent +' '+ w
    return new_sent


data['clean_data'] = data['Content'].apply(my_preprocessor)
data.shape

del data['book_id_x']
del data['book_id_y']
data.to_csv(r"/Users/shwetabhat/Desktop/ATML/ProjectTask/DataPrepAndFeatureExt2/PreprocessedNoStopWords.csv", header=True, index=False)

data.info()
####### Data distribution.##################################
####### EDA to visualize on  Genre.##################################

 data['guten_genre'].value_counts().plot(kind = 'bar')
 plt.show()
 plt.clf()
 
# =============================================================================
data['cnt']=1

####### EDA to visualize on  Autor Name.##################################

Group_Author=data.groupby('Author_Name')['cnt'].count()
Group_Author.plot(kind = 'bar')
plt.show()
plt.clf()
# ================================================================================================================================
####### EDA to visualize on  Autor Name and Genre.######################################################################
Group_Author=Group_Author.to_frame().reset_index()
Group_Author.to_csv(r"/Users/shwetabhat/Desktop/ATML/ProjectTask/DataPrepAndFeatureExt2/GroupByAuth.csv", header=True, index=False)

data.groupby(['guten_genre','Author_Name'])['cnt'].count().plot(kind = 'bar')
plt.show()
plt.clf()
# ================================================================================================================================
wordcloud = WordCloud(background_color="black").generate(str(data['clean_data']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
# ================================================================================================================================

####### EDA to visualize WordCloud per genre.########################################################################################################
data=pd.read_csv(r"/Users/shwetabhat/Desktop/ATML/ProjectTask/DataPrepAndFeatureExt2/PreprocessedNoStopWords.csv") 
wordcloud = WordCloud(background_color="black").generate(str(data['clean_data']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
# ================================================================================================================================

####### EDA to visualize WordCloud per genre.########################################################################################################
wordcloud = WordCloud(background_color="black").generate(str(data.loc[data['guten_genre'] =='Detective and Mystery'][['clean_data']]))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
# ================================================================================================================================
wordcloud = WordCloud(background_color="black").generate(str(data.loc[data['guten_genre'] =='Sea and Adventure'][['clean_data']]))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
# ================================================================================================================================
 
wordcloud = WordCloud(background_color="black").generate(str(data.loc[data['guten_genre'] =='Ghost and Horror'][['clean_data']]))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
# ================================================================================================================================
wordcloud = WordCloud(background_color="black").generate(str(data.loc[data['guten_genre'] =='Humorous and Wit and Satire'][['clean_data']]))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
# ================================================================================================================================
wordcloud = WordCloud(background_color="black").generate(str(data.loc[data['guten_genre'] =='Western Stories'][['clean_data']]))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
# ================================================================================================================================
wordcloud = WordCloud(background_color="black").generate(str(data.loc[data['guten_genre'] =='Love and Romance'][['clean_data']]))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
# ================================================================================================================================
wordcloud = WordCloud(background_color="black").generate(str(data.loc[data['guten_genre'] =='Christmas Stories'][['clean_data']]))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
#================================================================================================================================
wordcloud = WordCloud(background_color="black").generate(str(data.loc[data['guten_genre'] =='Allegories'][['clean_data']]))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
#================================================================================================================================


