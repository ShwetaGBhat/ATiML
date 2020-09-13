#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 21:03:51 2020

@author: shwetabhat
"""

# Starting with meta data feature extractions
#Importing the file
import pandas as pd
books=pd.read_csv(r"/Users/shwetabhat/Desktop/ATML/ProjectTask/DataPrepAndFeatureExt2/PreprocessedwithoutNA.csv") 

books.reset_index(inplace = True)
books.head(5)

#Importing NLTK libariaries
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

import numpy as np

# can run if needs subset of books
"""
books = books[0:25]
from collections import Counter
print(Counter(books.genre))
"""

# books.genre.unique    check unqiue genres

#This fucntion is taken from stackoverflow
def cleaning(var):
    """Take a string. Returns a string with only lowercase letters and the space between words."""
    plain_string = ""
    for x in var:
        x = x.lower()
        if (('a' <= x and x <= 'z') or x == ' '):
            plain_string += x
        elif x == '\'':  # any apostrophes(') are replaced by a space
            plain_string += ' '
    while '  ' in plain_string:  # any multiple spaces are replaced by a single space
        plain_string = plain_string.replace('  ', ' ')
    return plain_string

"""Defining Functions for tokenising sentence/words/charcters using nltk"""

import nltk
from nltk import sent_tokenize, word_tokenize, FreqDist
def token_sentence(text):
    sentences = nltk.Text(sent_tokenize(text)) #sentence token
    return len(sentences) # returns total number of sentences after tokeinsing on passes text

def token_word(text):
    words = nltk.Text(word_tokenize(((text)))) #word token
    return len(words) #returns total number of words after tokeinsing on passed text

def cal_len(text):
    #gives lenght of string passes
    return len(text)

# copying the dataframe to new 
new_books = pd.DataFrame({"data":books["Content"]})

"""Calculating average lenght of a book, avg number of sentences and words in a book"""

# total lenght of each book
books['book_lenght'] = books['Content'].apply(len)

#total lenght of all the books
total_lenght = books['book_lenght'].sum() #stored in a variable

#total sentences in each of the book
books['book_sen'] =  books['Content'].apply(token_sentence)

#total sentence lenght in all the books
total_sen = books['book_sen'].sum() #stored in a variable

#total words in each of the book
books['book_word'] =  books['Content'].apply(token_word)

#total words in all the books
total_words = new_books['book_word'].sum() #stored in a variable

# Average Book lenght
def lenght_book(text): 
    return len(text)/total_lenght

#Average number of sentences in a book
def number_sen(text):
    return token_sentence(text)/total_sen

#Average number of words in a book
def number_word(text):
    total_words = new_books['book_word'].sum() #stored in a variable
    return token_word(text)/total_words

# Total words
def tot_word(text):
  words = nltk.Text(word_tokenize(((text))))
  return len(words)


"""how many stop words in a book out of total number of words"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words("english")) #using nltk stop_words only not additional
def number_of_stop_words(text):
    tokens_of_words = word_tokenize(text) #using nltk
    get_stop_words = [word for word in tokens_of_words if word in stop_words]
    len_of_get_stop_words = len(get_stop_words) #number of stop_words
    len_of_total_words = len(tokens_of_words) #number of total words
    ratio = len_of_get_stop_words/len_of_total_words
    return ratio

"""How many Nouns in a book . Assumption is literay would have more nouns , romace shall have lower.it can corresponde to the number of charcters also."""

def get_number_of_nouns(text):
    list_of_nouns = []
    list_of_verbs = []
    list_of_adjectives = []
    list_of_numbers = []
    list_of_puch = []
    list_of_proper_nouns=[]
    for i in nltk.pos_tag(word_tokenize(text)):
        if i[1][0:2] == 'NN': # all noun tags start with NN
          list_of_nouns.append(i)
        elif i[1][0:2] == 'VB':
          list_of_verbs.append(i)
        elif i[1][0:2] == 'JJ':
          list_of_adjectives.append(i)
        elif i[1][0:2] == 'NNP':
          list_of_proper_nouns.append(i)
        elif i[1][0:2] == 'CD': #cardinal numbers
          list_of_numbers.append(i)
        elif i[1][0:2] == '.': #number of puctuations
          list_of_puch.append(i)
    return len(list_of_nouns)/tot_word(text), len(list_of_verbs)/tot_word(text), len(list_of_adjectives)/tot_word(text), len(list_of_proper_nouns)/tot_word(text), len(list_of_numbers)/tot_word(text),len(list_of_puch)/tot_word(text)


#list_of_meta_features = [
#    lenght_book 
#    ,number_sen 
#    ,number_word 
#    ,number_of_stop_words 
#     ,get_number_of_nouns
#    ,number_of_point
#    ]
#for i in list_of_meta_features:
#    print("Done:",i.__name__)
#    df_meta[i.__name__] = df_meta.data.apply(i)
 
df_meta = new_books.copy()

df_meta["lenght_book"] = df_meta.data.apply(lenght_book)
df_meta["number_sen"] = df_meta.data.apply(number_sen)
df_meta["number_word"] = df_meta.data.apply(number_word)
df_meta["number_of_stop_words"] = df_meta.data.apply(number_of_stop_words)
df_meta["get_number_of_nouns"] = df_meta.data.apply(get_number_of_nouns)


       
df_meta[['noun', 'verb', 'adj', 'ProperNoun', 'num', 'Puch']] = pd.DataFrame(df_meta['get_number_of_nouns'].tolist(), index=df_meta.index)


df_meta.columns=['data', 'lengthOfBook', 'NoOfSentences', 'NoOfWords', 'number_of_stop_words', 'noun', 'verb',
       'adj', 'ProperNoun', 'num', 'Puch']

df_meta.head(5)

final=pd.concat([books,df_meta], axis=1)
final.to_csv(r"/Users/shwetabhat/Desktop/ATML/ProjectTask/DataPrepAndFeatureExt2/dataWithBookStats.csv")

ttr_Processsed=pd.read_csv(r"/Users/shwetabhat/Desktop/ATML/ProjectTask/DataPrepAndFeatureExt2/Features_ttr_withchunks.csv") 

mergedTTR_bookStats_df=pd.merge(final,ttr_Processsed, how='inner',left_on='book_id', right_on='book_id', sort=False)

mergedTTR_bookStats_df.to_csv(r"/Users/shwetabhat/Desktop/ATML/ProjectTask/DataPrepAndFeatureExt2/dataWithBookStatsWithTTR.csv")
   
ttr_Processsed=pd.read_csv(r"/Users/shwetabhat/Desktop/ATML/ProjectTask/DataPrepAndFeatureExt2/Features_FrsFinal.csv") 

mergedTTR_bookStats_df=pd.merge(final,ttr_Processsed, how='inner',left_on='book_id', right_on='book_id', sort=False)

mergedTTR_bookStats_df.to_csv(r"/Users/shwetabhat/Desktop/ATML/ProjectTask/DataPrepAndFeatureExt2/data_Sntnce_Cmplxty_TTR_frs.csv")

 
import nltk
import pandas as pd
books=pd.read_csv(r"/Users/shwetabhat/Desktop/ATML/ProjectTask/DataPrepAndFeatureExt2/dataWithBookStatsWithTTR_Updated.csv")

books.columns
def get_number_of_Conjunctions(text):
    tot_wordCnt=0
    tot_tags=0
    tot_ConjPrep=0
    tokens=nltk.word_tokenize(text)
    tot_wordCnt=len(tokens)
    for i in nltk.pos_tag(tokens):
            if i[1][0:2] == 'NN' or i[1][0:2] == 'NNPS' or i[1][0:2] == 'NNS' or i[1][0:2] == 'NNP': # all noun tags start with NN
              tot_tags+=1
            elif i[1][0:2] == 'IN':x
                tot_ConjPrep+=1
    #print(tot_wordCnt,tot_tags)        
    return tot_tags/tot_wordCnt,tot_ConjPrep/tot_wordCnt


books["get_number_of_Conjunctions"] = books["data"].apply(get_number_of_Conjunctions)

books[['Nouns','Conj-Prep']] = pd.DataFrame(books['get_number_of_Conjunctions'].tolist(), index=books.index)
del  books['get_number_of_Conjunctions']

books.columns

##Slower methods.
###%timeit books['no_ofCommas']=books['Content'].str.strip().str.split(',').apply(len)
##%timeit books["no_ofCommas"]=books['Content'].map(lambda x: [i.strip() for i in x.split(",")])
##This one is fast 
books['no_ofCommas']=books['Content'].map(lambda x: len([i.strip() for i in x.split(",")]))
books['no_coln']=books['Content'].map(lambda x: len([i.strip() for i in x.split(":")]))
books['no_period']=books['Content'].map(lambda x: len([i.strip() for i in x.split(".")]))
books['no_doubleQuotes']=books['Content'].map(lambda x: len([i.strip() for i in x.split("'")]))

count = lambda l1,l2: sum([1 for x in l1 if x in l2])
books['Puch']=books["Content"].apply(lambda s: count(s, punctuation))

accumulate = lambda l1,l2: [x for x in l1 if x in l2]
accumulated=books["Content"].apply(lambda s: accumulate(s, punctuation))

books.to_csv(r"/Users/shwetabhat/Desktop/ATML/ProjectTask/DataPrepAndFeatureExt2/dataWithBookStatsWithTTR_Updated.csv")

books=pd.read_csv(r"/Users/shwetabhat/Desktop/ATML/ProjectTask/DataPrepAndFeatureExt2/data_Sntnce_Cmplxty_TTR_frs.csv")


# =============================================================================
#Remove the following.
#=============================================================================
# def get_number_of_PropNouns(text):
#     tot_wordCnt=0
#     tot_tags=0
#     tokens=nltk.word_tokenize(text)
#     tot_wordCnt=len(tokens)
#     for i in nltk.pos_tag(tokens):
#             if i[1][0:2] == 'NNP': # all noun tags start with NN
#               tot_tags+=1
#     #print(tot_wordCnt,tot_tags)        
#     return tot_tags/tot_wordCnt
# 
# 
# books["PropNouns"] = books["Content"].apply(get_number_of_PropNouns)
# books.columns
# books["PropNouns"]
# books["PropNouns"] = books["PropNouns"] /books["PropNouns"].sum()
# books.to_csv(r"/Users/shwetabhat/Desktop/ATML/ProjectTask/DataPrepAndFeatureExt2/data_Sntnce_Cmplxty_TTR_frs.csv")
# 
# del books["ProperNoun"]
# =============================================================================



import pickle

with open(r"/Users/shwetabhat/Desktop/ATML/ProjectTask/DataPrepAndFeatureExt2/Sentiment/Sentiment_dict.pickle", 'rb') as f:
    x = pickle.load(f)
    data=pd.DataFrame({"Compound":x[0],"Neg":x[1],"Neutral":x[2],"Positive":x[3]})
                       
print(data)


mergedTTR_bookStats_df=pd.concat([books,data],axis=1)
mergedTTR_bookStats_df.to_csv(r"/Users/shwetabhat/Desktop/ATML/ProjectTask/DataPrepAndFeatureExt2/data_Sntnce_Cmplxty_TTR_frs_Senti.csv")
# =============================================================================
#Naive Bayes on Dooc2Vec.
# =============================================================================

import pandas as pd
train=pd.read_csv(r"/Users/shwetabhat/Desktop/ATML/ProjectTask/DataPrepAndFeatureExt2/Train_doc2vec.csv")
test=pd.read_csv(r"/Users/shwetabhat/Desktop/ATML/ProjectTask/DataPrepAndFeatureExt2/Test_doc2vec.csv")
mergedDF=pd.concat([train,test],axis=0)
# =============================================================================
#ONE_HOT Encoding on  Author Name.
# =============================================================================
# 
 import pandas as pd
 import numpy as np
 from sklearn.preprocessing import OneHotEncoder
 # creating instance of one-hot-encoder
 enc = OneHotEncoder(handle_unknown='ignore')
 # passing bridge-types-cat column (label encoded values of bridge_types)
 enc_df = pd.DataFrame(enc.fit_transform(train[['Author_Name']]).toarray())
 # merge with main df bridge_df on key values
  mergedDF.join(enc_df)
 
# =============================================================================
