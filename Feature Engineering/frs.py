import nltk
from nltk.corpus import stopwords
from string import punctuation
import pandas as pd
import os
from bs4 import BeautifulSoup
from itertools import chain
import re
import syllables
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize 


# splitting the data 
data=pd.read_csv(r"C:/Personal/OVGU/SoSe2020/ATiML/Project/Gutenberg_English_Fiction_1k/master996.csv",delimiter=';',encoding='latin-1') 
data["id"]= data["book_id"].str.split(".", n = 1, expand = True)[0]
data.reset_index(inplace=True)
data.drop(["index"], axis=1, inplace=True)
print(data)

#url = r"C:/Personal/OVGU/SoSe2020/ATiML/Project/Gutenberg_English_Fiction_1k/Gutenberg_19th_century_English_Fiction/"
url = r"C:/Personal/OVGU/SoSe2020/ATiML/Project/Gutenberg_English_Fiction_1k/New/"
book_id=[]
content=[]
#data = [book_name,book_id,auhor,id]

for file in os.listdir(url):
    f=os.path.join(url, file)
    with open(f, encoding='UTF-8') as fp:
        soup = BeautifulSoup(fp,features="lxml")
        text = soup.get_text()
        word_tokens = word_tokenize(text)
        lemma_word = []
        wordnet_lemmatizer = WordNetLemmatizer()
        for w in word_tokens:
            word1 = wordnet_lemmatizer.lemmatize(w, pos = "n")
            word2 = wordnet_lemmatizer.lemmatize(word1, pos = "v")
            word3 = wordnet_lemmatizer.lemmatize(word2, pos = ("a"))
            lemma_word.append(word3)        #Text after lemmatization
            #print(lemma_word)

        #print (text)
        content.append(lemma_word)
        book_id.append(file.split("-")[0])
        #print(file.split("-")[0])

#book_id  = [pg10067,pg1032]
#content = [bookwordspg10067,bookwordspg1032]

dataCorpus=pd.DataFrame({"book_id":book_id,"Content":content})
dataCorpus.reset_index(inplace=True)
dataCorpus.drop(["index"], axis=1, inplace=True)
#print(dataCorpus)
#datacorpus = [book_id, content]


df_mergedDF = pd.merge(dataCorpus,data, left_on="book_id", right_on="id", how="outer")
df_mergedDF.isna().sum(axis=0)
df1 = df_mergedDF[df_mergedDF['id'].notna()]

features = df1['Content']#, 'Book_Name','Author_Name', 'id']]
labels = df1['guten_genre'] 
bookId = df1['id']
feature_names = df1['guten_genre'].unique()
book_id=[]
FRS=[]

for book,genre,bookId in zip(features,labels,bookId):
    if (len(book)== 0): 
        print(bookId,genre)
    else :
        num_sentences = len(book.split('.')) #Total sentences
        num_syllables = syllables.estimate(book)
        num_unique = len(set(re.findall('\w+', book.lower()))) #Unique words in each book
        num_words = len(book.split()) #Total number of words in each book
        #print(bookId,num_words,num_unique)
        
        fres = round(206.835 - (1.015 * (num_words/num_sentences)) - (84.6 * (num_syllables/num_words)),2) #Flesh Reading Score - Ease of Readability
        #print(num_sentences,num_words,num_syllables,fres)
        print(bookId)
        book_id.append(bookId)
        FRS.append(fres)

featuresDF = pd.DataFrame({"book_id":book_id,"flesh_reading_score":FRS})

featuresDF['normalized_frs']=(featuresDF['flesh_reading_score']-featuresDF['flesh_reading_score'].mean())/(featuresDF['flesh_reading_score'].max()-featuresDF['flesh_reading_score'].min())

featuresDF.to_csv(r"C:/Personal/OVGU/SoSe2020/ATiML/Project/Manali/Features_frs.csv", header=True, index=False)
