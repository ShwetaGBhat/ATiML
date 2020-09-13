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

def chunk(book,chunk_size=10000):
    result = []
    chunk = []
    if(len(book)<chunk_size):
        print('less',chunk_size-len(chunk))
        for item in book:
            chunk.append(item)
        if chunk:
            result.append(chunk)
        return result
    else:
        for item in book:
            chunk.append(item)
            if len(chunk) == chunk_size:
                result.append(chunk)
                chunk = []
        if len(chunk)< chunk_size:
            #print('less',chunk_size-len(chunk))
            less = int(chunk_size-len(chunk))
            for i in range(0,less):
                chunk.append(result[0][i])
        # don't forget the remainder!
        if chunk:
            result.append(chunk)
        return result


stop_words=stopwords.words('english')
stop_words+=punctuation
stop_words+=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

# splitting the data 
data=pd.read_csv(r"C:/Personal/OVGU/SoSe2020/ATiML/Project/Gutenberg_English_Fiction_1k/master996.csv",delimiter=';',encoding='latin-1') 
data["id"]= data["book_id"].str.split(".", n = 1, expand = True)[0]
data.reset_index(inplace=True)
data.drop(["index"], axis=1, inplace=True)
print(data)

url = r"C:/Personal/OVGU/SoSe2020/ATiML/Project/Gutenberg_English_Fiction_1k/Gutenberg_19th_century_English_Fiction/"
#url = r"C:/Personal/OVGU/SoSe2020/ATiML/Project/Gutenberg_English_Fiction_1k/New/"
book_id=[]
content=[]
#data = [book_name,book_id,auhor,id]

for file in os.listdir(url):
    f=os.path.join(url, file)
    with open(f, encoding='UTF-8') as fp:
        soup = BeautifulSoup(fp,features="lxml")
        text = soup.get_text()
        word_tokens = word_tokenize(text)
        filtered_sentence = [] 
        for w in word_tokens:
            if w not in stop_words: 
                filtered_sentence.append(w)     #Text after removing stopwords
                #print(filtered_sentence) 
        lemma_word = []
        wordnet_lemmatizer = WordNetLemmatizer()
        for w in filtered_sentence:
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
TTR_chunk=[]

for book,genre,bookId in zip(features,labels,bookId):
    if ((len(book))== 0): 
        print(bookId,genre)
    else:
        chunks = chunk(book)
        ttr = 0
        for eachChunk in chunks: 
            num_unique = len(list(set(eachChunk)))
            num_words = len(eachChunk) #Total number of words in each book
            ttr = ttr + ((num_unique/num_words)*100) #lexical diversity score
        num_chunks = len(chunks)
        ttr = ttr/num_chunks
        print(bookId)
        book_id.append(bookId)
        TTR_chunk.append(round(ttr,2)

featuresDF = pd.DataFrame({"book_id":book_id,"lexical_diversity_score_onChunk":TTR_chunk})

featuresDF['normalized_ttr_chunk']=round((featuresDF['lexical_diversity_score_onChunk']-featuresDF['lexical_diversity_score_onChunk'].mean())/(featuresDF['lexical_diversity_score_onChunk'].max()-featuresDF['lexical_diversity_score_onChunk'].min()),2)

featuresDF.to_csv(r"C:/Personal/OVGU/SoSe2020/ATiML/Project/Manali/Features_ttr_withchunks.csv", header=True, index=False)
