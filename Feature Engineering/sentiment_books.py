import re
import pandas as pd
import numpy as np
from collections import defaultdict

# Set Pandas to display all rows of dataframes
pd.set_option('display.max_rows', 500)

# nltk
from nltk import tokenize

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Plotting tools
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('fivethirtyeight')


from tqdm import tqdm_notebook as tqdm
from tqdm import trange

with open('corpus_dict.pickle', 'rb') as handle:
    hp = pickle.load(handle)

analyzer = SentimentIntensityAnalyzer()

for book in tqdm(hp, desc='Progress'):
    print(book)
    for chapter in tqdm(hp[book], postfix=book):
        print('  ', chapter)
        text = hp[book][chapter].replace('\n', '')
        sentence_list = tokenize.sent_tokenize(text)
        sentiments = {'compound': 0.0, 'neg': 0.0, 'neu': 0.0, 'pos': 0.0}
        
        for sentence in sentence_list:
            vs = analyzer.polarity_scores(sentence)
            sentiments['compound'] += vs['compound']
            sentiments['neg'] += vs['neg']
            sentiments['neu'] += vs['neu']
            sentiments['pos'] += vs['pos']
            
        sentiments['compound'] = sentiments['compound'] / len(sentence_list)
        sentiments['neg'] = sentiments['neg'] / len(sentence_list)
        sentiments['neu'] = sentiments['neu'] / len(sentence_list)
        sentiments['pos'] = sentiments['pos'] / len(sentence_list)

        hp[book][chapter] = (hp[book][chapter], sentiments)
#     print()

compound_sentiments = [hp[book][chapter][2]['compound'] for book in hp for chapter in hp[book]]