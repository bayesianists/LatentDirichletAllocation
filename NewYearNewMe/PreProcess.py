import sys
#https://github.com/blei-lab/onlineldavb/blob/master/onlineldavb.py
import numpy as np
import pandas as pd
import gensim
import nltk
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem.porter import *

# KTH-kollega
from NewYearNewMe.ExtractTexts import getTexts

stemmer = SnowballStemmer('english')

# Uncomment this the first time you run it!
nltk.download('wordnet')
nltk.download('stopwords')

# Youssef
stop = stopwords.words('english')


# lemmatizing and stemming
def lem_stem(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preProcessStolen(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in STOPWORDS and len(token) >= 3:
            result.append(lem_stem(token))
    return result


# returns dictionary and list of numpy docs where each word is a pointer to the vocab/dict
def preProcess(numFilesToImport=-1):
    texts, topics = getTexts(numFilesToImport=numFilesToImport)
    preProcessedDocs = []
    for i in range(len(texts)):
        preProcessedDocs.append(preProcessStolen(texts[i]))

    # Create a dictionary
    dictionary = gensim.corpora.Dictionary(preProcessedDocs)
    dictionary.filter_extremes(no_below=10, no_above=0.5)
    # bow_corpus = [dictionary.doc2bow(doc) for doc in preProcessedDocs]
    if sys.version_info[0] < 3:
        idxDocs = [np.array(filter(lambda a: a != -1, dictionary.doc2idx(doc))) for doc in preProcessedDocs]
    else:
        idxDocs = [np.array(list(filter(lambda a: a != -1, dictionary.doc2idx(doc)))) for doc in preProcessedDocs]
    # npCorpus = np.array([np.array(doc) for doc in preProcessedDocs])
    # print(np.array(idxDocs).shape)
    # lengths = [len(doc) for doc in idxDocs]
    # print(lengths)
    print("NumDocs (M):", len(idxDocs))
    print("Vocab Size (V):", len(dictionary))
    print("Average Doc Length (N):", np.mean([len(doc) for doc in idxDocs]))
    return dictionary, idxDocs


if __name__ == '__main__':
    preProcess(1)
