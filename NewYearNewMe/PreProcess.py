import sys
import numpy as np
import pandas as pd
import gensim
import nltk
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem.porter import *

from NewYearNewMe.ExtractTexts import getTexts

stemmer = SnowballStemmer('english')



stop = stopwords.words('english')

#Joey
# lemmatizing and stemming
def lem_stem(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

#Joey
def preProcessStolen(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in STOPWORDS and len(token) >= 3:
            result.append(lem_stem(token))
    return result

#Adrian
def getTextsAbstracts():
    documents = pd.read_csv('papers2017.csv', error_bad_lines=False)
    processed_docs = documents['abstract']
    return processed_docs

#Joey
# returns dictionary and list of numpy docs where each word is a pointer to the vocab/dict
def preProcess(numFilesToImport=-1, loadFromFile=False, reuters=True):
    if loadFromFile:
        print("Loading pre-processed data from hard drive!")
        dictionary = np.load("ProcessedData/vocab.npy", allow_pickle=True)
        idxDocs = np.load("ProcessedData/corpus.npy", allow_pickle=True)
        topics = np.load("ProcessedData/topics.npy")
    else:
        print("Pre-processing data!")
        if reuters:
            texts, topics = getTexts(numFilesToImport=numFilesToImport)
        else:
            texts = getTextsAbstracts()
            topics = None

        preProcessedDocs = []
        for i in range(len(texts)):
            preProcessedDocs.append(preProcessStolen(texts[i]))

        # Create a dictionary
        dictionary = gensim.corpora.Dictionary(preProcessedDocs)
        dictionary.filter_extremes(no_below=10, no_above=0.5)

        if sys.version_info[0] < 3:
            idxDocs = [np.array(filter(lambda a: a != -1, dictionary.doc2idx(doc))) for doc in preProcessedDocs]
        else:
            idxDocs = [np.array(list(filter(lambda a: a != -1, dictionary.doc2idx(doc)))) for doc in preProcessedDocs]

        np.save("ProcessedData/vocab", dictionary)
        np.save("ProcessedData/corpus", idxDocs)
        if reuters:
            np.save("ProcessedData/topics", topics)

    print("NumDocs (M):", len(idxDocs))
    print("Vocab Size (V):", len(dictionary))
    print("Average Doc Length (N):", np.mean([len(doc) for doc in idxDocs]))

    return dictionary, idxDocs, topics

#Youssef
def generateFreqList(corpus, V):
    M = len(corpus)
    docFreqs = np.zeros((M, V))
    for i, doc in enumerate(corpus):
        for w in doc:
            docFreqs[i][w] += 1
    return docFreqs


if __name__ == '__main__':
    preProcess(1)
