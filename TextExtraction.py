import re
from collections import defaultdict
import numpy as np
from bs4 import BeautifulSoup
import os
from nltk.corpus import stopwords

# import nltk
# nltk.download('stopwords')

stop = stopwords.words('english')

dir_path = os.path.dirname(os.path.realpath(__file__))
current_path = dir_path + '/reuters21578'
directory = os.fsencode(current_path)


class Document:

    def __init__(self, topics, text):
        self.topics = topics
        self.text = text

    def __str__(self):
        return str("TOPICS:" + "\n" + str(self.topics) + "\n" + self.text)


def extract_text(file_name):
    """
    Goes through .sgm file and extracts all bodies of text along with the corresponding topics
    :param - name of .sgm file
    :return - list of Document objects
    """

    documents = []
    path = 'reuters21578/' + file_name
    f = open(path, 'rb')
    data = f.read()
    soup = BeautifulSoup(data)
    contents = soup.findAll('reuters')
    for content in contents:
        if content['topics']:
            topic = content.findAll('topics')
            Ds = topic[0].findAll('d')
            topics = []
            for d in Ds:  ### Iterating through topics for specific document
                topics.append(d.text)

            textBody = content.findAll('body')
            if len(textBody) == 1:
                text = textBody[0].text  ### The actual document (in textform, needs to be converted into BoW)
                documents.append(Document(topics, text))
    return documents


def join_document():
    """
    :return: List of Document objects
    """
    documents = []
    i = 0
    for file in os.listdir(directory):
        i += 1
        filename = os.fsdecode(file)
        if filename.endswith(".sgm"):
            print(i)
            documents += extract_text(filename)
    return documents


def create_vocabulary(documents):
    """
    :param documents: List of document objects
    :return: List of words  (Vocabulary)
    """
    vocabulary = defaultdict(int)
    for d in documents:
        for word in re.sub('[^A-Za-z]+', ' ', d.text).split(' '):
            word = word.casefold()
            if word != 'reuter' and word != '' and word not in stop:
                vocabulary[word] += 1
    return vocabulary


def write_list_to_file(dictionary, maxint=1000):
    """
    Stores vocabulary in file 'vocabulary.txt'
    :param dictionary: Vocabulary with frequency for each word
    :param maxint: maximum number of words allowed
    :return: nothing
    """
    maxint -= 1
    f = open("vocabulary.txt", "w")
    i = 0
    for word, frequency in sorted(dictionary.items(), key=lambda x: x[1], reverse=True):
        if i > maxint:
            return
        f.write(word + '\n')
        i += 1


def get_word_vector(word, vocab):
    """
    Creates a word vector
    :param word: word that is to be transformed into vector
    :param vocab: Vocabulary
    :return: Word vector
    """
    vector = np.zeros((len(vocab)))
    if word in vocab:
        vector[vocab.index(word)] = 1
    return vector


def get_vocab():
    """
    :return: Vocabulary as a vector of words
    """
    with open('vocabulary.txt') as f:
        lines = [line.rstrip() for line in f]
        return lines


def make_corpus(vocabulary, cap=8000):
    """
    :param vocabulary: Vocabulary as vector of words
    :return: Corpus
    """

    CORPUS_FILE_NAME = "numpyCorpus.npy"
    if os.path.isfile(CORPUS_FILE_NAME):
        print("Loading corpus from file system...")
        corpus = np.load(CORPUS_FILE_NAME)
        print("Loaded corpus!")
        return corpus

    documents = []
    for doc in join_document():
        i = 0
        document = []
        for word in re.sub('[^A-Za-z]+', ' ', doc.text).split(' '):
            word = word.casefold()
            document.append(get_word_vector(word, vocabulary))
        i += 1
        documents.append(np.array(document))
        if i > cap:
            break

    corpus = np.array(documents)

    np.save(CORPUS_FILE_NAME, corpus)
    print("Saved corpus to file system!")

    return corpus


if __name__ == "__main__":
    write_list_to_file(create_vocabulary(join_document()), 100)
    print(make_corpus(get_vocab(), 500))
