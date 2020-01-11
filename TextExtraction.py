import re
from collections import defaultdict

from bs4 import BeautifulSoup
import os

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
        if content['topics'] != "NO":
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
    vocabulary = defaultdict(int)
    for d in documents:
        for word in re.sub('[^A-Za-z]+', ' ', d.text).split(' '):
            word = word.casefold()
            if word != 'reuter' and word != '':
                vocabulary[word] += 1
    return vocabulary


def write_list_to_file(dictionary, maxint=10000):
    f = open("vocabulary.txt", "w")
    i = 0
    for word, frequency in sorted(dictionary.items(), key=lambda x: x[1], reverse=True):
        if i > maxint:
            return
        f.write(word + '\n')
        i += 1


def get_word_vector(word, vocab):
    vector = [0] * len(vocab)
    if word in vocab:
        vector[vocab.index(word)] = 1
    return vector


def get_vocab():
    with open('vocabulary.txt') as f:
        lines = [line.rstrip() for line in f]
        return lines


def make_corpus(vocabulary):
    documents = []
    for doc in join_document():
        document = []
        for word in re.sub('[^A-Za-z]+', ' ', doc.text).split(' '):
            word = word.casefold()
            document.append(get_word_vector(word, vocabulary))
        documents.append(document)


make_corpus(get_vocab())
# write_list_to_file(create_vocabulary(join_document()))
