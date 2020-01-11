import re
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


def create_vocabulary(documents):
    vocabulary = []
    for d in documents:
        for word in re.sub('[^A-Za-z]+', ' ', d.text).split(' '):
            word = word.casefold()
            if word not in vocabulary and word != 'reuter':
                vocabulary.append(word)
    return vocabulary


def write_list_to_file(list):
    f = open("vocabulary.txt", "w")
    for element in list:
        f.write(element + '\n')
