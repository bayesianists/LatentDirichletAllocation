import os
import sys
import re

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

#Youssef
dir_path = os.path.dirname(os.path.realpath(__file__))
current_path = dir_path + '/../reuters21578/'
if sys.version_info[0] < 3 or True:
    directory = current_path
else:
    directory = os.fsencode(current_path)

#Joey
def removeSpecial(doc):
    return re.sub('[^A-Za-z]+', ' ', doc)

#Joey
def extract_text(file_name, dirPath):
    """
    Goes through .sgm file and extracts all bodies of text along with the corresponding topics
    :param - name of .sgm file
    :return - list of Document objects
    """

    texts = []
    topics = []
    # path = dirPath + 'reuters21578/' + file_name
    path = dirPath + file_name
    f = open(path, 'rb')
    data = f.read()
    soup = BeautifulSoup(data, features="html.parser")
    contents = soup.findAll('reuters')
    for content in contents:
        if content['topics'] != "NO":
            topic = content.findAll('topics')
            Ds = topic[0].findAll('d')
            tBin = 0
            for d in Ds:  # Iterating through topics for specific document
                if d.text == "earn":
                    tBin = 1

            textBody = content.findAll('body')
            if len(textBody) == 1:
                text = textBody[0].text  # The actual document (in textform, needs to be converted into BoW)
                texts.append(removeSpecial(text))
                topics.append(tBin)

    return texts, topics

#Youssef
def getTexts(numFilesToImport=-1):
    """
    :return: List of Document objects
    """
    texts = []
    topics = []
    i = 0
    for file in os.listdir(directory)[0:numFilesToImport]:
        i += 1
        if sys.version_info[0] < 3:
            filename = file
        else:
            filename = os.fsdecode(file)
        if filename.endswith(".sgm"):
            print("Importing data file", i)
            newTexts, newTopics = extract_text(filename, directory)
            texts += newTexts
            topics += newTopics
    return np.array(texts), np.array(topics)
