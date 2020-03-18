import re

import numpy as np
from sklearn import svm
from FailedLDAModel import TextExtraction as te
from nltk.corpus import stopwords

stop = stopwords.words('english')
"""
    Experiment description:
    
    In the text classiﬁcation problem, we wish to classify a document into two or more mutually exclusive classes. 
    As in any classiﬁcation problem, we may wish to consider generative approaches or discriminative approaches. 
    In particular, by using one LDA module for each class, we obtain a generative model for classiﬁcation. 
    It is also of interest to use LDA in the discriminative framework, and this is our focus in this section. 
    
    A challenging aspect of the document classiﬁcation problem is the choice of features. 
    Treating individual words as features yields a rich but very large feature set (Joachims, 1999). 
    One way to reduce this feature set is to use an LDA model for dimensionality reduction. In particular, 
    LDA reduces any document to a ﬁxed set of real-valued features—the posterior Dirichlet parameters γ∗(w) 
    associated with the document. It is of interest to see how much discriminatory information we lose in 
    reducing the document description to these parameters. 
    
    ***We want to classify a document into two or more mutually exclusive classes.
    ***Choice of features. Using many words for features gives rich, but large feature set. Use LDA model for 
    dimensionality.
    ***How much discriminatory info do we lose??
    
    Experiment instructions:
    
    We conducted two binary classiﬁcation experiments 
    using the Reuters-21578 dataset. The dataset contains 8000 documents and 15,818 words. 

    In these experiments, we estimated the parameters of an LDA model on all the documents, without reference to 
    their true class label. We then trained a support vector machine (SVM) on the low-dimensional representations 
    provided by LDA and compared this SVM to an SVM trained on all the word features. 
    
    Using the SVMLight software package (Joachims, 1999), 
    we compared an SVM trained on all the word features with those trained on features induced by a 50-topic LDA model. 
    Note that we reduce the feature space by 99.6 percent in this case.

"""


def SVM(X, Y, test):
    clf = svm.SVC()
    clf.fit(X, Y)
    return clf.predict(test)


# X = np.array([[1, 0], [1, 0, 0], [0]])
# Y = np.array([1, 0, 1])
# test = np.array([[2, 1, 3], [3, 1, 5]])


def accuracy(X, Y, data_split_ratio=0.2):
    split = np.random.randint(low=1, high=(1 -data_split_ratio)*100, size=1) / 100

    test_Y = Y[int(split * data_size):int((split + data_split_ratio) * data_size):1]
    training_Y = np.concatenate(
        (Y[0:int(split * data_size):1], Y[int((split + data_split_ratio) * data_size):np.size(Y):1]))
    test_X = X[int(split * data_size):int((split + data_split_ratio) * data_size):1]
    training_X = np.concatenate(
        (X[0:int(split * data_size):1], X[int((split + data_split_ratio) * data_size):np.size(X):1]))

    results = SVM(training_X, training_Y, test_X)

    summ = 0

    print(np.size(results))
    print(np.size(test_Y))
    for res, i in enumerate(results):
        if res == test_Y[i]:
            summ += 1
    return 1 - (summ / np.size(results))


def generate_data(str_topic, max_documents=8000):
    vocab = te.get_vocab()
    data = []
    topics = []
    V = len(vocab)
    corpus = te.join_document()
    i = 0
    for document in corpus:
        i += 1
        if i >= max_documents:
            break
        topic = 0
        if str_topic in document.topics:
            topic = 1
        topics.append(topic)
        features = np.zeros(V)
        for word in re.sub('[^A-Za-z]+', ' ', document.text).split(' '):
            word = word.casefold()
            if word != 'reuter' and word != '' and word not in stop:
                if word in vocab:
                    features[vocab.index(word)] += 1
        data.append(features)
    return np.array(data), np.array(topics)


if __name__ == '__main__':
    data_size = 2000

    X, Y = generate_data('earn', data_size)

    print("---------------")

    Accuracy = accuracy(X, Y, 0.05)
    print(Accuracy)
