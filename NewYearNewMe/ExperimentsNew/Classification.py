import numpy as np
from sklearn import svm


def SVM(X, Y, test):
    clf = svm.SVC()
    clf.fit(X, Y)
    return clf.predict(test)


# X = np.array([[1, 0], [1, 0, 0], [0]])
# Y = np.array([1, 0, 1])
# test = np.array([[2, 1, 3], [3, 1, 5]])


def accuracy(X, Y, trainDataRatio=0.8):
    X = np.array(X)
    Y = np.array(Y)
    print(Y)
    print(X.shape)
    print(Y.shape)
    numPoints = len(X)
    numTrain = int(numPoints * trainDataRatio)
    # split = np.random.randint(low=1, high=(1 - data_split_ratio) * 100, size=1) / 100

    trainX = X[0:numTrain]
    trainY = Y[0:numTrain]

    testX = X[numTrain:]
    testY = Y[numTrain:]

    '''
    test_Y = Y[int(split * numPoints):int((split + data_split_ratio) * numPoints):1]

    training_Y = np.concatenate(
        (Y[0:int(split * numPoints):1],
         Y[int((split + data_split_ratio) * numPoints):np.size(Y):1])
    )

    test_X = X[int(split * numPoints):int((split + data_split_ratio) * numPoints):1]

    training_X = np.concatenate(
        (X[0:int(split * numPoints):1],
         X[int((split + data_split_ratio) * numPoints):np.size(X):1])
    )

    print(training_X.shape)
    print(training_Y.shape)
    # print(test_X.shape)
    results = SVM(training_X, training_Y, test_X)
    '''
    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)

    results = SVM(trainX, trainY, testX)

    # print(np.size(results))
    # print(np.size(test_Y))
    summ = 0
    for res, i in enumerate(results):
        if res == testY[i]:
            summ += 1
    return 1 - (summ / np.size(results))


'''
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
'''

if __name__ == '__main__':
    data_size = 2000

    # X, Y = generate_data('earn', data_size)

    print("---------------")

    # Accuracy = accuracy(X, Y, 0.05)
    # print(Accuracy)
