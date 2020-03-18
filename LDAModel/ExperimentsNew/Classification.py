import numpy as np
from sklearn import svm

# ADRIAN

def SVM(X, Y, test):
    clf = svm.SVC()
    clf.fit(X, Y)
    return clf.predict(test)



def accuracy(X, Y, trainDataRatio=0.8):
    X = np.array(X)
    Y = np.array(Y)
    #print(Y)
    #print(X.shape)
    #print(Y.shape)
    numPoints = len(X)
    numTrain = int(numPoints * trainDataRatio)
    # split = np.random.randint(low=1, high=(1 - data_split_ratio) * 100, size=1) / 100

    trainX = X[0:numTrain]
    trainY = Y[0:numTrain]

    testX = X[numTrain:]
    testY = Y[numTrain:]



    results = SVM(trainX, trainY, testX)

    summ = 0
    for i, res in enumerate(results):
        if res == testY[i]:
            summ += 1

    return summ / np.size(results)


if __name__ == '__main__':
    data_size = 2000

    # X, Y = generate_data('earn', data_size)

    print("---------------")

    # Accuracy = accuracy(X, Y, 0.05)
    # print(Accuracy)
