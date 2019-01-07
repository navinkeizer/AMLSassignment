# Libraries
import numpy as np
from PIL import Image
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import csv
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import shutil
import scipy
import timeit


# Function to import data and gray scaling, gives us histrograms

def getGreyscaleTupple(length):
    # initiates an array to store our data of size 32x32
    picDat = np.ndarray(shape=(length,32,32))
    # goes through all images in the directory and converts to grayscaled data, which is then stored in our array
    for i in range(1, length):
        x = scipy.misc.imread('dataset/faceDat/' + str(i) + '.png', mode='P')
        picDat[i - 1] = np.resize(x, (32,32))
    # reshape the data to be 2 dimensional
    pd = picDat.reshape(picDat.shape[0], picDat.shape[1]*picDat.shape[2])

    return pd



"""More explainations of these functions in the preProcessing1.py file"""


# function to split the training and testing data
def getData(x, y):

    # split data as follows
    xTrain = x[:3500]
    xTest = x[3500:]
    yTrain = y[:3500]
    yTest = y[3500:]

    return xTrain, yTrain, xTest, yTest

# Gets the labels of the images that are considered noise as all labels are -1. Produce new datalabel with 1 or 0
# As noise or no noise
def getY():

    noise_labels = np.zeros(shape=(5000,))
    i = -2

    labels_file = open('dataset/attribute_list.csv')
    file_csv = csv.reader(labels_file)

    for row in file_csv:
        if (row[1] == '-1') and (row[2] == '-1') and (row[3] == '-1') and (row[4] == '-1') and (row[5] == '-1'):
            noise_labels[i] = 1.
        elif (i > -1):
            noise_labels[i] = 0.
        i = i+1

    return noise_labels


# Functions to detect the noisy images
def constructSVM(xTr, yTr, xTe, yTe):

    clf = svm.SVC(gamma='scale')

    clf.fit(xTr, yTr)

    ans = clf.score(xTe, yTe)

    return ans, clf.get_params()


def contructMLP(xTr, yTr, xTe, yTe):

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    clf.fit(xTr, yTr)

    ans = clf.score(xTe, yTe)

    return ans, clf.get_params()


def oneclassSVM(xTr, yTr, xTe, yTe):

    clf = svm.OneClassSVM(kernel="sigmoid", gamma='scale')
    clf.fit(xTr)
    y_pred_train = clf.predict(xTr)
    y_pred_test = clf.predict(xTe)
    # converts to -1 and 1 scale, both test abd training data
    yTr[yTr == 0.0] = -1
    yTe[yTe == 0.0] = -1
    # gets the error in both data
    error1 = np.mean(yTr != y_pred_train)
    error2 = np.mean(yTe != y_pred_test)
    # get the average of the error
    return np.add(error1,error2) /2


def isolationForrest(xTr, yTr, xTe, yTe, contamination, max_samples, n_estimators):

    clf = IsolationForest(behaviour='new', contamination=contamination, max_samples=max_samples, n_estimators=n_estimators)
    clf.fit(xTr)
    y_pred_train = clf.predict(xTr)
    y_pred_test = clf.predict(xTe)
    # converts to -1 and 1 scale, both test abd training data
    yTr[yTr == 0.0] = -1
    yTe[yTe == 0.0] = -1
    # gets the error in both data
    error1 = np.mean(yTr != y_pred_train)
    error2 = np.mean(yTe != y_pred_test)
    # get the average of the error
    return np.add(error1, error2) /2


def elliptic_envelope(xTr, yTr, xTe, yTe):

    clf = EllipticEnvelope()
    clf.fit(xTr)
    y_pred_train = clf.predict(xTr)
    y_pred_test = clf.predict(xTe)
    # converts to -1 and 1 scale, both test abd training data
    yTr[yTr == 0.0] = -1
    yTe[yTe == 0.0] = -1
    # gets the error in both data
    error1 = np.mean(yTr != y_pred_train)
    error2 = np.mean(yTe != y_pred_test)
    # get the average of the error
    return np.add(error1, error2) /2


# Main function, prints all the accuracy of the methods as well as some parameters
def main():

    a,b,c,d = getData(getGreyscaleTupple(5000), getY())

    # --------------------------------

    tmp = constructSVM(a, b, c, d)

    print('SVM accuracy: ' + str(repr(tmp[0])))
    # print('SVM parameters: ' + str(tmp[1]))

    # --------------------------------

    tmp1 = contructMLP(a, b, c, d)
    print('MLP accuracy: ' + str(repr(tmp1[0])))
    print('MLP parameters: ' + str(tmp1[1]))

    # --------------------------------

    print('One class SVM accuracy: ' + str(oneclassSVM(a, b, c, d)))
    #
    # # --------------------------------
    #
    print('Isolation forrest accuracy: ' + str(isolationForrest(a, b, c, d, 'auto','auto', 150)))
    #
    # # --------------------------------
    #
    print('Elliptic Envelope accuracy: ' + str(elliptic_envelope(a, b, c, d)))
    #

# Execute the main code
if __name__ == "__main__":
    start = timeit.default_timer()
    main()
    stop = timeit.default_timer()

    print('Time: ', stop - start)


