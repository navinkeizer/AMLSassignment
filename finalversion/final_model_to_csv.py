# Implements all the final models to produce the CSV files
# of the test data provided on moodle

from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
import facialFeatureExtractor as ffe
import facialFeatureExtractor2 as ffe2
from sklearn.model_selection import GridSearchCV
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import os
import haircolour as hc
import numpy as np
from keras.preprocessing import image
import csv
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cnn as CNN
import matplotlib.pyplot as plt
from keras.models import load_model
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


# Gets the data from feature extractor, scaling included
def getdata_scaled(j):
    xLabels, yLabels, names = ffe.extract_features_labels(j)

    p = 0.8

    part = int(len(yLabels) * p)

    xTrain = xLabels[:part]

    tr_Y = yLabels[:part]

    xTest = xLabels[part:]

    te_Y = yLabels[part:]

    tr1_X = xTrain.reshape(xTrain.shape[0], xTrain.shape[1] * xTrain.shape[2])

    te1_X = xTest.reshape(xTest.shape[0], xTest.shape[1] * xTest.shape[2])

    scaler = StandardScaler()

    sclllll = scaler.fit(tr1_X)
    StandardScaler(copy=True, with_mean=True, with_std=True)

    tr_X = scaler.transform(tr1_X)
    te_X = scaler.transform(te1_X)

    return tr_X, tr_Y, te_X, te_Y, names, sclllll

# Gets the data from feature extractor,
def getdata_unscaled(j):
    xLabels, yLabels, names = ffe.extract_features_labels(j)

    p = 0.8

    part = int(len(yLabels) * p)

    xTrain = xLabels[:part]

    tr_Y = yLabels[:part]

    xTest = xLabels[part:]

    te_Y = yLabels[part:]

    tr1_X = xTrain.reshape(xTrain.shape[0], xTrain.shape[1] * xTrain.shape[2])

    te1_X = xTest.reshape(xTest.shape[0], xTest.shape[1] * xTest.shape[2])

    return tr1_X, tr_Y, te1_X, te_Y, names


# Gets the data from the image pixels in grayscale
def getdata_MC():
    picDat = np.ndarray(shape=(100, 256, 256,1))
    testnames = np.ndarray(shape=(100,))

    for i in range(1, 101):
        testnames[i-1]=i
        img1 = image.load_img('dataset/testing_dataset/' + str(int(i)) + '.png' , target_size=((256, 256)), color_mode = "grayscale")
        x = image.img_to_array(img1)
        x = np.expand_dims(x, axis=0)
        picDat[i - 1] = x
        pd = picDat.reshape(picDat.shape[0], picDat.shape[1] * picDat.shape[2] * picDat.shape[3])
    return pd, testnames


# scales the data
def shapedata(x,names):
    x = x.astype('float32')
    x = x / 255.
    return x,names


# decision tree sclassifier implementation to get the prediction
def train_decision_tree(training_images, training_labels, test_images,names):

    clf = tree.DecisionTreeClassifier()

    clf.fit(training_images, training_labels)

    predicted_classes = np.argmax(np.round(clf.predict(test_images)), axis=1)

    for i in range(0, len(names)):
        print(str(names[i]), '.png ,', str(predicted_classes[i]))

    return 0


# MLP implementation for predictions
def mlp_scaled1(training_images, training_labels, test_images, test_labels, T1):

    clf = MLPClassifier(solver='adam', alpha=1, hidden_layer_sizes=(100,), activation='relu', learning_rate= 'adaptive')

    clf.fit(training_images, training_labels)

    ans = clf.predict(T1)

    return ans

def mlp_scaled2(training_images, training_labels, test_images, test_labels, T1):

    clf = MLPClassifier(solver='lbfgs', learning_rate='adaptive', alpha=1, hidden_layer_sizes=(50, 50, 50), activation='relu')

    clf.fit(training_images, training_labels)
        
    ans = clf.predict(T1)

    return ans


def mlp_scaled3(training_images, training_labels, test_images, test_labels, T1):
    clf = MLPClassifier(solver='adam', learning_rate='constant', alpha=1, hidden_layer_sizes=(50, 50, 50),
                        activation='identity')

    clf.fit(training_images, training_labels)

    ans = clf.predict(T1)

    return ans


def mlp_scaled4(training_images, training_labels, test_images, test_labels, T1):
    clf = MLPClassifier(solver='adam', learning_rate='constant', alpha=1, hidden_layer_sizes=(50, 100, 50),
                        activation='tanh')

    clf.fit(training_images, training_labels)

    ans = clf.predict(T1)

    return ans

# SVM implementations for prediction
def svm_test(training_images, training_labels, test_images, test_labels, T1):

    clf = svm.SVC(gamma=1e-05, C=0.01, kernel='linear')
    clf.fit(training_images, training_labels)

    ans = clf.predict(T1)
    return ans

def svm2(training_images, training_labels, test_images, test_labels, T1):

    clf = svm.SVC(gamma='scale', C=1, kernel='rbf')

    clf.fit(training_images, training_labels)

    ans = clf.predict(T1)

    return ans


def main():


    T1 = ffe2.extract_features_labels(0)[0]
    T2 = ffe2.extract_features_labels(0)[1]
    T1 = T1.reshape(T1.shape[0], T1.shape[1] * T1.shape[2])

    # task1, glasses, SVM O

    print('glasses')
    a,b,c,d,e,f = getdata_scaled(2)
    StandardScaler(copy=True, with_mean=True, with_std=True)
    T1 = f.transform(T1)
    svm_output = mlp_scaled4(a,b,c,d,T1)
    svm_output[svm_output == 0.0] = -1
    for i in range(0, len(T1)):
        print(str(T2[i]) , '.png ,' , str(svm_output[i]))




    # task2, emotions, MLP

    print('emotions')
    a1, b1, c1, d1,e1,f1 = getdata_scaled(3)
    StandardScaler(copy=True, with_mean=True, with_std=True)
    T1 = f1.transform(T1)
    emo_output = mlp_scaled1(a1, b1, c1, d1, T1)
    emo_output[emo_output == 0.0] = -1
    for i in range(0, len(T1)):
        print(str(T2[i]) , '.png ,' , str(emo_output[i]))


    # task3, Age, SVM NO

    print('Age')
    a2,b2,c2,d2,e2,f2 = getdata_scaled(4)
    StandardScaler(copy=True, with_mean=True, with_std=True)
    T1 = f2.transform(T1)
    age_output = mlp_scaled3(a2,b2,c2,d2, T1)
    age_output[age_output == 0.0] = -1
    for i in range(0, len(T1)):
        print(str(T2[i]) , '.png ,' , str(age_output[i]))


    # task4, human , MLP
    print('human')
    a3, b3, c3, d3, e3,f3 = getdata_scaled(5)
    StandardScaler(copy=True, with_mean=True, with_std=True)
    T1 = f3.transform(T1)
    humn_output = mlp_scaled2(a3, b3, c3, d3 , T1)
    humn_output[humn_output == 0.0] = -1
    for i in range(0, len(T1)):
        print(str(T2[i]), '.png ,', str(humn_output[i]))


    # task 5 hair, Decision Tree
    a, b, c, d, e, f, names = hc.get_data(hc.get_images(), hc.get_labels())
    i = shapedata(getdata_MC()[0], getdata_MC()[1])[0]
    j = shapedata(getdata_MC()[0],getdata_MC()[1])[1]
    train_decision_tree(a,b,i,j)


if __name__ == '__main__':
    main()
