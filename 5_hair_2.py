
import csv
import numpy as np
import shutil
from PIL import Image
import scipy.misc as spm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import cnn as CNN
import cv2
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def getnumbernoise():

    k = 0
    labels_file = open('dataset/attribute_list.csv')
    file_csv = csv.reader(labels_file)
    file_csv1 = csv.reader(labels_file)

    # remove noisy images
    for row in file_csv:
        if (row[1] == '-1'):# and (row[2] == '-1') and (row[3] == '-1') and (row[4] == '-1') and (row[5] == '-1'):
            k = k+1

    return k


def shapedata(k):
    labels_file = open('dataset/attribute_list.csv')
    file_csv = csv.reader(labels_file)

    count = 0
    i = -2
    countG = 0

    names = np.zeros(shape=(5000 - k,))
    newY = np.zeros(shape=(5000 - k, ))

    for row in file_csv:
        if (row[1] == '-1'): # and (row[2] == '-1') and (row[3] == '-1') and (row[4] == '-1') and (row[5] == '-1'):
            count = count +1
        elif(i > -1):
            newY[countG] = row[1]
            names[countG] = int(row[0])
            countG = countG + 1

        i = i + 1

    return newY, names


def fetch_image_data(names, length):

    picDat = np.ndarray(shape=(length, 256, 256,3))

    for i in range(0, length):
        img1 = image.load_img('dataset/newData/' + str(int(names[i])) + '.png' , target_size=((256, 256)))#, color_mode = "grayscale")
        x = image.img_to_array(img1)
        x = np.expand_dims(x, axis=0)
        # x = spm.imread('dataset/newData/' + str(int(names[i])) + '.png', mode='RGB')
        # x = np.resize(x, (32,32,3))
        picDat[i - 1] = x

    pd = picDat.reshape(picDat.shape[0], picDat.shape[1] * picDat.shape[2] * picDat.shape[3])

    return pd


def makenewdataset(y):
    # print('label length: ' + str(len(y[0])))
    i = 0
    x = y[1]
    for i in range(0, len(x)):
        shutil.copy('./dataset/faceDat/' + str(int(x[i])) + '.png', './dataset/newData/' + str(int(x[i])) + '.png')
        i = i + 1

    # print('new dataset length: ' + str(i))
    xLabels = fetch_image_data(x, len(y[0]))

    yLabels = y[0]

    p = 0.8
    part = int(len(y[0]) * p)

    tr_X = xLabels[:part]

    tr_Y = yLabels[:part]

    te_X = xLabels[part:]

    te_Y = yLabels[part:]

    tr_X = tr_X.astype('float32')
    te_X = te_X.astype('float32')
    tr_X = tr_X / 255.
    te_X = te_X / 255.

    # Change the labels from categorical to one-hot encoding
    train_Y_one_hot = to_categorical(tr_Y)
    test_Y_one_hot = to_categorical(te_Y)

    train_X, valid_X, train_label, valid_label = train_test_split(tr_X, train_Y_one_hot, test_size=0.2,random_state=13)

    return tr_X, train_Y_one_hot, te_X, test_Y_one_hot, valid_X, valid_label


def contruct_cnn(n, xTrain, yTrain, xTest, yTest, valid_X, valid_label):  #check data
    batch_size = 70
    epochs = 2
    num_classes = n

    classifier = CNN.build(num_classes)

    classifier.fit(xTrain, yTrain, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_X, valid_label))
    loss, acc = classifier.evaluate(xTest, yTest, verbose=0)
    # print(acc * 100)
    return acc


def train_knn_classifier(training_images, training_labels, test_images, test_labels):
    knn = KNeighborsClassifier(n_neighbors=6).fit(training_images, training_labels)
    accuracy = knn.score(test_images, test_labels)
    return accuracy


def inspect(train_X, train_Y, test_X, test_Y):
    print('Training data shape : ', train_X.shape, train_Y.shape)

    print('Testing data shape : ', test_X.shape, test_Y.shape)

    classes = np.unique(train_Y)

    nClasses = len(classes)

    print('Total number of outputs : ', nClasses)

    print('Output classes : ', classes)


def main():
    np.set_printoptions(threshold=np.inf)
    a,b,c,d,e,f = makenewdataset(shapedata(getnumbernoise()))
    inspect(a,b,c,d)
    # print(a.shape)
    # print(b.shape)
    print(train_knn_classifier(a,b,c,d))
    print(contruct_cnn(6,a,b,c,d,e,f))
    # print(c)
    # print(d)



if __name__ == '__main__':
        main()