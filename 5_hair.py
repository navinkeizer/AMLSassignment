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

    picDat = np.ndarray(shape=(length, 32, 32,3))

    for i in range(0, length):
        x = spm.imread('dataset/newData/' + str(int(names[i])) + '.png', mode='RGB')
        x = np.resize(x, (32,32,3))
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

    scaler = StandardScaler()

    scaler.fit(tr_X)
    StandardScaler(copy=True, with_mean=True, with_std=True)

    X_train = scaler.transform(tr_X)
    X_test = scaler.transform(te_X)

    return X_train, tr_Y, X_test, te_Y


def inspect(train_X, train_Y, test_X, test_Y):
    print('Training data shape : ', train_X.shape, train_Y.shape)

    print('Testing data shape : ', test_X.shape, test_Y.shape)

    classes = np.unique(train_Y)

    nClasses = len(classes)

    print('Total number of outputs : ', nClasses)

    print('Output classes : ', classes)


def train_decision_tree(training_images, training_labels, test_images, test_labels):

    clf = tree.DecisionTreeClassifier()

    clf.fit(training_images, training_labels)

    ans = clf.score(test_images, test_labels)
    # cm = confusion_matrix(test_images, clf.predict(test_images))
    return ans#, cm


def train_knn_classifier(training_images, training_labels, test_images, test_labels):
    knn = KNeighborsClassifier(n_neighbors=6).fit(training_images, training_labels)
    accuracy = knn.score(test_images, test_labels)
    # print(knn.predict(test_images))
    #knn_predictions = knn.predict(test_images)
    #cm = confusion_matrix(test_images, knn_predictions)

    return accuracy#, cm


def train_naive_bayes(training_images, training_labels, test_images, test_labels):
    gnb = GaussianNB().fit(training_images, training_labels)
    #gnb_predictions = gnb.predict(tes)
    accuracy = gnb.score(test_images, test_labels)
    #cm = confusion_matrix(y_test, gnb_predictions)
    return accuracy


def train_mlp(training_images, training_labels, test_images, test_labels):

    clf = MLPClassifier(hidden_layer_sizes=(50,50,50), activation='logistic', alpha=1, learning_rate='constant', solver='adam')
    clf.fit(training_images, training_labels)
    ans = clf.score(test_images, test_labels)
    # print(clf.predict(test_images))
    # confusion_matrix(test_labels, clf.predict(test_labels))
    return ans


def mlp_param_selection(X, y, nfolds):
    parameter_space = {
        # 'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,), (100,100,100)],
        'hidden_layer_sizes': [(50, 50, 50)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'adam', 'sgd'],
        'alpha': [1, 0.1, 0.01, 0.001],
        'learning_rate': ['constant', 'adaptive', 'invscaling'],
    }
    mlp = MLPClassifier()
    grid_search = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=nfolds)
    grid_search.fit(X, y)

    return grid_search.best_params_


def main():
    np.set_printoptions(threshold=np.inf)

    a,b,c,d = makenewdataset(shapedata(getnumbernoise()))
    # print(a.shape, b.shape, c.shape, d.shape)
    # inspect(a,b,c,d)
    # print(train_decision_tree(a,b,c,d))
    # print(train_knn_classifier(a,b,c,d))
    # print(train_naive_bayes(a,b,c,d))
    # print(train_mlp(a,b,c,d))
    # print(mlp_param_selection(c,d,2))

if __name__ == '__main__':
        main()