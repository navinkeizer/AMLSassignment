from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
import facialFeatureExtractor as ffe
from sklearn.model_selection import GridSearchCV
import itertools
from sklearn.preprocessing import StandardScaler


# Get the x and y coordinates of the 68 points using the facialFeatureExtractor
# Gets these arrays and maps them to the local xLabels and yLabels variables
# These are then split into training and testing data
# The p variable stands for the percentage that we want in the training set
# Lastly we reshape the x / image variables to work with our models


def getdata(j):

    xLabels, yLabels = ffe.extract_features_labels(j)

    p = 0.8

    part = int(len(yLabels) * p)

    xTrain = xLabels[:part]

    tr_Y = yLabels[:part]

    xTest = xLabels[part:]

    te_Y = yLabels[part:]

    tr1_X = xTrain.reshape(xTrain.shape[0], xTrain.shape[1]*xTrain.shape[2])

    te1_X = xTest.reshape(xTest.shape[0], xTest.shape[1]*xTest.shape[2])
    
    
    scaler = StandardScaler()

    scaler.fit(tr1_X)
    StandardScaler(copy=True, with_mean=True, with_std=True)

    tr_X = scaler.transform(tr1_X)
    te_X = scaler.transform(te1_X)
    
    
    return tr_X, tr_Y, te_X, te_Y


# Takes the x and y training data and fits it to an SVM model
# Uses the SciKit Learn svm.SVC function 
# Used the built in score function to get the performance accuracy 

def train_svm1(training_images, training_labels, test_images, test_labels):

    clf = svm.SVC(gamma='scale')

    clf.fit(training_images, training_labels)

    ans = clf.score(test_images, test_labels)

    return ans


def train_svm(training_images, training_labels, test_images, test_labels, gamma, C, kernel):

    clf = svm.SVC(gamma=gamma, kernel=kernel, C=C)

    clf.fit(training_images, training_labels)

    ans = clf.score(test_images, test_labels)

    return ans


# Takes the x and y training data and fits it to an MLP model
# Uses the SciKit Learn MLP classifier function
# Used the built in score function to get the performance accuracy 

def train_mlp1(training_images, training_labels, test_images, test_labels):

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    clf.fit(training_images, training_labels)

    ans = clf.score(test_images, test_labels)

    return ans


def train_mlp(training_images, training_labels, test_images, test_labels, activation, alpha, hiddenlayers, learningrate, solver):

    clf = MLPClassifier(solver=solver, alpha=alpha, hidden_layer_sizes=hiddenlayers,activation=activation, learning_rate=learningrate, random_state=1)

    clf.fit(training_images, training_labels)

    ans = clf.score(test_images, test_labels)

    return ans


# Function to get the optimal parameters for our SVM model


def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    kernels = ['linear','rbf','poly', 'sigmoid']
    param_grid = {'kernel' : kernels, 'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    return grid_search.best_params_


def mlp_param_selection(X, y, nfolds):
    parameter_space = {
        #'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,), (100, 100, 100)],
        'activation': ['identity','logistic','tanh', 'relu'],
        'solver': ['lbfgs', 'adam', 'sgd'],
        'alpha': [1, 0.1, 0.01, 0.001, 0.0001],
        'learning_rate': ['constant', 'adaptive', 'invscaling'],
    }
    mlp = MLPClassifier()
    grid_search = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=nfolds)
    grid_search.fit(X, y)
    
    return grid_search.best_params_


# Runs theSVM and MLP models on all binary classifications

def runeverybinary():
    names = ['eyeglasses', 'smiling', 'young', 'human']

    for i in range(2,6):
        a, b, c, d = getdata(i)
        print(names[i-2])
        print('SVM accuracy : ')
        print(train_svm1(a,b,c,d))
        print('MLP accuracy : ')
        print(train_mlp1(a,b,c,d))

    return 0


def main():
    runeverybinary()


if __name__ == '__main__':
    main()