# Mainly used by taking these functions in the binary classification classes

# Libraries
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
import facialFeatureExtractor as ffe
from sklearn.model_selection import GridSearchCV
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


# import matplotlib.pyplot as plt

# Plots the learning curve, depending on what model is used
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# Get the x and y coordinates of the 68 points using the facialFeatureExtractor
# Gets these arrays and maps them to the local xLabels and yLabels variables
# These are then split into training and testing data
# The p variable stands for the percentage that we want in the training set
# Lastly we reshape the x / image variables to work with our models

def getdata(j):
    # Extracts the x and y labels as well as the names
    xLabels, yLabels, names = ffe.extract_features_labels(j)
    # print(names[1], names[2])
    p = 0.8

    part = int(len(yLabels) * p)

    xTrain = xLabels[:part]

    tr_Y = yLabels[:part]

    xTest = xLabels[part:]

    te_Y = yLabels[part:]

    names_test = names[part:]

    tr1_X = xTrain.reshape(xTrain.shape[0], xTrain.shape[1] * xTrain.shape[2])

    te1_X = xTest.reshape(xTest.shape[0], xTest.shape[1] * xTest.shape[2])

    # Scales the image/x data to increase performance
    scaler = StandardScaler()

    scaler.fit(tr1_X)
    StandardScaler(copy=True, with_mean=True, with_std=True)

    tr_X = scaler.transform(tr1_X)
    te_X = scaler.transform(te1_X)

    # print(names_test)

    return tr_X, tr_Y, te_X, te_Y, names_test


# Takes the x and y training data and fits it to an SVM model
# Uses the SciKit Learn svm.SVC function
# Used the built in score function to get the performance accuracy

def train_svm1(training_images, training_labels, test_images, test_labels, names):
    clf = svm.SVC(gamma='scale')

    clf.fit(training_images, training_labels)

    an_sscore = clf.score(test_images, test_labels)
    # predicts the labels of test images
    ans = clf.predict(test_images)
    # Converts the 0s to -1s to be clear for classification purposes
    ans[ans == 0.0] = -1

    # prints out the name and prediction for the entire test data
    # This is then copied into excel in order to produce the csv
    # The function then outputs the accuracy score

    for i in range(0, len(names)):
        print(str(names[i]), '.png ,', str(ans[i]))

    return an_sscore


def train_svm(training_images, training_labels, test_images, test_labels, names, gamma, C, kernel):
    clf = svm.SVC(gamma=gamma, kernel=kernel, C=C)

    clf.fit(training_images, training_labels)

    an_score = clf.score(test_images, test_labels)
    # predicts the labels of test images
    ans = clf.predict(test_images)
    # Converts the 0s to -1s to be clear for classification purposes

    ans[ans == 0.0] = -1
    # prints out the name and prediction for the entire test data
    # This is then copied into excel in order to produce the csv
    # The function then outputs the accuracy score
    for i in range(0, len(names)):
        print(str(names[i]), '.png ,', str(ans[i]))

    return an_score


# Takes the x and y training data and fits it to an MLP model
# Uses the SciKit Learn MLP classifier function
# Used the built in score function to get the performance accuracy

def train_mlp1(training_images, training_labels, test_images, test_labels):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    clf.fit(training_images, training_labels)

    ans = clf.score(test_images, test_labels)

    return ans


def train_mlp(training_images, training_labels, test_images, test_labels, names, activation, alpha, hiddenlayers,
              learningrate, solver):
    clf = MLPClassifier(solver=solver, alpha=alpha, hidden_layer_sizes=hiddenlayers, activation=activation,
                        learning_rate=learningrate, random_state=1)

    clf.fit(training_images, training_labels)

    an_score = clf.score(test_images, test_labels)

    ans = clf.predict(test_images)

    ans[ans == 0.0] = -1
    # prints out the name and prediction for the entire test data
    # This is then copied into excel in order to produce the csv
    # The function then outputs the accuracy score
    for i in range(0, len(names)):
        print(str(names[i]), '.png ,', str(ans[i]))

    return ans, an_score


def train_decision_tree(training_images, training_labels, test_images, test_labels):
    clf = tree.DecisionTreeClassifier()

    clf.fit(training_images, training_labels)

    ans = clf.score(test_images, test_labels)

    return ans


# Function to get the optimal parameters for our SVM model
# uses gridsearch and parameters decided and hard coded

def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    param_grid = {'kernel': kernels, 'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    return grid_search.best_params_


def mlp_param_selection(X, y, nfolds):
    parameter_space = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,), (100, 100, 100)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'adam', 'sgd'],
        'alpha': [1, 0.1, 0.01, 0.001, 0.0001],
        'learning_rate': ['constant', 'adaptive', 'invscaling'],
    }
    mlp = MLPClassifier()
    grid_search = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=nfolds)
    grid_search.fit(X, y)

    return grid_search.best_params_


# Function to plot the SVM learning curve
def plot_learningcurve_SVM(a, b):
    title = "Learning Curves (SVM)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = svm.SVC(gamma=0.001, C=0.01, kernel='linear')
    plot_learning_curve(estimator, title, a, b, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    plt.show()


# Runs theSVM and MLP models on all binary classifications
# Only used when initially testing the code and models
def runeverybinary():
    names = ['eyeglasses', 'smiling', 'young', 'human']

    for i in range(2, 6):
        a, b, c, d, e = getdata(i)
        print(names[i - 2])
        print('SVM accuracy : ')
        print(train_svm1(a, b, c, d))
        print('MLP accuracy : ')
        print(train_mlp1(a, b, c, d))

    return 0


def main():
    runeverybinary()


if __name__ == '__main__':
    main()