import numpy as np
from keras.preprocessing import image
import csv
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cnn as CNN
# import matplotlib.pyplot as plt
from keras.models import load_model
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


def get_images():

    picDat = np.ndarray(shape=(5000, 256, 256,1))

    for i in range(1, 5001):
        img1 = image.load_img('dataset/faceDat/' + str(int(i)) + '.png' , target_size=((256, 256)), color_mode = "grayscale")
        x = image.img_to_array(img1)
        x = np.expand_dims(x, axis=0)
        picDat[i - 1] = x
        pd = picDat.reshape(picDat.shape[0], picDat.shape[1] * picDat.shape[2] * picDat.shape[3])
    return pd


def get_labels():
    labels_file = open('./dataset/attribute_list.csv')
    file_csv = csv.reader(labels_file)
    i = -2
    hair = np.ndarray(shape=(5000,))

    for row in file_csv:
        if (i > -1):
            hair[i] = int(row[1]) + 1
        i = i + 1
    return hair


def get_data(xLabels,yLabels):
    p = 0.75
    part = int(len(yLabels) * p)

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
    #print(train_Y_one_hot, tr_Y)
    train_X, valid_X, train_label, valid_label = train_test_split(tr_X, train_Y_one_hot, test_size=0.2,random_state=13)

    return train_X, train_label, te_X, test_Y_one_hot, valid_X, valid_label


def inspect(train_X, train_Y, test_X, test_Y, vX, vY):
    print('Training data shape : ', train_X.shape, train_Y.shape)

    print('Testing data shape : ', test_X.shape, test_Y.shape)

    print('Validation data shape : ', vX.shape, vY.shape)

    classes = np.unique(train_Y)

    nClasses = len(classes)

    print('Total number of outputs : ', nClasses)

    print('Output classes : ', classes)


def contruct_cnn(n, xTrain, yTrain, xTest, yTest, valid_X, valid_label):  #check data
    batch_size = 200
    epochs = 18
    num_classes = n

    classifier = CNN.build(num_classes)

    tdr = classifier.fit(xTrain, yTrain, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_X, valid_label))
    classifier.save("cnnmodel1.h5py")
    loss, acc = classifier.evaluate(xTest, yTest, verbose=0)
    # print(acc * 100)
    plot_cnn(tdr)
    return acc, loss


def plot_cnn(fashion_train_dropout):
    accuracy = fashion_train_dropout.history['acc']
    val_accuracy = fashion_train_dropout.history['val_acc']
    loss = fashion_train_dropout.history['loss']
    val_loss = fashion_train_dropout.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def get_statistacs(modelname, xTrain, yTrain, xTest, yTest, valid_X, valid_label,num_classes, ):
    model = load_model(modelname)
    loss, acc = model.evaluate(xTest, yTest, verbose=0)
    print(acc, loss)
    y_pred_one = model.predict(xTest)
    predicted_classes = np.argmax(np.round(y_pred_one), axis=1)


    confusion_matrix = metrics.confusion_matrix(y_true=np.argmax(np.round(yTest), axis=1), y_pred=predicted_classes)


    target_names = ["Class {}".format(i) for i in range(num_classes)]
    print(classification_report(np.argmax(np.round(yTest), axis=1), predicted_classes, target_names = target_names))

    return confusion_matrix


#######################################


#######################################


#######################################


def train_decision_tree(training_images, training_labels, test_images, test_labels):

    clf = tree.DecisionTreeClassifier()

    clf.fit(training_images, training_labels)

    ans = clf.score(test_images, test_labels)
    predicted_classes = np.argmax(np.round(clf.predict(test_images)), axis=1)
    cm = metrics.confusion_matrix(y_true=np.argmax(np.round(test_labels), axis=1), y_pred=predicted_classes)
    # cm = confusion_matrix(test_labels, clf.predict(test_images))
    return ans, cm


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


#######################################


#######################################


#######################################

def main():
    np.set_printoptions(threshold=np.inf)
    a,b,c,d,e,f = get_data(get_images(),get_labels())
    # inspect(a,b,c,d,e,f)
    print(contruct_cnn(6,a,b,c,d,e,f))
    print(get_statistacs("cnnmodel1.h5py",a,b,c,d,e,f,6))
    # print(train_decision_tree(a,b,c,d))
    # print(train_knn_classifier(a,b,c,d))
    # print(train_naive_bayes(a,b,c,d))
    # print(train_mlp(a,b,c,d))
    # print(mlp_param_selection(c,d,2))
    # print(get_labels())

if __name__ == '__main__':
    main()
