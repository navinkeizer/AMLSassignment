import numpy as np
from keras.preprocessing import image
import csv
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cnn as CNN
import matplotlib.pyplot as plt


def get_images():

    picDat = np.ndarray(shape=(5000, 256, 256,1))

    for i in range(1, 5001):
        img1 = image.load_img('dataset/faceDat/' + str(int(i)) + '.png' , target_size=((256, 256)), color_mode = "grayscale")
        x = image.img_to_array(img1)
        x = np.expand_dims(x, axis=0)
        picDat[i - 1] = x

    return picDat


def get_labels():
    labels_file = open('./dataset/attribute_list.csv')
    file_csv = csv.reader(labels_file)
    i = -2
    hair = np.ndarray(shape=(5000,))

    for row in file_csv:
        if (i > -1):
            hair[i] = row[1]
        i = i + 1
    return hair


def get_data(xLabels,yLabels):
    p = 0.8
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
    print(train_Y_one_hot, tr_Y)
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
    classifier.save("fashion_model_dropout.h5py")
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


def main():
    np.set_printoptions(threshold=np.inf)
    a,b,c,d,e,f = get_data(get_images(),get_labels())
    inspect(a,b,c,d,e,f)
    print(contruct_cnn(6,a,b,c,d,e,f))

if __name__ == '__main__':
    main()
