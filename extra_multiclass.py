from keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import lab2_landmarks2 as l2
import csv


def countfiles():
    labels_file = open('./dataset/labels.csv')
    file_csv = csv.reader(labels_file)
    count = -2
    for row in file_csv:
        count = count+1
    return count


def get_images(length):
    labels_file = open('./dataset/labels.csv')
    file_csv = csv.reader(labels_file)
    i = -2
    pic_dat = np.ndarray(shape=(length, 256, 256,1))

    for row in file_csv:
        if(i > -1):
            img = image.load_img('dataset/celeba/' + row[0] + '.png', target_size=((256, 256)), color_mode="grayscale")
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            pic_dat[i] = x

        i = i + 1

        pd = pic_dat.reshape(pic_dat.shape[0], pic_dat.shape[1] * pic_dat.shape[2]*pic_dat.shape[3])

    return pd


def get_labels(length):
    labels_file = open('./dataset/labels.csv')
    file_csv = csv.reader(labels_file)
    i = -2
    hair = np.ndarray(shape=(length,))

    for row in file_csv:
        if(i>-1):
            hair[i] = row[1]
        i = i+1
    return hair


def inspect(train_X, train_Y, test_X, test_Y):
    print('Training data shape : ', train_X.shape, train_Y.shape)

    print('Testing data shape : ', test_X.shape, test_Y.shape)

    classes = np.unique(train_Y)

    nClasses = len(classes)

    print('Total number of outputs : ', nClasses)

    print('Output classes : ', classes)


def get_data():
    count = countfiles()
    y = get_labels(count)
    x = get_images(count)

    p = 0.7
    part = int(len(y) * p)

    tr_X = x[:part]

    tr_Y = y[:part]

    te_X = x[part:]

    te_Y = y[part:]

    return tr_X, tr_Y, te_X, te_Y


def main():
    xtr, ytr, xte, yte = get_data()
    inspect(xtr, ytr, xte, yte)
    # print(xtr.shape, ytr.shape, yte.shape, xte.shape)
    print(ytr, yte)


if __name__ == '__main__':
    main()
