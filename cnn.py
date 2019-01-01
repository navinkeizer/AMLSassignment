# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import keras
from keras.models import Input,Model
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

def build(num_classes):
    # # Initialising the CNN
    # classifier = Sequential()
    # #
    # # # Step 1 - Convolution
    # classifier.add(Conv2D(32, (3, 3), input_shape = (256,256,1),activation = 'relu'))
    # #
    # # # Step 2 - Pooling
    # # classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # #
    # # # Adding a second convolutional layer
    # # classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    # # classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # #
    # # # Step 3 - Flattening
    # # classifier.add(Flatten())
    # #
    # # # Step 4 - Full connection
    # # classifier.add(Dense(units = 128, activation = 'relu'))
    # # classifier.add(Dense(units = 1, activation = 'sigmoid'))
    # #
    # # # Compiling the CNN
    # # classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # #
    # # # print(classifier.summary())
    #
    #
    #
    #
    #
    # # model = Sequential()
    # # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32,3)))
    # classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # classifier.add(BatchNormalization())
    # classifier.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # classifier.add(BatchNormalization())
    # classifier.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # classifier.add(BatchNormalization())
    # classifier.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
    # classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # classifier.add(BatchNormalization())
    # # classifier.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    # # classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # # classifier.add(BatchNormalization())
    # classifier.add(Dropout(0.2))
    # classifier.add(Flatten())
    # classifier.add(Dense(128, activation='relu'))
    # classifier.add(Dropout(0.3))
    # classifier.add(Dense(1, activation='softmax'))
    #
    # classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(classifier.summary())
    #
    # fashion_model = Sequential()
    # fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(256, 256, 1), padding='same'))
    # fashion_model.add(LeakyReLU(alpha=0.1))
    # fashion_model.add(MaxPooling2D((2, 2), padding='same'))
    # fashion_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
    # fashion_model.add(LeakyReLU(alpha=0.1))
    # fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    # fashion_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
    # fashion_model.add(LeakyReLU(alpha=0.1))
    # fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    # fashion_model.add(Flatten())
    # fashion_model.add(Dense(128, activation='linear'))
    # fashion_model.add(LeakyReLU(alpha=0.1))
    # fashion_model.add(Dense(num_classes, activation='softmax'))
    #
    # fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    # print(fashion_model.summary())
    #

    fashion_model = Sequential()
    fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(256, 256, 1)))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D((2, 2), padding='same'))
    fashion_model.add(Dropout(0.25))
    fashion_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    fashion_model.add(Dropout(0.25))
    fashion_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    fashion_model.add(Dropout(0.4))
    fashion_model.add(Flatten())
    fashion_model.add(Dense(128, activation='linear'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(Dropout(0.3))
    fashion_model.add(Dense(num_classes, activation='softmax'))

    fashion_model.summary()

    fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

    return fashion_model

