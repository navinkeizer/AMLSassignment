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

# This function constructs the CNN model using keras layers
def build(num_classes):
    fashion_model = Sequential()
    #INPUT SIZE OF OUR IMAGES IS (256, 256, 1)
    fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(256, 256, 1)))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D((2, 2), padding='same'))
    # ADDIND DROPOUT IN THE MODEL
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

    # Provides a summary od the model we use
    fashion_model.summary()
    # Compiles the model
    fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

    return fashion_model