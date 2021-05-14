# Imports
import matplotlib.pyplot as plt
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten, Dropout

from utils import *

class Classifier:

    def __init__(self, vocabulary_size, input_length):
        # Network architecture
        model = Sequential()
        model.add(Embedding(vocabulary_size, 32, input_length=input_length))
        model.add(Flatten())
        #model.add(Dense(32, activation='relu'))
        #model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        # Compiling the model
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc', f1])
        model.summary()

        self.model = model


    def fit(self, X_train, y_train, num_epochs = 15, batch_size = 32, validation_split = 0.2):
        history = self.model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=validation_split)

        # Plotting loss and accuracy

        epochs = range(num_epochs)

        # Loss plot
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Accuracy plot
        acc = history.history['acc']
        val_acc = history.history['val_acc']

        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and Validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


    def evaluate(self, X_test, y_test):
        self.model.evaluate(X_test, y_test)
