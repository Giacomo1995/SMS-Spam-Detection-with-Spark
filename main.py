# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import sklearn as sk
from sklearn.model_selection import train_test_split

import tensorflow.keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from spark_tensorflow_distributor import MirroredStrategyRunner

from utils import *
from training import *


'''
spark = SparkSession.builder.master("<Master>").appName("sms_spam_detector_with_spark")\
    .config("spark.driver.memory" , "1g")\
    .config("spark.executor.memory" , "1g").enableHiveSupport().getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("Error")
'''

# Training
MSR = MirroredStrategyRunner(num_slots=1, local_mode=True, use_gpu=False)
model, history = MSR.run(train)

# Testing
_, _, X_test, y_test = load_sms_spam_data()
print(model.evaluate(X_test, y_test))


num_epochs = 15
# Plot loss
loss = history.history['loss']
#val_loss = history.history['val_loss']

plt.plot(range(num_epochs), loss, 'b', label='Training loss')
#plt.plot(15, val_loss, 'r', label='Validation loss')
plt.title('Loss plot')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot accuracy
loss = history.history['acc']
#val_acc = history.history['val_acc']
plt.plot(range(num_epochs), loss, 'b', label='Training accuracy')
#plt.plot(15, val_loss, 'r', label='Validation loss')
plt.title('Accuracy plot')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
