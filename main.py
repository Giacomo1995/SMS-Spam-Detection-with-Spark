#%pip install tfds-nightly

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

from classifier import Classifier

try:
  df = pd.read_pickle('sms_spam_dataframe.pkl')
except Exception as e:
  #tfds.list_builders()

  # Loading data
  ds = tfds.load('huggingface:sms_spam', split='train', shuffle_files=True)
  assert isinstance(ds, tf.data.Dataset)

  # 0 - ham
  # 1 - spam

  df = pd.DataFrame(columns=['x', 'y'])  # Dataframe's header definition

  # Converting ds into a dataframe
  for sample in ds:
    current_sample = {'x': tf.keras.backend.get_value(sample['sms']).decode('utf-8'), 'y': tf.keras.backend.get_value(sample['label'])}
    df = df.append(current_sample, ignore_index=True)

    #print(list(sample.keys()))
    #print(sample['label'])
    #print(sample['sms'])

X_train, X_test, y_train, y_test = train_test_split(df['x'], df['y'], test_size=0.3, random_state=1)  # Splitting data into training set and test set

# Tokenization process
tokenizer = Tokenizer(num_words=None)
tokenizer.fit_on_texts(X_train)

tokenized_X_train = tokenizer.texts_to_sequences(X_train)
tokenized_X_test = tokenizer.texts_to_sequences(X_test)

vocabulary_size = len(tokenizer.word_index) + 1
maxlen = max(len(max(tokenized_X_train, key=len)), len(max(tokenized_X_test, key=len)))

# Padding process
X_train = pad_sequences(tokenized_X_train, padding='post', maxlen=maxlen)
X_train = np.array(X_train, dtype=np.float)
X_test = pad_sequences(tokenized_X_test, padding='post', maxlen=maxlen)
X_test = np.array(X_test, dtype=np.float)

y_train = np.array(y_train, dtype=np.float)
y_test = np.array(y_test, dtype=np.float)


num_epochs = 15
batch_size = 32
validation_split = 0.2

clf = Classifier(vocabulary_size, maxlen)
clf.fit(X_train, y_train, num_epochs, batch_size, validation_split)
clf.evaluate(X_test, y_test)
