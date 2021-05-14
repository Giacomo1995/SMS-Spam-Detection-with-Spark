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
from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten, Dropout

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

# Network architecture
model = Sequential()
model.add(Embedding(vocabulary_size, 32, input_length=maxlen))
model.add(Flatten())
#model.add(Dense(32, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

num_epochs = 15
batch_size = 32
validation_split = 0.2
history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=validation_split)

model.evaluate(X_test, y_test)

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
