import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from keras.layers import Embedding, Flatten, Dense
from keras.models import Sequential
from spark_tensorflow_distributor import MirroredStrategyRunner
from pyspark.sql import SparkSession

from classifier import Classifier
from utils import *


def train():
    import uuid
    BUFFER_SIZE = 100
    BATCH_SIZE = 32
    VOCABULARY_SIZE = 7380
    MAXLEN = 180


    def make_dataset():
        X_train, y_train, X_test, y_test = load_sms_spam_data()

        training_set = tf.data.Dataset.from_tensor_slices((
            tf.cast(X_train[..., tf.newaxis], tf.float32),
            tf.cast(y_train, tf.int64))
        )

        test_set = tf.data.Dataset.from_tensor_slices((
            tf.cast(X_test[..., tf.newaxis], tf.float32),
            tf.cast(y_test, tf.int64))
        )

        training_set = training_set.repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        test_set = test_set.repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

        return training_set, test_set


    def build_and_compile_model():
        clf = Classifier(VOCABULARY_SIZE, MAXLEN)

        return clf.model


    training_set, test_set = make_dataset()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    training_set = training_set.with_options(options)
    test_set = training_set.with_options(options)
    multi_worker_model = build_and_compile_model()

    # Training
    num_epochs = 15
    history = multi_worker_model.fit(x=training_set, epochs=num_epochs, steps_per_epoch=100, validation_steps=100, validation_data=test_set)

    # Testing
    #result = multi_worker_model.evaluate(test_set)
    #print(result)

    return multi_worker_model, history
