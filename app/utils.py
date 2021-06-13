# Imports
#import numpy as np
#import pandas as pd
#import tensorflow_datasets as tfds


def load_dataset(spark, file_name="dataset.csv"):
  """
  Reads the SMS spam dataset and splits it into training set (0.8) and test set (0.2) by a stratified sampling procedure.
  It returns both training set and test set as Spark Dataframes.
  """

  # Load CSV
  #dataset = spark.read.format("csv").option("header", True).option("multiLine", True).option("escape","\"").load("dataset.csv")
  dataset = spark.read.format("csv").option("header", True).option("multiLine", True).option("escape","\"").load("hdfs://s01:9000/dataset.csv")
  dataset = dataset.withColumnRenamed('x', 'text')
  dataset = dataset.withColumnRenamed('y', 'label')

  dataset = dataset.withColumn('label', dataset['label'].cast('int'))

  # Stratified Sampling: .8 training set / .2 test set
  # Split dataframes between legitimates and spams
  legitimates = dataset.filter(dataset['label'] == 0)
  spams = dataset.filter(dataset['label'] == 1)

  # Split datasets into training set and test set
  train_legit, test_legit = legitimates.randomSplit([0.8,0.2], seed=0)
  train_spam, test_spam = spams.randomSplit([0.8,0.2], seed=0)

  # Merge datasets
  training_set = train_legit.union(train_spam)
  test_set = test_legit.union(test_spam)

  return training_set, test_set


'''
def load_sms_spam_data():
    #tfds.list_builders()

    # Loading data
    ds = tfds.load('huggingface:sms_spam', split='train', shuffle_files=True)
    assert isinstance(ds, tf.data.Dataset)

    # 0 - legitimate label
    # 1 - spam label

    df = pd.DataFrame(columns=['x', 'y'])  # Dataframe's header definition

    # Converting ds into a dataframe
    for sample in ds:
        current_sample = {'x': tf.keras.backend.get_value(sample['sms']).decode('utf-8'), 'y': tf.keras.backend.get_value(sample['label'])}
        df = df.append(current_sample, ignore_index=True)

    #print(list(sample.keys()))
    #print(sample['label'])
    #print(sample['sms'])

    df.to_csv("dataset.csv", index=False, header=False)  # Save CSV

    return df
'''
