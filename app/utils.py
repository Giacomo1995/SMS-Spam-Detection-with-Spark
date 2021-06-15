# Imports
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import tensorflow_datasets as tfds
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import subprocess


def load_dataset(spark, file_name="dataset.csv", from_s3=True):
  """
  Reads the SMS spam dataset from Amazon S3, generate hdfs, and splits it into training set (0.8) and test set (0.2) by a stratified sampling procedure.
  It returns both training set and test set as Spark Dataframes.
  """
  
  if from_s3:
    run_cmd(['rm', file_name]) #delete the local file
    BUCKET_NAME = "ssdsdataset"
    download_from_s3(BUCKET_NAME, file_name)

  #make hdfs of dataset
  run_cmd(['hdfs', 'dfs', '-put', file_name, "/"+file_name])

  # Load CSV
  #dataset = spark.read.format("csv").option("header", True).option("multiLine", True).option("escape","\"").load("dataset.csv")
  dataset = spark.read.format("csv").option("header", True).option("multiLine", True).option("escape","\"").load("hdfs://s01:9000/"+file_name)
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

def download_from_s3(bucket_name, file_name):
  """
  Downloads a public file from a bucket of Amazon s3
  :param bucket_name: bucket from which read the file
  :param file_name: file to read
  :return: void
  """

  s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
  s3.download_file(bucket_name, file_name, file_name)  # get object and file (key) from bucket


def run_cmd(cmd):
  """
  Executes a bash command
  :param cmd: command splitted in a list
  :return: void
  """
  try:
    print('System command: {0}'.format(' '.join(cmd)))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    s_output, s_err = proc.communicate()
    s_return = proc.returncode
  except Exception as e:
    print(e)

    
'''
def plot():
    numFeaturesList = np.array([1, 2.5, 5, 7.5, 10])
    regParamList = np.array([0.1, 0.01, 0.001])
    elasticNetParamList = np.array([0, 0.25, 0.5, 0.75, 1])

    meshgrid = np.meshgrid(numFeaturesList, regParamList, elasticNetParamList)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter3D(meshgrid[0], meshgrid[1], meshgrid[2])
    fig.savefig('gridsearch.png', dpi=300)
    plt.show()
'''


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
