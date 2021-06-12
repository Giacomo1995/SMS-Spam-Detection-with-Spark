# Imports
import numpy as np
import pandas as pd
import os

import findspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from utils import *
from classifier import *


findspark.init()

spark = SparkSession.builder.master("spark://s01:7077")\
                            .appName("SmsSpamDetector")\
                            .config("spark.worker.cleanup.enabled", True)\
                            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")\
                            .config("spark.kryo.registrationRequired", "false")\
                            .getOrCreate()


spark.sparkContext.addPyFile('SSDS.zip')

training_set, test_set = load_dataset(spark)

# Initialize Classifier
clf = Classifier()

# Run cross-validation and choose the best set of parameters
clf.fit(training_set)

# Make predictions on test set on the best model found
result = clf.evaluate(test_set)

# Best model parameters
params = clf.cvModel.bestModel.stages[2]
print("MODEL PARAMETERS:")
print("Number of Features: ", params.numFeatures)
print(params.explainParam("regParam"))
print(params.explainParam("elasticNetParam"))

# Print Accuracy and F-Measure evaluation metrics
predictionAndLabels = result.select("prediction", "label")

evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("\nAccuracy: " + str(evaluator.evaluate(predictionAndLabels)))

evaluator = MulticlassClassificationEvaluator(metricName="fMeasureByLabel", metricLabel=1)
print("F-Measure: " + str(evaluator.evaluate(predictionAndLabels)))
