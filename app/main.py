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


def main():
    """
    Starts a SparkSession and perform the classification task to detect spam in SMS messages.
    It prints Accuracy and F-Measure of the models.
    """

    findspark.init()

    spark = SparkSession.builder.master("spark://s01:7077")\
                                .appName("SmsSpamDetector")\
                                .config("spark.worker.cleanup.enabled", True)\
                                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")\
                                .config("spark.kryo.registrationRequired", "false")\
                                .getOrCreate()


    spark.sparkContext.addPyFile('SSDS.zip')

    training_set, test_set = load_dataset(spark)

    # Initialize Classifiers
    lr_clf = Classifier(classifier="lr", k=3)
    mlp_clf = Classifier(classifier="mlp", k=3)
    nb_clf = Classifier(classifier="nb", k=3)

    # Run cross-validation and choose the best set of parameters for each model
    lr_clf.fit(training_set)
    mlp_clf.fit(training_set)
    nb_clf.fit(training_set)

    # Make predictions on test set on the best models found
    lr_result = lr_clf.evaluate(test_set)
    mlp_result = mlp_clf.evaluate(test_set)
    nb_result = nb_clf.evaluate(test_set)
    

    # Print the results
    print("LOGISTIC REGRESSION")
    params = lr_cvmodel.bestModel.stages[2]
    print("Number of Features: ", params.numFeatures)
    print(params.explainParam("regParam"))
    print(params.explainParam("elasticNetParam"))

    lr_prediction_and_labels = lr_result.select("prediction", "label")
    lr_evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    print("\nAccuracy: " + str(lr_evaluator.evaluate(lr_prediction_and_labels)))

    lr_evaluator = MulticlassClassificationEvaluator(metricName="fMeasureByLabel", metricLabel=1)
    print("F-Measure: " + str(lr_evaluator.evaluate(lr_prediction_and_labels)))

    print("\n--------------------------------------------------------------------------------------------")

    print("\nMLP")
    params = mlp_cvmodel.bestModel.stages[2]
    print("Number of Features: ", params.numFeatures)
    print(params.explainParam("blockSize"))
    print(params.explainParam("layers"))

    mlp_prediction_and_labels = mlp_result.select("prediction", "label")
    mlp_evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    print("\nAccuracy: " + str(mlp_evaluator.evaluate(mlp_prediction_and_labels)))

    mlp_evaluator = MulticlassClassificationEvaluator(metricName="fMeasureByLabel", metricLabel=1)
    print("F-Measure: " + str(mlp_evaluator.evaluate(mlp_prediction_and_labels)))


    print("\n--------------------------------------------------------------------------------------------")

    print("\nNAIVE BAYES")
    params = nb_cvmodel.bestModel.stages[2]
    print("Number of Features: ", params.numFeatures)
    print(params.explainParam("smoothing"))

    nb_prediction_and_labels = nb_result.select("prediction", "label")
    nb_evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    print("\nAccuracy: " + str(nb_evaluator.evaluate(nb_prediction_and_labels)))

    nb_evaluator = MulticlassClassificationEvaluator(metricName="fMeasureByLabel", metricLabel=1)
    print("F-Measure: " + str(nb_evaluator.evaluate(nb_prediction_and_labels)))


if __name__ == "__main__":
    main()
