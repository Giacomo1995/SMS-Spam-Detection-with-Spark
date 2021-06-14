# Imports
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, MultilayerPerceptronClassifier, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


class Classifier:
    """
    Classifier selected through a Grid Search and a k-fold cross-validation.
    """
    
    def __init__(self, classifier, k=3):
        # Configure an ML pipeline, which consists of tree stages: tokenizer, hashingTF, and clf
        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")

        # Logistic Regression
        if classifier == 'lr':
            clf = LogisticRegression(maxIter=1)
            pipeline = Pipeline(stages=[tokenizer, hashingTF, clf])

            # Set parameters for grid search
            numFeaturesList = [1000, 2500, 5000, 7500, 10000]
            regParamList = [0.1, 0.01, 0.001]
            elasticNetParamList = [0, 0.25, 0.5, 0.75, 1]

            paramGrid = ParamGridBuilder() \
                .addGrid(hashingTF.numFeatures, numFeaturesList) \
                .addGrid(clf.regParam, regParamList) \
                .addGrid(clf.elasticNetParam, elasticNetParamList) \
                .build()

        # Multi-Layer Perceptron
        if classifier == 'mlp':
            clf = MultilayerPerceptronClassifier(maxIter=10)
            pipeline = Pipeline(stages=[tokenizer, hashingTF, clf])

            # Set parameters for grid search
            numFeaturesList = [1000]
            layerList = [[1000, 100, 10, 2], [1000, 8, 2], [1000, 64, 8, 2]]
            blockSizeList = [32, 64, 128]

            paramGrid = ParamGridBuilder() \
                .addGrid(hashingTF.numFeatures, numFeaturesList) \
                .addGrid(clf.layers, layerList) \
                .addGrid(clf.blockSize, blockSizeList) \
                .build()

        # Naive Bayes
        if classifier == 'nb':
            clf = NaiveBayes()
            pipeline = Pipeline(stages=[tokenizer, hashingTF, clf])

            # Set parameters for grid search
            numFeaturesList = [1000, 2500, 5000, 7500, 10000]
            smoothings = [0, 0.25, 0.5, 0.75, 1]

            paramGrid = ParamGridBuilder() \
                .addGrid(hashingTF.numFeatures, numFeaturesList) \
                .addGrid(clf.smoothing, smoothings) \
                .build()


        num_models = len(paramGrid)

        crossval = CrossValidator(estimator=pipeline,
            estimatorParamMaps=paramGrid,
            evaluator=MulticlassClassificationEvaluator(),
            numFolds=k,
            parallelism=num_models)

        self.clf = clf
        self.crossval = crossval


    def fit(self, training_set):
        # Run cross-validation, and choose the best set of parameters
        self.cvModel = self.crossval.fit(training_set)


    def evaluate(self, test_set):
        # Make predictions on test set. cvModel uses the best model found
        return self.cvModel.transform(test_set)
