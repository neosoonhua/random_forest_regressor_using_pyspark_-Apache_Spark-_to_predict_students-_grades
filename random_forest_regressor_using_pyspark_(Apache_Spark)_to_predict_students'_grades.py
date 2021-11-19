# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 22:48:59 2021

@source: https://github.com/apache/spark/blob/master/examples/src/main/python/ml/random_forest_regressor_example.py
"""

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# $example on$
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
# $example off$
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("RandomForestRegressorToPredictStudentsGrades")\
        .getOrCreate()

    # $example on$
    # Load and parse the data file, converting it to a DataFrame.
    # help(spark.read.csv)
    
    trainFilename = "train - Ace for A-level replaced with 1.csv"
    testFilename = "test - Ace for A-level replaced with 1.csv"

    feature_list = ['JC1 CT', 'JC1 T3 LT', 'Promo', 'JC2 T1 LT', 'JC2 CT', 'Prelim']
    assembler = VectorAssembler(inputCols=feature_list, outputCol="features")
    
    trainingData = spark.read.option("inferSchema", True).csv(trainFilename, header=True)
    
    testData = spark.read.option("inferSchema", True).csv(testFilename, header=True)

    # Train a RandomForest model.
    rf = RandomForestRegressor(featuresCol="features", labelCol="A Level Grade")

    # Chain indexer and forest in a Pipeline
    pipeline = Pipeline(stages=[assembler, rf])

    # Train model.  This also runs the indexer.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("prediction", "A Level Grade", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(labelCol="A Level Grade", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    rfModel = model.stages[1]
    print(rfModel)  # summary only
    # $example off$

    spark.stop()
