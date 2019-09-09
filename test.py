#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 1: Creating String Indexer pipeline and saving it

Usage:

    $ spark-submit indexer.py

'''


import sys
import os
import numpy as np
import pandas as pd

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

# TODO: you may need to add imports here
from pyspark.ml.feature import StringIndexer
from pyspark.ml import PipelineModel
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F
from pyspark.sql.functions import explode
from pyspark.mllib.evaluation import RankingMetrics
from pyspark import SparkContext

def main(spark):
    '''

    Parameters
    ----------
    spark : SparkSession object
    '''

    # File names
    test_file = 'hdfs:/user/bm106/pub/project/cf_test.parquet'
    train_sample_file = 'hdfs:/user/ah3243/extension1_count_greater_1.parquet'

    # Reading the parquet files
    test = spark.read.parquet(test_file)
    train_sample = spark.read.parquet(train_sample_file)

    # StringIndexer 
    print("String Indexer entered")
    StringIndexer = PipelineModel.load('hdfs:/user/dev241/DieterStringIndexer')
    test_idx = StringIndexer.transform(test)
    train_idx = StringIndexer.transform(train_sample)
    print("String Indexer done")
	
    #change to best
    rank = 78
    alpha = 14.287069059772636
    reg = 0.41772043857578584

    #model
    als = ALS(rank = rank, alpha = alpha, regParam = reg, userCol="user_idx", itemCol="track_idx", ratingCol="count", coldStartStrategy="drop", implicitPrefs = True)
    model = als.fit(train_idx)
    print("Model fit for test done")
    model.save("Test_Model")
    print("Model save for test done")

    #test ranking metrics
    test_idx = test_idx.select('user_idx','track_idx','count')
    test_users = test_idx.select('user_idx').distinct()
    test_comb = test_idx.groupBy('user_idx').agg(F.collect_set('track_idx').alias('test_labels'))
    track_number = 500
    rec_test = model.recommendForUserSubset(test_users, track_number)
    join = test_comb.join(rec_test,test_comb.user_idx == rec_test.user_idx)
    predictionAndLabels = join.rdd.map(lambda r: ([track.track_idx for track in r.recommendations], r.test_labels))
    metrics = RankingMetrics(predictionAndLabels)
    mavgp = metrics.meanAveragePrecision
    print("Test mean Average Precision : ",mavgp)
    pass

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('extension1').getOrCreate()

    # Call our main routine
    main(spark)
