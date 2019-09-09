#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 2: Saving hypertuned models

Usage:

    $ spark-submit saving_models.py

'''

# We need sys to get the command line arguments
import sys
import os
import numpy as np
import pandas as pd
# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

# TODO: you may need to add imports here
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml.recommendation import ALS
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F
from pyspark.sql.functions import explode
from pyspark.mllib.evaluation import RankingMetrics
from pyspark import SparkContext

#cllect_list

def main(spark):
    '''

    Parameters
    ----------
    spark : SparkSession object
    '''


    # Reading train and transforming with StringIndexer
    train_file = 'hdfs:/user/dev241/train_sample.parquet'
    val_file = 'hdfs:/user/bm106/pub/project/cf_validation.parquet'

    train_sample = spark.read.parquet(train_file)
    val = spark.read.parquet(val_file)

    idx_pipe = PipelineModel.load('hdfs:/user/dev241/DieterStringIndexer')

    train_idx = idx_pipe.transform(train_sample)
    val_idx = idx_pipe.transform(val)

    val_idx = val_idx.select('user_idx','track_idx','count')
    val_users = val_idx.select('user_idx').distinct()
    val_comb = val_idx.groupBy('user_idx').agg(F.collect_set('track_idx').alias('val_labels'))

    # Hyperparameter values
    results = []
    i = 0

    # Looping through the hyperparameter values - low alpha
    for i in range(50):
        rank = np.random.randint(100)
        alpha = np.random.uniform(0.1,15)
        reg = np.random.uniform(0.1,1)
        als = ALS(rank = rank, alpha = alpha, regParam = reg, userCol="user_idx", itemCol="track_idx", ratingCol="count", coldStartStrategy="drop", implicitPrefs = True)
        model = als.fit(train_idx)
        model.save('model_random_search'+str(rank)+'_'+str(alpha)+'_'+str(reg))
        track_number = 500
        rec_val = model.recommendForUserSubset(val_users, track_number)
        join = val_comb.join(rec_val,val_comb.user_idx == rec_val.user_idx)
        predictionAndLabels = join.rdd.map(lambda r: ([track.track_idx for track in r.recommendations], r.val_labels))
        metrics = RankingMetrics(predictionAndLabels)
        mavgp = metrics.meanAveragePrecision
        results.append((rank,alpha,reg,mavgp))
        print("Rank : ",rank,"Alpha : ",alpha,"Reg : ",reg,"MAP : ",mavgp)
    print('First Validation completed.')
    sc.parallelize(results).saveAsTextFile("MAP_random_search_high.txt")
    Print('MAP_random_search_high.txt saved')

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('Random_search_val').getOrCreate()

    # Call our main routine
    main(spark)
