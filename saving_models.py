#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 2: Saving hypertuned models

Usage:

    $ spark-submit saving_models.py

'''


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

from pyspark.ml.feature import StringIndexer
from pyspark.ml import PipelineModel
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

def main(spark):
    '''

    Parameters
    ----------
    spark : SparkSession object
    '''


    # Reading train and transforming with StringIndexer
    train_file = 'hdfs:/user/dev241/train_sample.parquet'

    train_sample = spark.read.parquet(train_file)

    idx_pipe = PipelineModel.load('DieterStringIndexer')

    train_idx = idx_pipe.transform(train_sample)

    # Hyperparameter values
    rank_list = [10, 25, 50, 75, 100]
    alpha_list = [0.01, 0.05, 0.1, 0.5, 1.0]
    reg_list = [0.01, 0.05, 0.1, 0.5, 1.0]
    results = []

    # Looping through the hyperparameter values
    for rank in rank_list:
        for alpha in alpha_list:
            for reg in reg_list:
                als = ALS(rank = rank, alpha = alpha, regParam = reg, userCol="user_idx", itemCol="track_idx", ratingCol="count", coldStartStrategy="drop", implicitPrefs = True)
                model = als.fit(train_idx)
                model.save('model_'+str(rank)+'_'+str(alpha)+'_'+str(reg))
                print('model_'+str(rank)+'_'+str(alpha)+'_'+str(reg)+' saved by Dieter ;)')

    pass

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('saving_models').getOrCreate()

    # Call our main routine
    main(spark)