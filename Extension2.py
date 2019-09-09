#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 1: Creating String Indexer pipeline and saving it

Usage:

    $ spark-submit indexer.py

'''


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import explode
from pyspark.mllib.evaluation import RankingMetrics

def main(spark):
    '''

    Parameters
    ----------
    spark : SparkSession object
    '''


    # File names
    test_file = 'hdfs:/user/bm106/pub/project/cf_test.parquet'

    # Reading the parquet files
    test = spark.read.parquet(test_file)
    test.createOrReplaceTempView('test')

    test = test.withColumn("log_count", F.log("count"))
    print('Log done.')

    test.createOrReplaceTempView('test')
    test = spark.sql('SELECT user_id, track_id, log_count AS count FROM test')
    test.createOrReplaceTempView('test')
    print('Log updated.')

    train_sample = spark.read.parquet('hdfs:/user/dev241/extension2_logs.parquet')
    train_sample.createOrReplaceTempView('train_sample')
    print("Training sample loaded")

    idx_pipe = PipelineModel.load('DieterStringIndexer')
    train_idx = idx_pipe.transform(train_sample)
    test_idx = idx_pipe.transform(test)

    print('Pipeline transforms created.')

    #change to best
    rank = 78
    alpha = 14.287069059772636
    reg = 0.41772043857578584

    #model
    model = ALSModel.load("Extension2model")
    print('Model loaded')

    #test ranking metrics
    test_idx = test_idx.select('user_idx','track_idx','count')
    test_users = test_idx.select('user_idx').distinct()
    test_comb = test_idx.groupBy('user_idx').agg(F.collect_set('track_idx').alias('test_labels'))
    track_number = 500
    rec_test = model.recommendForUserSubset(test_users, track_number)
    print('Rec done.')
    #rec_test.write.parquet("rec_test2_2.parquet")
    print('Rec saved woooo!')
    join = test_comb.join(rec_test,test_comb.user_idx == rec_test.user_idx)
    #print('Join done.')
    join = join.toDF('user_idx', 'test_labels','user_idx2','recommendations')
    #j2.write.parquet("ext2join_rl.parquet")
    #print('Join2 ext written! Woohooo :D xD :P ;) :3')
    predictionAndLabels = join.rdd.map(lambda r: ([track.track_idx for track in r.recommendations], r.test_labels))
    print('Map done.')
    metrics = RankingMetrics(predictionAndLabels)
    print('RM done.')

    mavgp = metrics.meanAveragePrecision
    print("Test mean Average Precision : ",mavgp)
    pass

    pass

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('Extension2').getOrCreate()

    # Call our main routine
    main(spark)