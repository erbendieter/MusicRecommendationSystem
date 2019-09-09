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
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

def main(spark):
    '''

    Parameters
    ----------
    spark : SparkSession object
    '''

    # File names
    train_file = 'hdfs:/user/bm106/pub/project/cf_train.parquet'
    val_file = 'hdfs:/user/bm106/pub/project/cf_validation.parquet'
    test_file = 'hdfs:/user/bm106/pub/project/cf_test.parquet'
    meta_file = 'hdfs:/user/bm106/pub/project/metadata.parquet'

    # Reading the parquet files
    train = spark.read.parquet(train_file)
    validation = spark.read.parquet(val_file)
    test = spark.read.parquet(test_file)
    meta = spark.read.parquet(meta_file)

    # Creating the train sample
    # All validation and test users from train, and 10% of the rest of the train
    train.createOrReplaceTempView('train')
    validation.createOrReplaceTempView('validation')
    test.createOrReplaceTempView('test')

    uniq_users = spark.sql('SELECT DISTINCT user_id FROM (SELECT user_id from test UNION ALL SELECT user_id from validation)')
    uniq_users.createOrReplaceTempView('uniq_users')

    in_join_users = spark.sql('SELECT train.user_id ,count, track_id, __index_level_0__  FROM train INNER JOIN uniq_users ON train.user_id = uniq_users.user_id')
    in_join_users.createOrReplaceTempView('in_join_users')

    diff_set = spark.sql('SELECT * FROM train MINUS SELECT train.user_id ,count, track_id, __index_level_0__  FROM train INNER JOIN uniq_users ON train.user_id = uniq_users.user_id')

    diff_set_sample = diff_set.sample(.1)
    train_sample = diff_set_sample.union(in_join_users)
    train_sample.write.parquet('hdfs:/user/dev241/train_sample.parquet')

    # Creating the sample to fit the indexer on
    meta.createOrReplaceTempView('meta')
    train_sample.createOrReplaceTempView('train_sample')

    uniq_tracks = spark.sql('SELECT DISTINCT track_id FROM meta')
    uniq_tracks.createOrReplaceTempView('uniq_tracks')

    all_users_tracks = spark.sql('SELECT user_id, uniq_tracks.track_id FROM train_sample FULL OUTER JOIN uniq_tracks ON train_sample.track_id = uniq_tracks.track_id')
    all_users_tracks.createOrReplaceTempView('all_users_tracks')

    indexer_sample = spark.sql('SELECT DISTINCT user_id, track_id FROM all_users_tracks')

    # String Indexer Pipeline
    indexer = StringIndexer(inputCol="user_id", outputCol="user_idx", handleInvalid='skip')
    indexer2 = StringIndexer(inputCol="track_id", outputCol="track_idx", handleInvalid='skip')
    pipeline = Pipeline(stages=[indexer,indexer2])

    # Fitting pipeline on indexer sample
    idx_pipe = pipeline.fit(indexer_sample.repartition(5000,"user_id"))
    idx_pipe.save("DieterStringIndexer")
    
    print('Pipeline fit.')
    
    pass

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('indexer').getOrCreate()

    # Call our main routine
    main(spark)