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
    # train_file = 'hdfs:/user/bm106/pub/project/cf_train.parquet'
    test_file = 'hdfs:/user/bm106/pub/project/cf_test.parquet'
    # val_file = 'hdfs:/user/bm106/pub/project/cf_validation.parquet'

    # Reading the parquet files
    # train = spark.read.parquet(train_file)
    test = spark.read.parquet(test_file)
    # val = spark.read.parquet(val_file)

    #extension part
    # train=train.filter(train['count']>1)
    test=test.filter(test['count']>1)
    # val=val.filter(val['count']>1)
    print("Filtering done")

    # Creating the train sample
    # All validation and test users from train, and 10% of the rest of the train
    # train.createOrReplaceTempView('train')
    # val.createOrReplaceTempView('validation')
    test.createOrReplaceTempView('test')

    # uniq_users = spark.sql('SELECT DISTINCT user_id FROM (SELECT user_id from test UNION ALL SELECT user_id from validation)')
    # uniq_users.createOrReplaceTempView('uniq_users')

    # in_join_users = spark.sql('SELECT train.user_id ,count, track_id, __index_level_0__  FROM train INNER JOIN uniq_users ON train.user_id = uniq_users.user_id')
    # in_join_users.createOrReplaceTempView('in_join_users')

    # diff_set = spark.sql('SELECT * FROM train MINUS SELECT train.user_id ,count, track_id, __index_level_0__  FROM train INNER JOIN uniq_users ON train.user_id = uniq_users.user_id')

    # diff_set_sample = diff_set.sample(.10)
    # train_sample = diff_set_sample.union(in_join_users)
    # train_sample.write.parquet('hdfs:/user/ah3243/extension1_count_greater_1.parquet')
    train_sample = spark.read.parquet('hdfs:/user/ah3243/extension1_count_greater_1.parquet')
    print("Training sample for Ext1 done")

    StringIndexer = PipelineModel.load('hdfs:/user/dev241/DieterStringIndexer')
    test_idx = StringIndexer.transform(test)
    train_idx = StringIndexer.transform(train_sample)

    #change to best
    rank = 78
    alpha = 14.287069059772636
    reg = 0.41772043857578584

    #model
    als = ALS(rank = rank, alpha = alpha, regParam = reg, userCol="user_idx", itemCol="track_idx", ratingCol="count", coldStartStrategy="drop", implicitPrefs = True)
    model = als.fit(train_idx)
    print("Model fit for Ext1 done")
    model.save("Extension1(Count_1)")
    print("Model save for Ext1 done")

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

    pass

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('extension1').getOrCreate()

    # Call our main routine
    main(spark)