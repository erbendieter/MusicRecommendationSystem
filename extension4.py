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
from pyspark.sql.functions import *
from pyspark.sql.window import Window

def main(spark):
    '''

    Parameters
    ----------
    spark : SparkSession object
    '''
    test_file = 'hdfs:/user/bm106/pub/project/cf_test.parquet'
    test = spark.read.parquet(test_file)
    test.createOrReplaceTempView('test')

    w = Window.partitionBy("user_id")

    def ratio_count(c, w):
        return (col(c) / count(c).over(w))


    test = test.select("user_id", "track_id", ratio_count("count", w).alias("count"))
    test.createOrReplaceTempView('test')
    print("Ratio scores done")

    train_sample = spark.read.parquet('hdfs:/user/dev241/extension4_ratio.parquet')
    train_sample.createOrReplaceTempView('train_sample')
    print("Training sample ext4 loaded")

    StringIndexer = PipelineModel.load('hdfs:/user/dev241/DieterStringIndexer')
    test_idx = StringIndexer.transform(test)
    train_idx = StringIndexer.transform(train_sample)

    #change to best
    rank = 78 
    alpha = 14.287069059772636
    reg = 0.41772043857578584

    model = ALSModel.load("Extension4_ratio")
    print('Model loaded')

    #test ranking metrics
    test_idx = test_idx.select('user_idx','track_idx','count')
    test_users = test_idx.select('user_idx').distinct()
    test_comb = test_idx.groupBy('user_idx').agg(F.collect_set('track_idx').alias('test_labels'))
    track_number = 500
    rec_test = spark.read.parquet('hdfs:/user/dev241/rec_test4.parquet')
    print('Rec test loaded.')
    join = test_comb.join(rec_test,test_comb.user_idx == rec_test.user_idx)
    print('Join done.')
    j4 = join.toDF('user_idx', 'test_labels','user_idx2','recommendations')
    j4.write.parquet("ext4join")
    print('j4 parquet written')
    predictionAndLabels = join.rdd.map(lambda r: ([track.track_idx for track in r.recommendations], r.test_labels))
    print('Map done.')
    metrics = RankingMetrics(predictionAndLabels)
    print('RM done.')
    mavgp = metrics.meanAveragePrecision
    print("Test mean Average Precision : ",mavgp)
    pass

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('extension4').getOrCreate()

    # Call our main routine
    main(spark)