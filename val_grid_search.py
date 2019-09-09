#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 3: Validation and testing

Usage:

    $ spark-submit validaton.py

'''


# We need sys to get the command line arguments
import sys
import os

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

def main(spark):
    '''Main routine for unsupervised training

    Parameters
    ----------
    spark : SparkSession object
    '''

    ###
    # TODO: YOUR CODE GOES HERE
    ###

    print('Running...')
    # File names
    val_file = 'hdfs:/user/bm106/pub/project/cf_validation.parquet'

    # Reading relevant files
    val = spark.read.parquet(val_file)

    # Transforming all tables using String Indexer
    StringIndexer = PipelineModel.load('hdfs:/user/dev241/DieterStringIndexer')
    val_idx = StringIndexer.transform(val)

    # Value for HyperParameter Tuning #check values 
    rank_list = [10,25,50,75,100]
    alpha_list = [0.01,0.05,0.1,0.5,1.0]
    reg_list = [.01,.05,0.1,0.5,1.0]
    results = []

    val_idx = val_idx.select('user_idx','track_idx','count')
    val_users = val_idx.select('user_idx').distinct()
    val_comb = val_idx.groupBy('user_idx').agg(F.collect_set('track_idx').alias('val_labels'))

    print('Entering loop...')
    for rank in rank_list:
        for alpha in alpha_list:
            for reg in reg_list:
                model = ALSModel.load('hdfs:/user/dev241/model_'+str(rank)+'_'+str(alpha)+'_'+str(reg))
                print('Model loaded')
                track_number = 500
                rec_val = model.recommendForUserSubset(val_users, track_number)
                print('Recommendations done')
                join = val_comb.join(rec_val,val_comb.user_idx == rec_val.user_idx)
                predictionAndLabels = join.rdd.map(lambda r: ([track.track_idx for track in r.recommendations], r.val_labels))
                metrics = RankingMetrics(predictionAndLabels)
                print('Ranking Metrics done')
                mavgp = metrics.meanAveragePrecision
                results.append((rank,alpha,reg,mavgp))
                print("Rank : ",rank,"Alpha : ",alpha,"Reg : ",reg,"MAP : ",mavgp)
    
    print('Validation completed.')
    sc.parallelize(results).saveAsTextFile("MAP_grid_search_new.txt")
    Print('MAP_grid_search.txt saved')
    pass

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('validation').getOrCreate()

    # Call our main routine
    main(spark)