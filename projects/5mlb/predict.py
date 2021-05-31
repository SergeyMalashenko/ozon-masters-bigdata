import argparse
import os
import sys
import random

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

import mlflow

def parse_args():
    parser = argparse.ArgumentParser(description="5mlb")
    parser.add_argument(
        "--train_path_in",
        type=str,
        default='',
        help="path to prepaired test",
        required=True
    )
    parser.add_argument(
        "--sklearn_model",
        type=str,
        default='model5b',
        help="model ml flow name",
        required=True
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default='1',
        help="model ml flow version",
        required=True
    )
    parser.add_argument(
        "--predict_path_out",
        type=str,
        default='',
        help="path to predictions",
        required=True
    )

    return parser.parse_args()
from pyspark.sql.types import *

def main():
    try:
        args = parse_args()
        train_path_in = args.train_path_in
        predict_path_out = args.predict_path_out

    except:
        print(sys.argv, len(sys.argv))
        sys.exit(1)


    SPARK_HOME = "/usr/hdp/current/spark2-client"
    os.environ["SPARK_HOME"] = SPARK_HOME
    PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
    sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.7-src.zip"))
    sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    import pyspark.sql.functions as sf
    from mlflow.tracking import MlflowClient
    experiment_id = mlflow.get_experiment_by_name('5mlb').experiment_id
    client = MlflowClient()
    mod_ = None
    for rm in client.list_registered_models():
        rm = dict(rm)
        print(rm)
        if rm['name']== args.sklearn_model:
            mod_=rm
            break
    uri_ = mod_['latest_versions'][0].source
    print("U   R   I   ======", uri_)    
    experiment = client.get_experiment(experiment_id)
    experiment_id = mlflow.get_experiment_by_name('5mlb').experiment_id    
    print("Name: {}".format(experiment.name))
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

    
    
    #load_model_name = client.search_runs(experiment_id)[0].info.artifact_uri + '/' + args.sklearn_model
    print(train_path_in)
    df = spark.read.parquet(train_path_in)
    print(df.show())
    from pyspark.sql.functions import expr   
    #arr_size = 100
    #df2 = df.select([expr('features[' + str(x) + ']') for x in range(0, arr_size)])
    #new_colnames = ['val_' + str(i) for i in range(0, arr_size)] 
    #df2 = df2.toDF(*new_colnames)
    df2 = df 
    print(df2.show())
    print(df2.columns)
    print(list(df2.columns)[1:])
    predict = mlflow.pyfunc.spark_udf(spark, uri_)
    dfs = df2.withColumn("predictions", predict(*df2.columns[1:]))
    print(dfs.show())
    dfs.select("id","predictions").write.mode("overwrite").csv(args.predict_path_out)



if __name__ == "__main__":
    main()
