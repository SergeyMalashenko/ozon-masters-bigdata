import argparse
import os, sys
import logging
import mlflow
import mlflow.sklearn
from pyspark.sql import SparkSession

from pyspark.sql.types import *
from pyspark.ml.feature import *
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType


from pyspark.ml import Pipeline

from pyspark.ml.pipeline import Transformer
from pyspark.ml.feature import HashingTF, IDF, Tokenizer ,StopWordsRemover
from pyspark.ml.util import  MLWritable
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params
from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasOutputCols, Param, Params
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GBTRegressor


##
def parse_args():
    parser = argparse.ArgumentParser(description="example")
    parser.add_argument(
        "--train_path_in",
        type=str,
        default='./',
        help="file",
    )
    parser.add_argument(
        "--train_path_out",
        type=str,
        default='./',
        help="file path )",
    )
    return parser.parse_args() 


def split_array_to_list(col):
    def to_list(v):
        return v.toArray().tolist()
    return F.udf(to_list, ArrayType(DoubleType()))(col)


def etl_data(train_path_in, train_path_out):
    SPARK_HOME = "/usr/hdp/current/spark2-client"
    os.environ["SPARK_HOME"] = SPARK_HOME
    PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
    sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.7-src.zip"))
    sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    logging.info(f"converting data")
    df =  (
        spark.read.format("json")
            .load(train_path_in)
           )
    tokenizer = Tokenizer(inputCol="reviewText", outputCol="rTwords")
    wremover = StopWordsRemover(inputCol="rTwords", outputCol="frTwords")
    hashingTF = HashingTF(inputCol="frTwords", outputCol="rawFeatures", numFeatures=100)
    idf = IDF(inputCol="rawFeatures", outputCol="idf_features")
    

    stage_filter = VectorAssembler(inputCols=['idf_features'],
                          outputCol='features').setHandleInvalid("skip")
    
    pipeline = Pipeline(stages=[ tokenizer , wremover, hashingTF , idf, stage_filter ])
    
    mod = pipeline.fit(df)
    df_new = mod.transform(df)
    if 'label' in df.columns:
        df_new = df_new.select("id","label", "features")
    else:
        df_new = df_new.select("id","features")
    
    @F.udf(ArrayType(FloatType()))
    def sparseVectorToArray(row):
        return row.toArray().tolist()

    features_col = "features"
    df_new_to = df_new.withColumn(features_col, sparseVectorToArray(features_col))
   
    from pyspark.sql.functions import expr
    arr_size = 100
    if 'label' in df.columns:
        df2 = df_new_to.select(['id','label']+[expr('features[' + str(x) + ']') for x in range(0, arr_size)])
        new_colnames = ['id','label']+ ['val_' + str(i) for i in range(0, arr_size)]
        df2 = df2.toDF(*new_colnames)
    else:
        df2 = df_new_to.select(['id']+[expr('features[' + str(x) + ']') for x in range(0, arr_size)])
        new_colnames = ['id']+['val_' + str(i) for i in range(0, arr_size)]
        df2 = df2.toDF(*new_colnames)



    ###df2 = df_new_to.select("*")
    df2.write.parquet(train_path_out ,mode="overwrite")
    logging.info(f"Uploading Parquet df: {train_path_out}")
    logging.info(f"df: {df2.show()}")
    print(df2.columns)
    print("etl")
    print(df2.show())

    
if __name__ == "__main__":

        try:
            args = parse_args()
            train_path_in = args.train_path_in
            train_path_out = args.train_path_out
        except:
            logging.critical("Need to pass dataset paths")
            sys.exit(1)

        etl_data(train_path_in, train_path_out)
