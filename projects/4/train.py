#!/opt/conda/envs/dsenv/bin/python
import os, sys
import logging
from pyspark.sql import SparkSession
from pyspark.sql.types import *

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from model import pipeline

logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

try:
    path_to_dataset = sys.argv[1]
    path_to_save_model = sys.argv[2]
except:
    logging.critical("Need to pass both project_id and train dataset path")
    sys.exit(1)

schema = StructType([
    StructField("overall", DoubleType()),
    StructField("vote", StringType()),
    StructField("verified", BooleanType()),
    StructField("reviewTime", StringType()),
    StructField("reviewerID", StringType()),
    StructField("asin", StringType()),
    StructField("id", StringType()),
    StructField("reviewerName", StringType()),
    StructField("reviewText", StringType()),
    StructField("summary", StringType()),
    StructField("unixReviewTime", TimestampType())
])

dataset = spark.read.json(path_to_dataset, schema=schema)

dataset = dataset.fillna({'summary':''})

dataset.cache()

pipeline_model = pipeline.fit(dataset)

pipeline_model.write().overwrite().save(path_to_save_model)

spark.stop()
