#!/opt/conda/envs/dsenv/bin/python
import os, sys
import logging

from pyspark.sql import SparkSession
from pyspark.sql.types import *

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from pyspark.ml import Pipeline, PipelineModel

logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

try:
    path_to_saved_model = sys.argv[1]
    path_to_test_dataset = sys.argv[2]
    path_to_save_inference = sys.argv[3]
except:
    logging.critical("Need to pass both project_id and train dataset path")
    sys.exit(1)

schema = StructType([
    StructField("overall", DoubleType()),
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

model = PipelineModel.load(path_to_saved_model)

test = spark.read.json(path_to_test_dataset, schema=schema)
test = test.fillna({'summary':''})

predictions = model.transform(test)

predictions.select("id", "prediction").write.save(path_to_save_inference)

spark.stop()
