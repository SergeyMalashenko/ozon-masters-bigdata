#!/opt/conda/envs/dsenv/bin/python
import os, sys
import argparse
import logging
import mlflow

import pandas as pd

from sklearn.compose       import ColumnTransformer
from sklearn.impute        import SimpleImputer
from sklearn.linear_model  import LogisticRegression
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from pyspark.ml.feature import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml         import Transformer

from pyspark     import SparkConf
from pyspark.sql import SparkSession

from pyspark.sql.types import *

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('train_path'   , type=str, default= "/datasets/amazon/all_reviews_5_core_train_extra_small_sentiment.json")
    parser.add_argument('sklearn_model', type=str)
    parser.add_argument('model_param1' , type=int, default=100, help='n_estimators')
    args = parser.parse_args()
    
    source_train_path = args.train_path
    sklearn_model     = args.sklearn_model
    model_param1      = args.model_param1
    
    target_train_path = "" 
    
    logging.basicConfig(level=logging.DEBUG)
    logging.info("CURRENT_DIR {}"     .format(os.getcwd() ) )
    logging.info("SCRIPT CALLED AS {}".format(sys.argv[0 ]) )
    logging.info("ARGS {}"            .format(sys.argv[1:]) )

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        
        logging.info( f"python3 etl.py {source_train_path} {target_train_path}" )
        os.system   ( f"python3 etl.py {source_train_path} {target_train_path} {run_id}" )
        logging.info( f"python3 train.py {target_train_path} {sklearn_model} {model_param1}" )
        os.system   ( f"python3 train.py {target_train_path} {sklearn_model} {model_param1} {run_id}" )

    logging.info("COMPLETED")

if __name__ == "__main__":
    main()
