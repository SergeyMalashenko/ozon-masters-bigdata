import numpy as np
import pandas as pd
import argparse
import os, sys
import logging
import mlflow
import mlflow.sklearn
import pyarrow.parquet as pq
from sklearn.linear_model import RidgeClassifier ,LogisticRegression
from sklearn.metrics import log_loss

def parse_args():
    parser = argparse.ArgumentParser(description="example")
    parser.add_argument(
        "--train_path_in",
        type=str,
        default='./',
        help="file",
    )
    parser.add_argument(
        "--sklearn_model",
        type=str,
        default='model5b',
        help="model name",
    )
    parser.add_argument(
        "--model_param1",
        type=str,
        default='l2',
        help="params",
    )
    return parser.parse_args() 


if __name__ == "__main__":
    try:
            args = parse_args()
            train_path_in = args.train_path_in
            sklearn_model = args.sklearn_model
            model_param1  = args.model_param1
    except:
            logging.critical("Need to pass dataset paths")
            sys.exit(1)
    logging.info(f"Uploading Parquet df: {train_path_in}")

    df = pq.read_table(source=train_path_in).to_pandas()
    y = df['label']
    X = df.iloc[:,2:]
    print(df.columns, df.shape)
    print(X.shape, y.shape)
    print(X.head(3))
    experiment_id = mlflow.get_experiment_by_name('5mlb').experiment_id
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    experiment = client.get_experiment(experiment_id)
    print("Name: {}".format(experiment.name))
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
    
    print("exp_id=",experiment_id)
    with mlflow.start_run(experiment_id=experiment_id):
    #mlflow.log_param("model_param1", model_param1) 
    #mlflow.log_param("train_path_in", train_path_in)
    #mlflow.log_param("sklearn_model",sklearn_model)
        
        clf = LogisticRegression().fit(X, y)

        score = clf.score(X, y)
        probability_class_1 = clf.predict_proba(X)[:, 1]
        ll_score = log_loss(y, probability_class_1)

        mlflow.log_metrics({"log_loss": ll_score, "score": score}) 

        mlflow.sklearn.log_model(sk_model=clf, artifact_path='model' , registered_model_name  =sklearn_model )
