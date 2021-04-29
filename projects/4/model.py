#!/opt/conda/envs/dsenv/bin/python
from pyspark.ml.feature import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline

tokenizer_rt = RegexTokenizer(inputCol="reviewText", outputCol="words_rt", pattern="\\W")
# tokenizer_summary = RegexTokenizer(inputCol="summary", outputCol="words_summary", pattern="\\W")

stop_words = StopWordsRemover.loadDefaultStopWords("english")

swr_rt = StopWordsRemover(inputCol=tokenizer_rt.getOutputCol(),
                          outputCol="words_filtered_rt", stopWords=stop_words)
# swr_summary = StopWordsRemover(inputCol=tokenizer_summary.getOutputCol(),
#                                outputCol="words_filtered_summary", stopWords=stop_words)

hasher_rt = HashingTF(numFeatures=100, binary=True,
                      inputCol=swr_rt.getOutputCol(), outputCol="word_vector_rt")
# hasher_summary = HashingTF(numFeatures=100, binary=True,
#                       inputCol=swr_summary.getOutputCol(), outputCol="word_vector_summary")

assembler = VectorAssembler(inputCols=['word_vector_rt'], outputCol="features")

gbt = GBTRegressor(featuresCol="features", labelCol="overall", maxIter=10)

pipeline = Pipeline(stages=[
    tokenizer_rt,
    # tokenizer_summary,
    swr_rt,
    # swr_summary,
    hasher_rt,
    # hasher_summary,
    assembler,
    gbt
])
