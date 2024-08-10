from pyspark.sql import SparkSession, types, functions, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, FloatType
import pandas as pd

def processData(df):
    df = df.withColumnRenamed("body", "comment").withColumnRenamed("selftext", "post_body")
    null_comments = ["[deleted]", "[removed]"]
    cleaned_df = df.filter(~F.col("comment").isin(null_comments))
    cleaned_df = cleaned_df.filter((F.col("post_body").isNull()) | (F.trim(F.col("post_body")) == ""))

    cleaned_df = cleaned_df.withColumn("comment_score_ratio", F.col("comment_score")/F.col("max_comment_score"))

    # Get rid of duplicate comments
    window_spec = Window.partitionBy('post_id', 'comment').orderBy(F.col('comment_score').desc())
    without_dupes = cleaned_df.withColumn('row_number', F.row_number().over(window_spec))
    without_dupes = without_dupes.filter(F.col('row_number') == 1)
    without_dupes = without_dupes.drop('row_number')
    cleaned_df = without_dupes
    unwanted_columns = ["post_id", "comment_id", "post_body", "post_score", "max_comment_score", "post_ups", "controversiality", "num_comments"]
    cleaned_df = cleaned_df.drop(*unwanted_columns)

    return cleaned_df

if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("askreddit data process") \
        .getOrCreate()

    train_set_raw = spark.read.parquet("./data/joined_train_raw")
    test_set_raw = spark.read.parquet("./data/joined_test_raw")

    train_set_cleaned = processData(train_set_raw)
    test_set_cleaned = processData(test_set_raw)

    train_set_cleaned.write.parquet("./data/joined_train_clean", mode="overwrite")
    test_set_cleaned.write.parquet("./data/joined_test_clean", mode="overwrite")