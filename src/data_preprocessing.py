from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col


def load_all_data():
    spark = SparkSession.builder \
        .appName("Movie Recommendation System") \
        .getOrCreate()

    # Load raw data
    ratings = spark.read.text("data/ratings.dat")
    movies = spark.read.text("data/movies.dat")
    tags = spark.read.text("data/tags.dat")

    # Parse ratings.dat → userId::movieId::rating::timestamp
    ratings = ratings.withColumn("value", split(col("value"), "::")) \
        .select(
            col("value")[0].cast("int").alias("userId"),
            col("value")[1].cast("int").alias("movieId"),
            col("value")[2].cast("float").alias("rating"),
            col("value")[3].alias("timestamp")
        )

    # Parse movies.dat → movieId::title::genres
    movies = movies.withColumn("value", split(col("value"), "::")) \
        .select(
            col("value")[0].cast("int").alias("movieId"),
            col("value")[1].alias("title"),
            col("value")[2].alias("genres")
        )

    # Parse tags.dat → userId::movieId::tag::timestamp
    tags = tags.withColumn("value", split(col("value"), "::")) \
        .select(
            col("value")[0].cast("int").alias("userId"),
            col("value")[1].cast("int").alias("movieId"),
            col("value")[2].alias("tag"),
            col("value")[3].alias("timestamp")
        )

    return ratings, movies, tags, spark