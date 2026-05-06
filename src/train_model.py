from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from src.data_preprocessing import load_all_data
def train():
    # Load data
    ratings, movies, tags, spark = load_all_data()
    # Split data
    train_data, test_data = ratings.randomSplit([0.8, 0.2])
    # ALS Model
    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True
    )
    # Train model
    model = als.fit(train_data)
    # Predictions
    predictions = model.transform(test_data)
    # Evaluation
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    print("RMSE:", rmse)
    # Generate recommendations
    print("Available sample users selected from dataset.")
    print("Enter comma-separated user IDs:")
    user_list = list(map(int, input().split(",")))
    users = ratings.where(ratings.userId.isin(user_list)).select("userId").distinct()
    recommendations = model.recommendForUserSubset(users, 5)

    print("\nTop 5 Recommendations for given users:\n")
    # Collect results
    recs = recommendations.collect()
    for row in recs:
        user_id = row["userId"]
        rec_list = row["recommendations"]   # ✅ FIXED
        # Get movie IDs
        movie_ids = [r["movieId"] for r in rec_list]
        # Fetch movie titles
        movie_data = movies.filter(movies.movieId.isin(movie_ids)) \
                           .select("movieId", "title") \
                           .collect()
        movie_dict = {m.movieId: m.title for m in movie_data}
        # Format output
        formatted = [
            {
                "movie": movie_dict.get(r["movieId"], "Unknown"),
                "rating": float(r["rating"])
            }
            for r in rec_list
        ]
        print(f"{user_id}: [")
        for item in formatted:
            print(f"   {{movie: {item['movie']}, rating: {item['rating']}}},")
        print("]\n")
    spark.stop()
if __name__ == "__main__":
    train()