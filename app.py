import streamlit as st
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, avg

st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #fffaf0, #fdf6e3);
    color: black;
}

/* All text */
html, body, [class*="css"] {
    color: black !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #fff0d6;
    color: black;
}


/* Buttons */
.stButton>button {
    background-color: #ffb347;
    color: black;
    border-radius: 8px;
}

div[data-baseweb="input"] input {
    background-color: #ffe5b4 !important;   /* light orange */
    color: black !important;
    border-radius: 6px;
}

/* Text area */
textarea {
    background-color: #ffe5b4 !important;
    color: black !important;
}

/* Radio label (Select Input Type) */
div[data-testid="stRadio"] label {
    color: black !important;
}

/* Radio options (Movie ID, Movie Name) */
div[data-testid="stRadio"] div {
    color: black !important;
}

/* Labels */
label {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Spark + Data
# -----------------------------
@st.cache_resource
def load_data():
    spark = SparkSession.builder \
        .appName("Movie Recommendation UI") \
        .getOrCreate()

    ratings = spark.read.text("data/ratings.dat")
    movies = spark.read.text("data/movies.dat")
    tags = spark.read.text("data/tags.dat")

    # Parse ratings
    ratings = ratings.withColumn("value", split(col("value"), "::")) \
        .select(
            col("value")[0].cast("int").alias("userId"),
            col("value")[1].cast("int").alias("movieId"),
            col("value")[2].cast("float").alias("rating"),
        )

    # Parse movies
    movies = movies.withColumn("value", split(col("value"), "::")) \
        .select(
            col("value")[0].cast("int").alias("movieId"),
            col("value")[1].alias("title"),
            col("value")[2].alias("genres")
        )

    # Parse tags
    tags = tags.withColumn("value", split(col("value"), "::")) \
        .select(
            col("value")[0].cast("int").alias("userId"),
            col("value")[1].cast("int").alias("movieId"),
            col("value")[2].alias("tag")
        )

    return spark, ratings, movies, tags


# -----------------------------
# Train / Load Model
# -----------------------------
@st.cache_resource
def train_model(_ratings):
    from pyspark.ml.recommendation import ALS

    model = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop"
    ).fit(_ratings)

    return model


# -----------------------------
# UI START
# -----------------------------
st.title("🎬 Movie Recommendation System")

spark, ratings, movies, tags = load_data()
model = train_model(ratings)

# -----------------------------
# 1️⃣ USER RECOMMENDATION
# -----------------------------
st.header("👤Get Recommendations by User ID")

user_input = st.text_input("Enter User IDs (comma separated)", "1,2")

if st.button("Get Recommendations", key="rec_button"):
    user_list = [int(x.strip()) for x in user_input.split(",")]

    users = ratings.where(ratings.userId.isin(user_list)).select("userId").distinct()
    recs = model.recommendForUserSubset(users, 5).collect()

    for row in recs:
        user_id = row["userId"]
        rec_list = row["recommendations"]

        movie_ids = [r["movieId"] for r in rec_list]
        movie_data = movies.filter(movies.movieId.isin(movie_ids)).collect()
        movie_dict = {m.movieId: m.title for m in movie_data}

        st.subheader(f"User {user_id}")

        for r in rec_list:
            pred_rating = min(float(r["rating"]), 5.0)  # scale fix

            st.write(f"🎥 {movie_dict.get(r['movieId'])} ⭐ {round(pred_rating,2)}")
            st.progress(pred_rating / 5)


# -----------------------------
# 2️⃣ MOVIE REVIEWS
# -----------------------------
st.header("🎥 Get Movie Reviews")

option = st.radio("Select Input Type", ["Movie ID", "Movie Name"])

if option == "Movie ID":
    movie_id = st.number_input("Enter Movie ID", min_value=1, step=1)

    if st.button("Show Reviews (ID)", key="movie_id_btn"):
        movie_name = movies.filter(movies.movieId == movie_id).collect()

        if movie_name:
            st.subheader(movie_name[0]["title"])

            # ✅ overall rating
            avg_rating = ratings.filter(ratings.movieId == movie_id) \
                .agg(avg("rating").alias("avg_rating")) \
                .collect()[0]["avg_rating"]

            st.write(f"Overall Rating: ⭐ {round(avg_rating,2)}")
            st.progress(min(avg_rating / 5, 1.0))

            # individual reviews
            reviews = ratings.filter(ratings.movieId == movie_id).limit(20).collect()

            st.write("### User Reviews:")
            for r in reviews:
                st.write(f"User {r['userId']} ➝ ⭐ {r['rating']}")
        else:
            st.write("Movie not found")

else:
    movie_name_input = st.text_input("Enter Movie Name")

    if st.button("Show Reviews (Name)", key="movie_name_btn"):
        movie_match = movies.filter(movies.title.contains(movie_name_input)).collect()

        if movie_match:
            movie_id = movie_match[0]["movieId"]
            st.subheader(movie_match[0]["title"])

            # ✅ overall rating
            avg_rating = ratings.filter(ratings.movieId == movie_id) \
                .agg(avg("rating").alias("avg_rating")) \
                .collect()[0]["avg_rating"]

            st.write(f"Overall Rating: ⭐ {round(avg_rating,2)}")
            st.progress(min(avg_rating / 5, 1.0))

            # reviews
            reviews = ratings.filter(ratings.movieId == movie_id).limit(20).collect()

            st.write("### User Reviews:")
            for r in reviews:
                st.write(f"User {r['userId']} ➝ ⭐ {r['rating']}")
        else:
            st.write("Movie not found")


# -----------------------------
# 3️⃣ TOP RATED MOVIES
# -----------------------------
st.header("💥 Top Rated Movies")

if st.button("Show Top Movies", key="top_movies_btn"):
    top_movies = ratings.groupBy("movieId") \
        .agg(avg("rating").alias("avg_rating")) \
        .orderBy(col("avg_rating").desc()) \
        .limit(10)

    result = top_movies.join(movies, "movieId").collect()

    for r in result:
        st.write(f"🎬 {r['title']} ⭐ {round(r['avg_rating'],2)}")
        st.progress(min(r['avg_rating'] / 5, 1.0))