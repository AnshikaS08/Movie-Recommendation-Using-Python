# ----------------------------------------------
# MOVIE RECOMMENDATION SYSTEM USING PYTHON
# ----------------------------------------------

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset 
df = pd.read_csv("movies.csv")

# Select necessary columns
df = df[['title','genres','keywords','cast','director']]
df.dropna(inplace=True)

# Create combined tags column
df['tags'] = df['genres'] + " " + df['keywords'] + " " + df['cast'] + " " + df['director']

# Convert text → vectors using NLP
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()

# Calculate similarity score
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    if movie not in df['title'].values:
        print("⚠ Movie not found — check name again.\n")
        return

    movie_index = df[df['title'] == movie].index[0]
    distances = similarity[movie_index]

    # top 5 similar movies (excluding itself)
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

    print(f"\n🎬 Recommended Movies for ➤ {movie}\n")
    for i in movies_list:
        print(" ➤", df.iloc[i[0]].title)

# ----------- RUN (Enter any movie to get recommendations) ----------

print("\n🔹 Welcome to Movie Recommendation System 🔹")
movie_name = input("\nEnter Movie Name: ")

recommend(movie_name)
print("\nDone ✔")
