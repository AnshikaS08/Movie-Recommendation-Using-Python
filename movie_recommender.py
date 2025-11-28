# ===============================================================
# 🎬 MOVIE RECOMMENDATION SYSTEM (Content Based Filtering)
# Libraries: sklearn + pandas
# Works directly in Jupyter Notebook (.ipynb)
# No CSV required — sample dataset included
# ===============================================================

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# SAMPLE DATA (Replace anytime)
# ---------------------------
movies = {
    'title': ['Avatar','Titanic','John Wick','Avengers','Iron Man','The Notebook','La La Land','Deadpool'],
    'genre' : ['Action Sci-Fi','Romance Drama','Action Thriller','Action Sci-Fi','Action Sci-Fi','Romance Drama','Romance Musical','Action Comedy'],
    'overview': [
        'A marine on an alien planet',
        'A love story on the Titanic ship',
        'Hitman takes revenge',
        'Superheroes fight to save Earth',
        'Billionaire becomes armored hero',
        'Couple fall in love and face struggles',
        'Jazz musician and actress fall in love',
        'Funny superhero with guns and swords'
    ]
}

df = pd.DataFrame(movies)

# -----------------------------------
# FEATURE COMBINATION FOR BETTER MATCH
# -----------------------------------
df['combined_features'] = df['genre'] + " " + df['overview']

# Convert text to vector
cv = CountVectorizer(stop_words='english')
vector = cv.fit_transform(df['combined_features'])

# Similarity score using cosine similarity
similarity = cosine_similarity(vector)

# ---------------------------------------------------
# RECOMMENDER FUNCTION 🧠
# input = movie name (string)
# output = top 5 similar movies
# ---------------------------------------------------
def recommend(movie_name):
    movie_name = movie_name.lower()
    
    # Find movie index safely
    matches = df[df['title'].str.lower() == movie_name]
    if matches.empty:
        print("❌ Movie not found in database!")
        return
    
    index = matches.index[0]
    
    # Get similarity scores
    distances = list(enumerate(similarity[index]))
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

    print(f"\n✨ Recommended movies similar to '{movie_name.title()}':\n")
    for i in movies_list:
        print("📌", df.iloc[i[0]].title)

# ---------------------------
# Try it
# ---------------------------
recommend("Avatar")
recommend("Titanic")
