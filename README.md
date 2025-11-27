# 🎬 Movie Recommendation System (Python)

This is a simple **Content-Based Movie Recommendation System** built using
**Python, Pandas, NumPy & scikit-learn**.  
It recommends similar movies based on *genre similarity* using **Cosine Similarity**.

---
## 📁 Project Files

| File | Description |
|------|-------------|
| `movies.csv` | Movie dataset containing titles, genres, and ratings |
| `movie_recommendation.py` | Main Python script for generating movie recommendations |
| `requirements.txt` | List of dependencies required to run the project |
| `README.md` | Project documentation (this file) |

---

## 🚀 How it Works?

1. Load the dataset (`movies.csv`)
2. Convert *genres* into numerical vectors using **CountVectorizer**
3. Compute similarity between movies using **Cosine Similarity**
4. Recommend movies that are most similar to the selected movie

---

## 🔧 Tech Used

| Library | Purpose |
|--------|---------|
| Pandas | Data handling |
| NumPy | Numerical computation |
| Scikit-Learn | Feature vectorization + Cosine similarity |
