import requests
from django.conf import settings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .models import Profile, LikedMovie

BASE_URL = "https://www.omdbapi.com/"


def search_movies(query):
    response = requests.get(
        BASE_URL,
        params={
            "apikey": settings.OMDB_API_KEY,
            "s": query
        }
    )

    return response.json()


def get_movie(imdb_id):
    response = requests.get(
        BASE_URL,
        params={
            "apikey": settings.OMDB_API_KEY,
            "i": imdb_id,
            "plot": "full"
        }
    )

    return response.json()

def get_movie_description_from_tmdb(imdb_id):
    """
    Fetches the movie overview description from TMDB using the imdb_id.
    """
    api_key = getattr(settings, "TMDB_API_KEY", "YOUR_TMDB_API_KEY")
    url = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key={api_key}&external_source=imdb_id"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            movie_results = response.json().get("movie_results", [])
            if movie_results:
                return movie_results[0].get("overview", "")
    except Exception:
        pass
    return ""

def update_profile_taste_vector(user):
    """
    Rebuilds a user's taste vector using TF-IDF word weighting.
    No Hugging Face models needed.
    """
    profile = user.profile
    liked_movies = LikedMovie.objects.filter(user=user)

    # 1. Cold start fallback execution state
    if not liked_movies.exists():
        profile.taste_vector = None
        profile.save()
        return

    # 2. Extract every descriptive text string available across the database
    # TF-IDF needs to see what words exist overall to calculate weights accurately
    all_liked_entries = LikedMovie.objects.all()
    
    # We compile a background text dictionary map of: { imdb_id: "Title and description" }
    corpus_dict = {}
    user_texts = []
    
    for entry in all_liked_entries:
        if entry.imdb_id not in corpus_dict:
            # In a heavy production app, cache this text description layer on a Movie model
            overview = get_movie_description_from_tmdb(entry.imdb_id)
            corpus_dict[entry.imdb_id] = f"{entry.movie_title} {overview}"
            
        # If this item belongs to our current targeted user context, append it to their calculations
        if entry.user == user:
            user_texts.append(corpus_dict[entry.imdb_id])

    if not user_texts:
        return

    # 3. Fit the Vectorizer across all unique movie descriptions available
    # max_features=384 locks the dimension size down exactly to match your existing Profile structure
    vectorizer = TfidfVectorizer(stop_words='english', max_features=384)
    
    # Read text list matrices
    all_corpus_texts = list(corpus_dict.values())
    vectorizer.fit(all_corpus_texts)
    
   # ... (Step 4 matrix calculations inside services.py) ...
    user_vectors_matrix = vectorizer.transform(user_texts).toarray()
    mean_vector = np.mean(user_vectors_matrix, axis=0)
    
    # --- FIX CRASH: If the result is invalid or full of NaNs, reset it cleanly ---
    if np.isnan(mean_vector).any():
        profile.taste_vector = None
    else:
        profile.taste_vector = mean_vector.tolist()
        
    profile.save()
    # 5. Save the computed numeric array directly down to the text field model property
    profile.taste_vector = mean_vector.tolist()
    profile.save()