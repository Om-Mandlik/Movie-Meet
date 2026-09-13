import json
import numpy as np
from django.db import transaction, IntegrityError
from django.db.models import Q, Count, Max
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from sklearn.metrics.pairwise import cosine_similarity
from django.utils.dateparse import parse_datetime
from django.contrib import messages

# Models
from .models import Watchlist, LikedMovie, Profile, MatchAction, DateMatch, ChatMessage

# Services 
from .services import search_movies, get_movie, update_profile_taste_vector

# ==========================================
# HELPER DATA CLEANER FOR AIVEN MYSQL STRING STORAGE
# ==========================================
def parse_mysql_vector(vector_field_data):
    """
    Safely converts strings or JSON blocks from managed cloud MySQL 
    instances back into clean mathematical float arrays.
    """
    if vector_field_data is None:
        return None
    if isinstance(vector_field_data, str):
        try:
            return json.loads(vector_field_data)
        except (json.JSONDecodeError, TypeError):
            try:
                # Manual parsing engine if commas are escaped natively
                return [float(x) for x in vector_field_data.strip('[]').split(',') if x.strip()]
            except ValueError:
                return None
    if isinstance(vector_field_data, list):
        return vector_field_data
    return None

# ==========================================
# SAFE EXTERNAL MOVIE API HELPERS
# ==========================================

def safe_search_movies(query):
    if not query:
        return []
    try:
        result = search_movies(query) or {}
        movies = result.get("Search", [])
        return movies if isinstance(movies, list) else []
    except Exception:
        return []


def safe_get_movie(imdb_id):
    if not imdb_id:
        return {}
    try:
        result = get_movie(imdb_id) or {}
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


def get_excluded_movie_ids(user):
    liked = set(LikedMovie.objects.filter(user=user).values_list("imdb_id", flat=True))
    saved = set(Watchlist.objects.filter(user=user).values_list("imdb_id", flat=True))
    return liked | saved


def content_recommendations(user, limit=12):
    """Lightweight content recommendations based on genres and liked titles."""
    liked = list(LikedMovie.objects.filter(user=user).order_by("-liked_at")[:10])
    if not liked:
        return []

    excluded = get_excluded_movie_ids(user)
    seen = set(excluded)
    recommendations = []
    genre_counts = {}

    # With the current models there is no local movie-metadata table, so these
    # calls are the remaining API cost. Cache movie metadata in the next step.
    for liked_movie in liked:
        details = safe_get_movie(liked_movie.imdb_id)
        for genre in str(details.get("Genre", "")).split(","):
            genre = genre.strip()
            if genre:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

    for genre, _count in sorted(genre_counts.items(), key=lambda item: item[1], reverse=True)[:4]:
        for movie in safe_search_movies(genre):
            imdb_id = movie.get("imdbID")
            if not imdb_id or imdb_id in seen:
                continue
            seen.add(imdb_id)
            recommendations.append({
                "imdb_id": imdb_id,
                "title": movie.get("Title", ""),
                "poster": movie.get("Poster", ""),
                "reason": f"Because you like {genre} movies",
            })
            if len(recommendations) >= limit:
                return recommendations

    # Fallback when genre metadata is missing.
    for liked_movie in liked[:5]:
        for movie in safe_search_movies(liked_movie.movie_title):
            imdb_id = movie.get("imdbID")
            if not imdb_id or imdb_id in seen:
                continue
            seen.add(imdb_id)
            recommendations.append({
                "imdb_id": imdb_id,
                "title": movie.get("Title", ""),
                "poster": movie.get("Poster", ""),
                "reason": f"Similar to {liked_movie.movie_title}",
            })
            if len(recommendations) >= limit:
                return recommendations

    return recommendations


def collaborative_recommendations(user, profile, limit=12):
    """Recommend movies liked by users with similar taste vectors."""
    user_vector = parse_mysql_vector(profile.taste_vector)
    if not user_vector:
        return []

    try:
        user_vector = np.asarray(user_vector, dtype=np.float32)
        if user_vector.ndim != 1 or not user_vector.size or not np.isfinite(user_vector).all():
            return []
    except (TypeError, ValueError):
        return []

    my_gender = profile.gender
    my_interest = profile.interested_in

    peers = Profile.objects.filter(
        Q(gender=my_interest) | Q(gender__isnull=True) | Q(gender=""),
        Q(interested_in=my_gender) | Q(interested_in__isnull=True) | Q(interested_in=""),
    ).exclude(user=user).select_related("user")[:50]

    valid_profiles, vectors = [], []
    for peer in peers:
        vector = parse_mysql_vector(peer.taste_vector)
        if not vector:
            continue
        try:
            vector = np.asarray(vector, dtype=np.float32)
            if vector.ndim == 1 and vector.shape == user_vector.shape and np.isfinite(vector).all():
                valid_profiles.append(peer)
                vectors.append(vector)
        except (TypeError, ValueError):
            continue

    if not vectors:
        return []

    try:
        scores = cosine_similarity(user_vector.reshape(1, -1), np.asarray(vectors))[0]
    except ValueError:
        return []

    peer_scores = sorted(zip(valid_profiles, scores), key=lambda item: float(item[1]), reverse=True)
    peer_scores = [(peer, float(score)) for peer, score in peer_scores[:5] if float(score) >= 0.40]
    if not peer_scores:
        return []

    score_map = {peer.user_id: score for peer, score in peer_scores}
    excluded = get_excluded_movie_ids(user)
    peer_ids = list(score_map)

    peer_likes = LikedMovie.objects.filter(user_id__in=peer_ids).exclude(imdb_id__in=excluded).order_by("-liked_at")
    movie_scores = {}
    movie_rows = {}
    for liked in peer_likes:
        score = score_map.get(liked.user_id, 0.0)
        movie_scores[liked.imdb_id] = movie_scores.get(liked.imdb_id, 0.0) + score
        movie_rows.setdefault(liked.imdb_id, liked)

    results = []
    for imdb_id, _score in sorted(movie_scores.items(), key=lambda item: item[1], reverse=True)[:limit]:
        liked = movie_rows[imdb_id]
        results.append({
            "imdb_id": imdb_id,
            "title": liked.movie_title,
            "poster": liked.poster,
            "reason": "Liked by people with similar taste",
        })
    return results


def trending_recommendations(user, limit=8):
    excluded = get_excluded_movie_ids(user)
    rows = LikedMovie.objects.exclude(imdb_id__in=excluded).values(
        "imdb_id", "movie_title", "poster"
    ).annotate(
        like_count=Count("id"),
        latest_like=Max("liked_at"),
    ).order_by("-like_count", "-latest_like")[:limit]

    return [
        {
            "imdb_id": row["imdb_id"],
            "title": row["movie_title"],
            "poster": row["poster"],
            "reason": "Trending on Movie Meet",
        }
        for row in rows
    ]


def personalized_home_movies(user, profile, liked_count):
    """Hybrid feed: content -> collaborative -> trending."""
    if liked_count == 0:
        return []

    results, seen = [], get_excluded_movie_ids(user)

    for movie in content_recommendations(user, 12):
        if movie["imdb_id"] not in seen:
            results.append(movie)
            seen.add(movie["imdb_id"])

    if liked_count >= 3:
        for movie in collaborative_recommendations(user, profile, 12):
            if movie["imdb_id"] not in seen:
                results.append(movie)
                seen.add(movie["imdb_id"])

    if len(results) < 8:
        for movie in trending_recommendations(user, 8):
            if movie["imdb_id"] not in seen:
                results.append(movie)
                seen.add(movie["imdb_id"])

    return results[:12]


# ==========================================
# AUTHENTICATION & REGISTRATION VIEWS
# ==========================================

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(
            request,
            username=username,
            password=password
        )
        if user is not None:
            login(request, user)
            
            # Dynamic Redirect Fix: Route to previous page if requested by interceptors
            next_url = request.GET.get('next')
            if next_url and next_url.startswith('/') and not next_url.startswith('//'):
                return redirect(next_url)
                
            return redirect('home')
        else:
            return render(request, 'accounts/login.html', {
                'error': 'Invalid username or password'
            })
    return render(request, 'accounts/login.html')


def logout_view(request):
    if request.method == 'POST':
        logout(request)
    return redirect('login')


def register_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")
        bio = request.POST.get("bio", "").strip()
        gender_input = request.POST.get("gender")

        if not username or not password or not gender_input:
            return render(request, "accounts/register.html", {
                'error': "Username, Password, and Gender are required fields.",
                'hide_navbar': True
            })

        if User.objects.filter(username=username).exists():
            return render(request, "accounts/register.html", {
                'error': "That username already exists. Please choose another one.",
                'hide_navbar': True
            })

        # NORMALIZE INPUT TO SHORT CHOICE KEYS ('M', 'F', 'O')
        if gender_input in ["Male", "M", "male"]:
            db_gender = "M"
            db_interest = "F"
        elif gender_input in ["Female", "F", "female"]:
            db_gender = "F"
            db_interest = "M"
        else:
            db_gender = "O"
            db_interest = "O"

        user = User.objects.create_user(username=username, email=email, password=password)
        user_profile, created = Profile.objects.get_or_create(user=user)
        user_profile.bio = bio
        user_profile.gender = db_gender
        user_profile.interested_in = db_interest  # Auto-populate to guarantee matching setup
        user_profile.save()

        login(request, user)
        return redirect("home")

    return render(request, "accounts/register.html", {
        'hide_navbar': True
    })

# ==========================================
# USER PROFILE MANAGEMENT VIEW
# ==========================================

@login_required
def profile_view(request):
    user = request.user
    user_profile, created = Profile.objects.get_or_create(user=user)
    
    if request.method == 'POST':
        new_bio = request.POST.get('bio', '').strip()
        user_profile.bio = new_bio 
        user_profile.save()
        
        messages.success(request, "Your profile has been updated successfully!")
        return redirect('user_profile')

    return render(request, 'accounts/profile.html', {'user': user, 'profile': user_profile})

# ==========================================
# CORE MOVIES ACTIONS & ARCHIVE VIEWS
# ==========================================

@login_required
def home(request):
    user = request.user
    query = (request.GET.get("q") or "").strip()
    watchlist_count = Watchlist.objects.filter(user=user).count()
    liked_count = LikedMovie.objects.filter(user=user).count()
    profile, _ = Profile.objects.get_or_create(user=user)

    if query:
        movies = safe_search_movies(query)
        recommendation_mode = "search"
    elif liked_count == 0:
        movies = safe_search_movies("Marvel")
        recommendation_mode = "marvel"
    else:
        movies = personalized_home_movies(user, profile, liked_count)
        recommendation_mode = "personalized"
        if not movies:
            movies = safe_search_movies("Marvel")
            recommendation_mode = "marvel_fallback"

    user_vector = parse_mysql_vector(profile.taste_vector)
    cold_start = liked_count < 3 or not user_vector
    if liked_count == 0:
        cold_start_message = "Like a movie to start getting recommendations based on your taste."
    elif liked_count < 3:
        cold_start_message = "Keep liking movies. After 3 likes, AI recommendations can also use similar users."
    else:
        cold_start_message = ""

    return render(request, "accounts/home.html", {
        "movies": movies[:12],
        "query": query,
        "watchlist_count": watchlist_count,
        "liked_count": liked_count,
        "cold_start": cold_start,
        "cold_start_message": cold_start_message,
        "recommended_movies": movies[:8] if recommendation_mode == "personalized" else [],
        "recommendation_mode": recommendation_mode,
    })

def movie_detail(request, imdb_id):
    movie = safe_get_movie(imdb_id)
    if not movie:
        return render(request, "accounts/movie_detail.html", {"movie": {}, "error": "Movie details are currently unavailable."}, status=404)
    return render(request, "accounts/movie_detail.html", {"movie": movie})


@login_required
def add_to_watchlist(request):
    if request.method == "POST":
        imdb_id = request.POST.get("imdb_id")
        title = request.POST.get("title")
        poster = request.POST.get("poster")

        Watchlist.objects.get_or_create(
            user=request.user,
            imdb_id=imdb_id,
            defaults={
                "movie_title": title,
                "poster": poster
            }
        )
    return redirect("watchlist")


@login_required
def watchlist(request):
    movies = Watchlist.objects.filter(user=request.user).order_by("-added_at")
    return render(request, "accounts/watchlist.html", {"movies": movies})


@login_required
def remove_watchlist(request, pk):
    if request.method != "POST":
        return redirect("watchlist")
    movie = get_object_or_404(Watchlist, id=pk, user=request.user)
    movie.delete()
    return redirect("watchlist")


@login_required
def like_movie(request):
    if request.method == "POST":
        imdb_id = request.POST.get("imdb_id")
        title = request.POST.get("title")
        poster = request.POST.get("poster")

        liked_movie, created = LikedMovie.objects.get_or_create(
            user=request.user,
            imdb_id=imdb_id,
            defaults={
                "movie_title": title,
                "poster": poster
            }
        )

        if created:
            try:
                update_profile_taste_vector(request.user)
            except Exception:
                messages.warning(request, "Movie liked, but your taste profile could not be updated yet.")

    return redirect("liked_movies")


@login_required
def liked_movies(request):
    movies = LikedMovie.objects.filter(user=request.user).order_by("-liked_at")
    return render(request, "accounts/liked_movies.html", {"movies": movies})

# ==========================================
# DISCOVER MATCHES ALGORITHMS & ACTIONS
# ==========================================

@login_required
def discover_taste_matches(request):
    user = request.user
    user_profile, created = Profile.objects.get_or_create(user=user)
    
    # MYSQL FIX: Parse vector field through custom parsing engine
    user_vector = parse_mysql_vector(user_profile.taste_vector)

    if user_vector is None:
        if request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.GET.get('format') == 'json':
            return JsonResponse({
                "status": "cold_start",
                "message": "Please like a few movies first to calculate your match suggestions!"
            }, status=400)
        return render(request, "accounts/discover.html", {
            "error": "Please like a few movies first to calculate your match suggestions!"
        })

    user_vector_np = np.array(user_vector, dtype=np.float32)
    already_interacted_ids = MatchAction.objects.filter(user_from=user).values_list('user_to_id', flat=True)

    # NORMALIZE USER ORIENTATION VALUES
    my_gender = user_profile.gender
    my_interest = user_profile.interested_in

    if my_gender in ["Male", "male", "M"]: my_gender = "M"
    elif my_gender in ["Female", "female", "F"]: my_gender = "F"

    if my_interest in ["Male", "male", "M"]: my_interest = "M"
    elif my_interest in ["Female", "female", "F"]: my_interest = "F"

    # Robust reciprocal filter configuration via explicit database conditional gates
    candidate_profiles = (
        Profile.objects
        .filter(
            Q(gender=my_interest) | Q(gender__isnull=True) | Q(gender=""),
            Q(interested_in=my_gender) | Q(interested_in__isnull=True) | Q(interested_in="")
        )
        .exclude(user=user)
        .exclude(user_id__in=already_interacted_ids)
        .select_related('user')
    )

    valid_candidates = []
    candidate_vectors = []

    for p in candidate_profiles:
        parsed_vector = parse_mysql_vector(p.taste_vector)
        if not parsed_vector:
            continue

        try:
            vector_np = np.array(parsed_vector, dtype=np.float32)
            if not np.isnan(vector_np).any() and vector_np.shape == user_vector_np.shape:
                valid_candidates.append(p)
                candidate_vectors.append(vector_np)
        except Exception:
            continue  

    # Catch empty candidates condition gracefully
    if not valid_candidates or len(candidate_vectors) == 0:
        if request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.GET.get('format') == 'json':
            return JsonResponse({"results": []})
        return render(request, "accounts/discover.html", {"candidates": []})

    user_matrix = user_vector_np.reshape(1, -1)
    candidate_matrix = np.array(candidate_vectors)
    scores = cosine_similarity(user_matrix, candidate_matrix)[0]

    scored_candidates = []
    for idx, profile in enumerate(valid_candidates):
        match_percentage = round(float(scores[idx]) * 100, 1)
        
        scored_candidates.append({
            "user_id": profile.user.id,
            "username": profile.user.username,
            "bio": profile.bio,
            "match_score": f"{match_percentage}%",
            "raw_score": match_percentage 
        })

    scored_candidates.sort(key=lambda x: x['raw_score'], reverse=True)
    top_matches = scored_candidates[:10]

    if request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.GET.get('format') == 'json':
        return JsonResponse({"results": top_matches})
        
    return render(request, "accounts/discover.html", {"candidates": top_matches})


@login_required
def swipe_user(request, target_user_id, action_type):
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "POST required"}, status=405)

    user = request.user
    try:
        target_user_id = int(target_user_id)
    except (TypeError, ValueError):
        return JsonResponse({"success": False, "error": "Invalid target user"}, status=400)

    if target_user_id == user.id:
        return JsonResponse({"success": False, "error": "You cannot match with yourself"}, status=400)

    if not User.objects.filter(id=target_user_id).exists():
        return JsonResponse({"success": False, "error": "User not found"}, status=404)

    action_type = action_type.upper()
    
    if action_type not in ['LIKE', 'DISLIKE']:
        return JsonResponse({"success": False, "error": "Invalid action value"}, status=400)
    
    try:
        with transaction.atomic():
            action, created = MatchAction.objects.get_or_create(
                user_from=user,
                user_to_id=target_user_id,
                defaults={'action_type': action_type}
            )
            if not created and action.action_type != action_type:
                action.action_type = action_type
                action.save(update_fields=['action_type'])
    except IntegrityError:
        return JsonResponse({"success": False, "error": "Could not save this action"}, status=409)
    
    is_match = False
    if action_type == 'LIKE':
        reverse_like = MatchAction.objects.filter(
            user_from_id=target_user_id, 
            user_to=user, 
            action_type='LIKE'
        ).exists()
        
        if reverse_like:
            is_match = True
            
            my_likes = set(LikedMovie.objects.filter(user=user).values_list('imdb_id', flat=True))
            their_likes = set(LikedMovie.objects.filter(user_id=target_user_id).values_list('imdb_id', flat=True))
            common_count = len(my_likes.intersection(their_likes))
            
            DateMatch.objects.get_or_create(
                user_one_id=min(user.id, int(target_user_id)),
                user_two_id=max(user.id, int(target_user_id)),
                defaults={'common_movies_count': common_count, 'status': 'PENDING'}
            )
            
    return JsonResponse({"success": True, "mutual_match": is_match})

# ==========================================
# MUTUAL MATCH DASHBOARD VIEW
# ==========================================

@login_required
def matches_dashboard(request):
    user = request.user
    matches = DateMatch.objects.filter(
        Q(user_one=user) | Q(user_two=user)
    ).select_related('user_one', 'user_two')
    
    mutual_matches = []
    for m in matches:
        other_user = m.user_two if m.user_one == user else m.user_one
        other_profile = getattr(other_user, 'profile', None)
        bio = other_profile.bio if other_profile else "No bio provided."
        
        unread_count = ChatMessage.objects.filter(
            match=m,
            is_read=False
        ).exclude(sender=user).count()
        
        mutual_matches.append({
            "match_id": m.id,
            "user_id": other_user.id,
            "username": other_user.username,
            "bio": bio,
            "common_movies": m.common_movies_count,
            "unread_count": unread_count, 
        })
        
    return render(request, "accounts/matches_dashboard.html", {"matches": mutual_matches})

# ==========================================
# CHAT ROOM CONTROLLERS
# ==========================================

@login_required
def chat_room(request, match_id):
    match_instance = get_object_or_404(DateMatch, id=match_id)
    
    if request.user != match_instance.user_one and request.user != match_instance.user_two:
        return render(request, 'accounts/suspended.html', {'error': 'Access Denied'})

    ChatMessage.objects.filter(
        match=match_instance,
        is_read=False
    ).exclude(sender=request.user).update(is_read=True)
        
    recipient = match_instance.user_two if match_instance.user_one == request.user else match_instance.user_one
    
    interaction_exists = MatchAction.objects.filter(
        Q(user_from=request.user, user_to=recipient, action_type='DISLIKE') |
        Q(user_from=recipient, user_to=request.user, action_type='DISLIKE')
    ).exists()
    
    if interaction_exists:
        return render(request, "accounts/chat_blocked.html", {
            "error": "This chat is unavailable because one of the parties has passed on the match."
        })
        
    chat_history = ChatMessage.objects.filter(match=match_instance).order_by('timestamp')
        
    return render(request, "accounts/chat_room.html", {
        "match": match_instance,
        "recipient": recipient,
        "chat_history": chat_history  
    })

# ==========================================
# REAL-TIME POLLED DATA API ENDPOINTS
# ==========================================

@login_required
def send_message_api(request, match_id):
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "Invalid request method"}, status=405)
        
    match_instance = get_object_or_404(DateMatch, id=match_id)
    
    if request.user != match_instance.user_one and request.user != match_instance.user_two:
        return JsonResponse({"success": False, "error": "Unauthorized Access"}, status=403)

    recipient = match_instance.user_two if match_instance.user_one == request.user else match_instance.user_one
    blocked = MatchAction.objects.filter(
        Q(user_from=request.user, user_to=recipient, action_type='DISLIKE') |
        Q(user_from=recipient, user_to=request.user, action_type='DISLIKE')
    ).exists()
    if blocked:
        return JsonResponse({"success": False, "error": "This chat is unavailable"}, status=403)
        
    try:
        data = json.loads(request.body)
        text = str(data.get("message", "")).strip()
    except json.JSONDecodeError:
        return JsonResponse({"success": False, "error": "Malformed JSON input"}, status=400)
        
    if not text:
        return JsonResponse({"success": False, "error": "Message body cannot be empty"}, status=400)

    if len(text) > 2000:
        return JsonResponse({"success": False, "error": "Message is too long"}, status=400)
        
    msg = ChatMessage.objects.create(
        match=match_instance,
        sender=request.user,
        message_text=text
    )
    
    return JsonResponse({
        "success": True,
        "username": msg.sender.username,
        "text": msg.message_text,
        "timestamp": msg.timestamp.strftime('%H:%M')
    })


@login_required
def get_new_messages_api(request, match_id):
    if request.method != "GET":
        return JsonResponse({"success": False, "error": "GET required"}, status=405)

    match_instance = get_object_or_404(DateMatch, id=match_id)
    
    if request.user != match_instance.user_one and request.user != match_instance.user_two:
        return JsonResponse({"success": False, "error": "Unauthorized Access"}, status=403)
        
    after_timestamp_str = request.GET.get('after', '')
    messages_query = ChatMessage.objects.filter(match=match_instance)
    
    if after_timestamp_str:
        after_dt = parse_datetime(after_timestamp_str)
        if not after_dt:
            return JsonResponse({"success": False, "error": "Invalid timestamp"}, status=400)
        messages_query = messages_query.filter(timestamp__gt=after_dt)
            
    message_list = []
    for m in messages_query:
        message_list.append({
            "id": m.id,
            "sender": m.sender.username,
            "is_me": m.sender == request.user,
            "text": m.message_text,
            "timestamp_iso": m.timestamp.isoformat(),
            "timestamp_display": m.timestamp.strftime('%H:%M')
        })
        
    return JsonResponse({"success": True, "messages": message_list})
