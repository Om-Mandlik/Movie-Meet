import json
import numpy as np
from django.core.exceptions import PermissionDenied
from django.db.models import Q
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
            if next_url:
                return redirect(next_url)
                
            return redirect('home')
        else:
            return render(request, 'accounts/login.html', {
                'error': 'Invalid username or password'
            })
    return render(request, 'accounts/login.html')


def logout_view(request):
    logout(request)
    return redirect('login')


def register_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")
        bio = request.POST.get("bio", "").strip()
        gender = request.POST.get("gender")

        if not username or not password or not gender:
            return render(request, "accounts/register.html", {
                'error': "Username, Password, and Gender are required fields.",
                'hide_navbar': True
            })

        if User.objects.filter(username=username).exists():
            return render(request, "accounts/register.html", {
                'error': "That username already exists. Please choose another one.",
                'hide_navbar': True
            })

        user = User.objects.create_user(username=username, email=email, password=password)
        user_profile, created = Profile.objects.get_or_create(user=user)
        user_profile.bio = bio
        user_profile.gender = gender
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
    query = request.GET.get("q")

    # 1. Standard Search Handling
    if query:
        movies = search_movies(query).get("Search", [])
    else:
        movies = search_movies("Marvel").get("Search", [])

    # Fetch counters for the user sidebar/stats indicators
    watchlist_count = Watchlist.objects.filter(user=request.user).count()
    liked_count = LikedMovie.objects.filter(user=request.user).count()

    # 2. PERSONALIZED RECOMMENDATION FEED GENERATION
    user = request.user
    user_profile, created = Profile.objects.get_or_create(user=user)
    
    # MYSQL FIX: Route raw data through parsing engine
    user_vector = parse_mysql_vector(user_profile.taste_vector)

    cold_start = False
    cold_start_message = ""
    recommended_movies = []

    if user_vector is None or liked_count < 3:
        cold_start = True
        cold_start_message = "Like at least 3 movies so our AI engine can map your taste profile!"
    else:
        user_vector_np = np.array(user_vector, dtype=np.float32)
        
        my_liked_ids = LikedMovie.objects.filter(user=user).values_list('imdb_id', flat=True)
        my_watchlist_ids = Watchlist.objects.filter(user=user).values_list('imdb_id', flat=True)
        excluded_ids = set(my_liked_ids) | set(my_watchlist_ids)

        # Pull matching orientation candidates
        peer_profiles = Profile.objects.filter(
            gender=user_profile.interested_in,
            interested_in=user_profile.gender
        ).exclude(user=user)[:50] # Pulled a higher batch size to filter in-memory safely

        peer_vectors = []
        valid_peers = []
        
        for p in peer_profiles:
            parsed_v = parse_mysql_vector(p.taste_vector)
            if parsed_v:
                try:
                    v_np = np.array(parsed_v, dtype=np.float32)
                    if not np.isnan(v_np).any() and v_np.shape == user_vector_np.shape:
                        peer_vectors.append(v_np)
                        valid_peers.append(p)
                except Exception:
                    continue

        seen_imdb_ids = set()

        if valid_peers and len(peer_vectors) > 0:
            user_matrix = user_vector_np.reshape(1, -1)
            peer_matrix = np.array(peer_vectors)
            scores = cosine_similarity(user_matrix, peer_matrix)[0]
            
            # Extract top 5 most similar users
            top_peer_indices = np.argsort(scores)[::-1][:5]
            top_peer_users = [valid_peers[idx].user for idx in top_peer_indices if scores[idx] > 0.4]

            peer_favorites = LikedMovie.objects.filter(
                user__in=top_peer_users
            ).exclude(imdb_id__in=excluded_ids).order_by('-liked_at')[:20]
            
            for movie in peer_favorites:
                if movie.imdb_id not in seen_imdb_ids:
                    seen_imdb_ids.add(movie.imdb_id)
                    recommended_movies.append({
                        "imdb_id": movie.imdb_id,
                        "title": movie.movie_title,
                        "poster": movie.poster,
                        "reason": "Popular among your top taste matches"
                    })

        # Fallback system if peer history tracking runs low
        if len(recommended_movies) < 4:
            global_trending = LikedMovie.objects.exclude(
                imdb_id__in=excluded_ids
            ).order_by('-liked_at')[:20]
            
            for movie in global_trending:
                if movie.imdb_id not in excluded_ids and movie.imdb_id not in seen_imdb_ids:
                    seen_imdb_ids.add(movie.imdb_id)
                    recommended_movies.append({
                        "imdb_id": movie.imdb_id,
                        "title": movie.movie_title,
                        "poster": movie.poster,
                        "reason": "Trending on Movie Meet"
                    })

    return render(
        request,
        "accounts/home.html",
        {
            "movies": movies,
            "query": query,
            "watchlist_count": watchlist_count,
            "liked_count": liked_count,
            "cold_start": cold_start,
            "cold_start_message": cold_start_message,
            "recommended_movies": recommended_movies[:8]
        }
    )


def movie_detail(request, imdb_id):
    movie = get_movie(imdb_id)
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
    movie = Watchlist.objects.get(id=pk, user=request.user)
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
            update_profile_taste_vector(request.user)

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
    
    # MYSQL FIX: Parse vector field through custom parsing middleware sanitiser
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

    candidate_profiles = (
        Profile.objects
        .filter(
            gender=user_profile.interested_in,
            interested_in=user_profile.gender
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
            # Ensure dimensions line up perfectly before feeding matrix engines
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
    user = request.user
    action_type = action_type.upper()
    
    if action_type not in ['LIKE', 'DISLIKE']:
        return JsonResponse({"success": False, "error": "Invalid action value"}, status=400)
    
    action, created = MatchAction.objects.get_or_create(
        user_from=user,
        user_to_id=target_user_id,
        defaults={'action_type': action_type}
    )
    
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
        
    try:
        data = json.loads(request.body)
        text = data.get("message", "").strip()
    except json.JSONDecodeError:
        return JsonResponse({"success": False, "error": "Malformed JSON input"}, status=400)
        
    if not text:
        return JsonResponse({"success": False, "error": "Message body cannot be empty"}, status=400)
        
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
    match_instance = get_object_or_404(DateMatch, id=match_id)
    
    if request.user != match_instance.user_one and request.user != match_instance.user_two:
        return JsonResponse({"success": False, "error": "Unauthorized Access"}, status=403)
        
    after_timestamp_str = request.GET.get('after', '')
    messages_query = ChatMessage.objects.filter(match=match_instance)
    
    if after_timestamp_str:
        after_dt = parse_datetime(after_timestamp_str)
        if after_dt:
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
