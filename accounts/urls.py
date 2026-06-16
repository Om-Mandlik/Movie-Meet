from django.urls import path
from . import views 
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('',views.home,name='home'),
    path('movie/<str:imdb_id>',views.movie_detail,name='movie_detail'),
    path("watchlist/add/",views.add_to_watchlist,name="add_to_watchlist"),
    path("watchlist/",views.watchlist,name="watchlist"),
    path("watchlist/remove/<int:pk>/",views.remove_watchlist,name="remove_watchlist"),
    path("movie/like/",views.like_movie,name="like_movie"),
    path("liked-movies/",views.liked_movies,name="liked_movies"),
    path('register/',views.register_view,name='register'),
    path('discover/', views.discover_taste_matches, name='discover_matches'),
    path('swipe/<int:target_user_id>/<str:action_type>/', views.swipe_user, name='swipe_user'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('chat/<int:match_id>/', views.chat_room, name='chat_room'),
    path('matches/', views.matches_dashboard, name='matches_dashboard'),
    path('profile/', views.profile_view, name='user_profile'),
    path('chat/<int:match_id>/send/', views.send_message_api, name='send_message_api'),
    path('chat/<int:match_id>/sync/', views.get_new_messages_api, name='get_new_messages_api'),
]
