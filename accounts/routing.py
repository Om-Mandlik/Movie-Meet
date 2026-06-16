# accounts/routing.py
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    # Captures match_id dynamically from the WebSocket URL endpoint connection
    re_path(r'^ws/chat/(?P<match_id>\d+)/$', consumers.ChatConsumer.as_asgi()),
]