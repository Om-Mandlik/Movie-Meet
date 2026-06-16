# Movie_Meet/asgi.py
import os
import django
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Movie_Meet.settings')
django.setup()  # Initializes your apps and models safely for async usage

# Import channels layers after django.setup() to avoid loading race conditions
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import accounts.routing

application = ProtocolTypeRouter({
    # Standard HTTP handling
    "http": get_asgi_application(),
    
    # Asynchronous WebSocket Routing
    "websocket": AuthMiddlewareStack(
        URLRouter(
            accounts.routing.websocket_urlpatterns
        )
    ),
})