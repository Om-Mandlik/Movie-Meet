from django.db.models import Q
from .models import ChatMessage

def unread_messages_counter(request):
    if request.user.is_authenticated:
        # 1. Find messages where the user is part of the match room (either user_one or user_two)
        # 2. Exclude messages where the current user is the sender
        # 3. Filter for unread messages only
        unread_count = ChatMessage.objects.filter(
            Q(match__user_one=request.user) | Q(match__user_two=request.user),
            is_read=False
        ).exclude(sender=request.user).count()
        
        return {'unread_messages_count': unread_count}
    
    return {'unread_messages_count': 0}