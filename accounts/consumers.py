import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .models import DateMatch, ChatMessage

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.match_id = self.scope['url_route']['kwargs']['match_id']
        self.room_group_name = f'chat_{self.match_id}'
        self.user = self.scope["user"]

        if not self.user.is_authenticated or not await self.is_user_in_match():
            await self.close()
            return

        # Join the channel group for this chat room
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        # Leave the channel group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            message_text = data.get('message', '').strip()
        except Exception:
            return

        if not message_text:
            return

        # Corrected method lookup to match our database wrapper below
        msg_obj = await self.save_message_to_db(message_text)
        timestamp_str = msg_obj.timestamp.strftime('%H:%M')

        # Broadcast message to EVERYONE connected to this room group name
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',  # Explicitly matches the method name below
                'message': message_text,
                'sender': self.user.username,
                'timestamp': timestamp_str
            }
        )

    # This method handles the 'chat_message' event sent by group_send above
    async def chat_message(self, event):
        # Push the message down over the WebSocket to the browser
        await self.send(text_data=json.dumps({
            'sender': event['sender'],
            'text': event['message'],
            'timestamp_display': event['timestamp']
        }))

    @database_sync_to_async
    def is_user_in_match(self):
        try:
            match_instance = DateMatch.objects.get(id=self.match_id)
            return self.user == match_instance.user_one or self.user == match_instance.user_two
        except DateMatch.DoesNotExist:
            return False

    @database_sync_to_async
    def save_message_to_db(self, message_text):
        """
        Asynchronously fetches the match model instance, creates the entry,
        and saves it with a default unread status tracker.
        """
        try:
            match_instance = DateMatch.objects.get(id=self.match_id)
            return ChatMessage.objects.create(
                match=match_instance,
                sender=self.user,
                message_text=message_text,
                is_read=False  # Triggers global navbar unread badges if recipient is away
            )
        except DateMatch.DoesNotExist:
            # Fallback error recovery if a room drops out mid-session
            class DummyMessage:
                import datetime
                timestamp = datetime.datetime.now()
            return DummyMessage()