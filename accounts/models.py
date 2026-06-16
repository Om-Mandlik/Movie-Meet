from django.db import models
from django.contrib.auth.models import User
import json



class Watchlist(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    imdb_id = models.CharField(max_length=20)
    movie_title = models.CharField(max_length=255)
    poster = models.URLField(blank=True)
    added_at = models.DateTimeField(auto_now_add=True)

class Review(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    imdb_id = models.CharField(max_length=20)
    rating = models.IntegerField()
    comment = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

class LikedMovie(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='liked_movies')
    imdb_id = models.CharField(max_length=20)
    movie_title = models.CharField(max_length=255)
    poster = models.URLField(blank=True)
    liked_at = models.DateTimeField(auto_now_add=True)

    
    class Meta:
        unique_together = ('user', 'imdb_id')

    def __str__(self):
        return f"{self.user.username} - {self.movie_title}"



class Profile(models.Model):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    interested_in = models.CharField(max_length=1, choices=GENDER_CHOICES)
    bio = models.TextField(blank=True, max_length=500)
    
    # Store the vector array as a JSON text string for SQLite compatibility
    _taste_vector = models.TextField(db_column='taste_vector', blank=True, null=True)

    # Python properties to easily save/get lists instead of parsing strings manually
    @property
    def taste_vector(self):
        if self._taste_vector:
            return json.loads(self._taste_vector)
        return None

    @taste_vector.setter
    def taste_vector(self, value):
        if value is not None:
            self._taste_vector = json.dumps(list(value))
        else:
            self._taste_vector = None

    def __str__(self):
        return f"{self.user.username}'s Profile"

class MatchAction(models.Model):
    ACTION_CHOICES = [
        ('LIKE', 'Like'),
        ('DISLIKE', 'Dislike'),
    ]
    user_from = models.ForeignKey(User, on_delete=models.CASCADE, related_name='actions_sent')
    user_to = models.ForeignKey(User, on_delete=models.CASCADE, related_name='actions_received')
    action_type = models.CharField(max_length=7, choices=ACTION_CHOICES)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user_from', 'user_to')

    def __str__(self):
        return f"{self.user_from.username} {self.action_type}ed {self.user_to.username}"


class DateMatch(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'Matched - Chat Open'),
        ('PLANNING', 'Planning a Date'),
        ('SCHEDULED', 'Date Scheduled'),
        ('COMPLETED', 'Date Completed'),
    ]
    user_one = models.ForeignKey(User, on_delete=models.CASCADE, related_name='dates_initiated')
    user_two = models.ForeignKey(User, on_delete=models.CASCADE, related_name='dates_accepted')
    
    
    common_movies_count = models.IntegerField(default=0)
    
    status = models.CharField(max_length=15, choices=STATUS_CHOICES, default='PENDING')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Match: {self.user_one.username} & {self.user_two.username} ({self.status})"
    
class ChatMessage(models.Model):
    # Links directly to the unique mutual match authorization row
    match = models.ForeignKey(DateMatch, on_delete=models.CASCADE, related_name='messages')
    sender = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sent_messages')
    message_text = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)
    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        return f"From {self.sender.username} at {self.timestamp.strftime('%H:%M')}"