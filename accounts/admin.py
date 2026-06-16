from django.contrib import admin
from .models import Profile, LikedMovie, MatchAction
# Register your models here.
admin.site.register(Profile)
admin.site.register(LikedMovie)
admin.site.register(MatchAction)