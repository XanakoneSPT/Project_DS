from django.db import models

class NewsArticle(models.Model):
    title = models.CharField(max_length=500)
    content = models.TextField()
    source = models.CharField(max_length=200)
    date_published = models.DateTimeField()
    is_fake = models.BooleanField()
    
    def __str__(self):
        return self.title