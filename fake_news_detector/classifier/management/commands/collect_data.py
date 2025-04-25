from django.core.management.base import BaseCommand
import pandas as pd
from classifier.models import NewsArticle
from datetime import datetime
import os
from django.conf import settings

class Command(BaseCommand):
    help = 'Collect data from CSV files'
    
    def handle(self, *args, **options):
        # Define paths to data files
        base_dir = settings.BASE_DIR
        real_news_path = os.path.join(base_dir, 'data', 'True.csv')
        fake_news_path = os.path.join(base_dir, 'data', 'Fake.csv')
        
        # Check if files exist
        if not os.path.exists(real_news_path) or not os.path.exists(fake_news_path):
            self.stdout.write(self.style.ERROR('Data files not found. Please place data files in the "data" directory.'))
            return
            
        # Load datasets
        real_news = pd.read_csv(real_news_path)
        fake_news = pd.read_csv(fake_news_path)
        
        # Process real news
        for _, row in real_news.iterrows():
            NewsArticle.objects.create(
                title=row['title'],
                content=row['text'],
                source=row.get('source', 'Unknown'),
                date_published=datetime.now(),  # Adjust based on your dataset
                is_fake=False
            )
        
        # Process fake news
        for _, row in fake_news.iterrows():
            NewsArticle.objects.create(
                title=row['title'],
                content=row['text'],
                source=row.get('source', 'Unknown'),
                date_published=datetime.now(),  # Adjust based on your dataset
                is_fake=True
            )
        
        self.stdout.write(self.style.SUCCESS(f'Successfully imported {real_news.shape[0]} real and {fake_news.shape[0]} fake news articles'))