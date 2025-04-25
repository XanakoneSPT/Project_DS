from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import NewsArticle
from .ml_models import FakeNewsModel
from django.core.management import call_command
from django.contrib import messages

def index(request):
    return render(request, 'classifier/index.html')

def train_model(request):
    if request.method == 'POST':
        try:
            # Load data if needed
            if NewsArticle.objects.count() == 0:
                call_command('collect_data')
            
            # Train model
            model = FakeNewsModel()
            metrics = model.train(NewsArticle.objects.all())
            
            messages.success(request, f"Model trained successfully! Accuracy: {metrics['accuracy']:.2f}")
            return redirect('index')
        except Exception as e:
            messages.error(request, f"Error training model: {str(e)}")
    
    return render(request, 'classifier/train.html')

def detect_fake_news(request):
    if request.method == 'POST':
        title = request.POST.get('title', '')
        content = request.POST.get('content', '')
        
        try:
            model = FakeNewsModel()
            result = model.predict(title, content)
            
            return render(request, 'classifier/result.html', {
                'title': title,
                'content': content,
                'is_fake': result['is_fake'],
                'confidence': result['confidence'] * 100  # Convert to percentage
            })
        except Exception as e:
            messages.error(request, f"Error in prediction: {str(e)}")
            return redirect('detect')
    
    return render(request, 'classifier/detect.html')