from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('detect/', views.detect_fake_news, name='detect'),
    path('train/', views.train_model, name='train'),
]