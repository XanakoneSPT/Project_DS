import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

class FakeNewsModel:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.model_path = os.path.join(os.path.dirname(__file__), 'models/classifier.pkl')
        self.vectorizer_path = os.path.join(os.path.dirname(__file__), 'models/vectorizer.pkl')
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Load model if it exists
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.load_model()
    
    def preprocess_text(self, text):
        # Simple preprocessing (you can expand this)
        if isinstance(text, str):
            return text.lower()
        return ""
    
    def train(self, articles):
        # Convert data to pandas DataFrame
        df = pd.DataFrame(list(articles.values('title', 'content', 'is_fake')))
        
        # Combine title and content
        df['text'] = df['title'] + " " + df['content']
        df['text'] = df['text'].apply(self.preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['is_fake'], test_size=0.2, random_state=42
        )
        
        # Create TF-IDF features
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Save model
        self.save_model()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def predict(self, title, content):
        if not self.model or not self.vectorizer:
            raise ValueError("Model not trained yet")
        
        # Preprocess
        text = self.preprocess_text(f"{title} {content}")
        
        # Vectorize
        text_vec = self.vectorizer.transform([text])
        
        # Predict
        prediction = self.model.predict(text_vec)[0]
        
        # Get probabilities safely
        proba = self.model.predict_proba(text_vec)[0]
        # If we have 2 classes (fake and real), use index 1 for fake probability
        # If we have 1 class, use the single value
        if len(proba) > 1:
            probability = proba[1] if prediction else 1 - proba[0]
        else:
            probability = proba[0]
        
        return {
            'is_fake': bool(prediction),
            'confidence': float(probability)
        }
    
    def save_model(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def load_model(self):
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(self.vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)