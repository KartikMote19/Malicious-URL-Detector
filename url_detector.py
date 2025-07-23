import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import re
import urllib.parse
import pickle
import os
from collections import Counter
import math

class URLFeatureExtractor:
    def __init__(self):
        self.suspicious_keywords = [
            'secure', 'account', 'update', 'confirm', 'login', 'signin', 'bank',
            'paypal', 'ebay', 'amazon', 'microsoft', 'apple', 'google', 'verify',
            'suspended', 'limited', 'click', 'here', 'now', 'urgent', 'immediate'
        ]
    
    def extract_features(self, url):
        """Extract features from a single URL"""
        features = {}
        
        # Basic URL properties
        features['url_length'] = len(url)
        features['num_dots'] = url.count('.')
        features['num_slashes'] = url.count('/')
        features['num_underscores'] = url.count('_')
        features['num_dashes'] = url.count('-')
        features['num_equals'] = url.count('=')
        features['num_questions'] = url.count('?')
        features['num_ampersands'] = url.count('&')
        features['num_at_symbols'] = url.count('@')
        features['num_percent'] = url.count('%')
        
        # Parse URL components
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path
            query = parsed.query
        except:
            domain = ""
            path = ""
            query = ""
        
        # Domain features
        features['domain_length'] = len(domain)
        features['num_subdomains'] = domain.count('.') - 1 if domain.count('.') > 0 else 0
        features['has_ip'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', domain) else 0
        
        # Path features
        features['path_length'] = len(path)
        features['num_path_segments'] = len([x for x in path.split('/') if x])
        
        # Query features
        features['query_length'] = len(query)
        features['num_query_params'] = len([x for x in query.split('&') if x]) if query else 0
        
        # Suspicious keywords
        url_lower = url.lower()
        features['suspicious_keywords'] = sum(1 for keyword in self.suspicious_keywords if keyword in url_lower)
        
        # Character diversity (entropy)
        features['entropy'] = self.calculate_entropy(url)
        
        # Special character ratios
        features['digit_ratio'] = sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0
        features['letter_ratio'] = sum(c.isalpha() for c in url) / len(url) if len(url) > 0 else 0
        features['special_char_ratio'] = sum(not c.isalnum() for c in url) / len(url) if len(url) > 0 else 0
        
        # URL scheme
        features['is_https'] = 1 if url.startswith('https://') else 0
        features['is_http'] = 1 if url.startswith('http://') else 0
        
        # Suspicious patterns
        features['has_shortener'] = 1 if any(short in domain for short in ['bit.ly', 'tinyurl', 't.co', 'goo.gl', 'ow.ly']) else 0
        features['has_suspicious_tld'] = 1 if any(tld in domain for tld in ['.tk', '.ml', '.ga', '.cf']) else 0
        
        return features
    
    def calculate_entropy(self, text):
        """Calculate Shannon entropy of text"""
        if not text:
            return 0
        
        counter = Counter(text)
        length = len(text)
        entropy = 0
        
        for count in counter.values():
            p = count / length
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy

class MaliciousURLDetector:
    def __init__(self):
        self.feature_extractor = URLFeatureExtractor()
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.feature_names = None
        self.is_trained = False
    
    def load_data(self, csv_file):
        """Load data from CSV file"""
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} URLs from {csv_file}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def prepare_features(self, urls):
        """Extract features from list of URLs"""
        features_list = []
        
        print("Extracting features...")
        for i, url in enumerate(urls):
            if i % 100 == 0:
                print(f"Processed {i}/{len(urls)} URLs")
            
            features = self.feature_extractor.extract_features(url)
            features_list.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Store feature names for consistency
        if self.feature_names is None:
            self.feature_names = features_df.columns.tolist()
        
        return features_df
    
    def train_model(self, csv_file):
        """Train the Random Forest model"""
        print("Starting model training...")
        
        # Load data
        df = self.load_data(csv_file)
        if df is None:
            return False
        
        # Check required columns
        required_columns = ['url', 'result']
        if not all(col in df.columns for col in required_columns):
            print(f"CSV must contain columns: {required_columns}")
            return False
        
        # Extract features
        X = self.prepare_features(df['url'].tolist())
        y = df['result'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train model
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Enhanced Accuracy Display
        print(f"\n{'='*50}")
        print(f"           MODEL PERFORMANCE RESULTS")
        print(f"{'='*50}")
        print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Additional metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Precision:        {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:           {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-Score:         {f1:.4f} ({f1*100:.2f}%)")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                Benign  Malicious")
        print(f"Actual Benign     {cm[0][0]:4d}      {cm[0][1]:4d}")
        print(f"     Malicious    {cm[1][0]:4d}      {cm[1][1]:4d}")
        
        # Performance interpretation
        print(f"\n{'='*50}")
        print(f"           PERFORMANCE INTERPRETATION")
        print(f"{'='*50}")
        
        if accuracy >= 0.95:
            print("EXCELLENT: Model shows excellent performance!")
        elif accuracy >= 0.90:
            print("VERY GOOD: Model shows very good performance!")
        elif accuracy >= 0.85:
            print("GOOD: Model shows good performance!")
        elif accuracy >= 0.80:
            print("FAIR: Model shows fair performance - consider more training data!")
        else:
            print("POOR: Model needs improvement - add more diverse training data!")
        
        print(f"{'='*50}")
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious']))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        return True
    
    def predict_url(self, url):
        """Predict if a URL is malicious or benign"""
        if not self.is_trained:
            print("Model is not trained yet!")
            return None
        
        # Extract features
        features = self.feature_extractor.extract_features(url)
        features_df = pd.DataFrame([features])
        
        # Ensure feature consistency
        for feature in self.feature_names:
            if feature not in features_df.columns:
                features_df[feature] = 0
        
        features_df = features_df[self.feature_names]
        
        # Make prediction
        prediction = self.model.predict(features_df)[0]
        probability = self.model.predict_proba(features_df)[0]
        
        return {
            'prediction': prediction,
            'label': 'Malicious' if prediction == 1 else 'Benign',
            'confidence': max(probability),
            'malicious_probability': probability[1] if len(probability) > 1 else probability[0]
        }
    
    def save_model(self, filename='url_detector_model.pkl'):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='url_detector_model.pkl'):
        """Load a pre-trained model"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            
            print(f"Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def main():
    print("=" * 60)
    print("         MALICIOUS URL DETECTOR")
    print("=" * 60)
    print()
    
    detector = MaliciousURLDetector()
    
    # Check if pre-trained model exists
    if os.path.exists('url_detector_model.pkl'):
        load_existing = input("Found existing model. Load it? (y/n): ").lower().strip()
        if load_existing == 'y':
            detector.load_model()
    
    # Ask for model training
    if not detector.is_trained:
        train_model = input("Do you want to train the model? (y/n): ").lower().strip()
        
        if train_model == 'y':
            csv_file = input("Enter CSV file path (default: 'url_data.csv'): ").strip()
            if not csv_file:
                csv_file = 'url_data.csv'
            
            if not os.path.exists(csv_file):
                print(f"CSV file '{csv_file}' not found!")
                print("Please ensure the CSV file exists with columns: url, label, result")
                return
            
            success = detector.train_model(csv_file)
            if not success:
                print("Model training failed!")
                return
        else:
            print("Skipping model training...")
            if not detector.is_trained:
                print("No trained model available. Exiting...")
                return
    
    print("\n" + "=" * 60)
    print("         URL DETECTION INTERFACE")
    print("=" * 60)
    
    # Main detection loop
    while True:
        print()
        url = input("Enter URL to check (or 'quit' to exit): ").strip()
        
        if url.lower() in ['quit', 'exit', 'q']:
            break
        
        if not url:
            print("Please enter a valid URL.")
            continue
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        print(f"\nAnalyzing URL: {url}")
        print("-" * 50)
        
        result = detector.predict_url(url)
        
        if result:
            print(f"Prediction: {result['label']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Malicious Probability: {result['malicious_probability']:.4f}")
            
            if result['prediction'] == 1:
                print("WARNING: This URL appears to be MALICIOUS!")
            else:
                print("SAFE: This URL appears to be BENIGN.")
        
        print()
        continue_check = input("Do you want to check another URL? (y/n): ").lower().strip()
        if continue_check != 'y':
            break
    
    print("\nThank you for using the Malicious URL Detector!")
    print("Stay safe online!")

if __name__ == "__main__":
    main()
