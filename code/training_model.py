import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from urllib.parse import urlparse
from scipy.sparse import hstack
import nltk

# The print statements were for debugging

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# Function to extract domain from URL
def extract_domain(url):
    if isinstance(url, str):  # Check if the URL is a string
        try:
            parsed_url = urlparse(url)
            return parsed_url.netloc
        except Exception as e:
            print(f"Error parsing URL: {url}, error: {e}")
            return ''
    else:
        return ''

# Function to preprocess text (basic tokenization)
def preprocess_text(text):
    return text.lower()  # Basic preprocessing (lowercase)

print("Loading datasets...")
# Load datasets
news_data = pd.read_csv('data set/fake_or_real_news.csv')
articles_data = pd.read_csv('data set/articles.csv')

print("Processing first dataset...")
# Extract domains from URLs
news_data['domain'] = news_data['news_url'].apply(extract_domain)
news_data = news_data[['title', 'domain', 'real']].dropna()
news_data.rename(columns={'real': 'label'}, inplace=True)

print("Processing second dataset...")
# Process the second dataset
articles_data = articles_data[['title', 'text', 'label']].dropna()
articles_data['domain'] = ''  # No domain available, so add empty column

# Convert labels to numeric values (0 = Fake, 1 = Real)
label_mapping = {'FAKE': 0, 'REAL': 1}
news_data['label'] = news_data['label'].map(label_mapping)
articles_data['label'] = articles_data['label'].map(label_mapping)

# Ensure all labels are numeric
news_data = news_data.dropna(subset=['label'])
articles_data = articles_data.dropna(subset=['label'])

# Convert label column to integer
news_data['label'] = news_data['label'].astype(int)
articles_data['label'] = articles_data['label'].astype(int)

# Merge both datasets
print("Merging datasets...")
data = pd.concat([news_data, articles_data], ignore_index=True)

# Prepare features and labels
X_text = data['title'].apply(preprocess_text)  # Simple preprocessing
X_domain = data['domain']
y = data['label']  # Target label

# Handle missing values
X_text = X_text.fillna('')  # Replace NaN values with empty strings
X_domain = X_domain.fillna('')  # Replace NaN values with empty strings

# Replace empty domains with a placeholder
X_domain = X_domain.replace('', 'no_domain')

# Vectorize text data with unigrams and bigrams
print("Vectorizing text data...")
text_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))  # Unigrams and bigrams
X_text_vectorized = text_vectorizer.fit_transform(X_text)

# Vectorize domain names
print("Vectorizing domain names...")
domain_vectorizer = TfidfVectorizer()
X_domain_vectorized = domain_vectorizer.fit_transform(X_domain)

# Combine text and domain features
print("Combining features...")
X_combined = hstack([X_text_vectorized, X_domain_vectorized])

# Split dataset into training and testing sets
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model with GridSearchCV
print("Training model with GridSearchCV...")

param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'],  # L2 regularization
    'solver': ['liblinear']
}

log_reg_model = LogisticRegression(max_iter=500)

# Grid search for hyperparameter tuning
grid_search = GridSearchCV(log_reg_model, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters found by GridSearchCV
best_model = grid_search.best_estimator_
print("Best model found:", grid_search.best_params_)

# Evaluate the best model
print("Evaluating model...")
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Save the best model and vectorizers
print("Saving model and vectorizers...")
os.makedirs('pkl files', exist_ok=True)  # Ensure the directory exists
joblib.dump(best_model, 'pkl files/model.pkl')
joblib.dump(text_vectorizer, 'pkl files/vectorizer.pkl')
joblib.dump(domain_vectorizer, 'pkl files/domain_vectorizer.pkl')

print("Model training complete and saved!")
