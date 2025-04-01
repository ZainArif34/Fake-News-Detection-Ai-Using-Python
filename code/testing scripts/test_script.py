import joblib
import numpy as np
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizers
model = joblib.load('pkl files/model.pkl')
text_vectorizer = joblib.load('pkl files/vectorizer.pkl')
domain_vectorizer = joblib.load('pkl files/domain_vectorizer.pkl')

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

# Function to fetch and extract clean text from the webpage
def fetch_page_text(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url, timeout=10)  # Adding a timeout for requests
        
        if response.status_code == 200:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text from paragraphs and remove unwanted tags (e.g., ads)
            paragraphs = soup.find_all('p')
            page_text = ' '.join([para.get_text() for para in paragraphs if para.get_text().strip() != ''])
            
            # Remove any special characters or HTML artifacts that may have slipped through
            page_text = ' '.join(page_text.split())
            return page_text
        else:
            print(f"Error fetching the URL: {url}. Status code: {response.status_code}")
            return ''
    except requests.exceptions.RequestException as e:
        print(f"Error fetching or parsing the URL: {url}, error: {e}")
        return ''

# Get user input
article_url = input("Enter the URL of the article: ").strip()
article_title = input("Enter the title of the article: ").strip()

# Extract domain from the URL
article_domain = extract_domain(article_url)

# Fetch content from the URL (if possible)
page_text = fetch_page_text(article_url)

# If no content is found, treat it as an issue
if not page_text:
    print("Could not fetch valid content from the URL. Please check the URL and try again.")
else:
    # Vectorize input text (title and page content) and domain
    X_text_vectorized = text_vectorizer.transform([article_title + " " + page_text])
    X_domain_vectorized = domain_vectorizer.transform([article_domain])

    # Combine features
    X_combined = hstack([X_text_vectorized, X_domain_vectorized])

    # Make prediction
    prediction = model.predict(X_combined)[0]

    # Display result
    if prediction == 1:
        print("The article is REAL.")
    else:
        print("The article is FAKE.")
