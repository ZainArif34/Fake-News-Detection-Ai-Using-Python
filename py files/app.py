from flask import Flask, render_template, request, jsonify
import joblib
from urllib.parse import urlparse
import os

app = Flask(__name__)

# Load the trained model and vectorizers
model = joblib.load(os.path.join("pkl files", "model.pkl"))
text_vectorizer = joblib.load(os.path.join("pkl files", "vectorizer.pkl"))
domain_vectorizer = joblib.load(os.path.join("pkl files", "domain_vectorizer.pkl"))

# Function to extract domain from URL
def extract_domain(url):
    try:
        parsed_url = urlparse(url)
        return parsed_url.netloc  # Extract domain
    except:
        return "unknown_domain"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        url = request.form.get("url", "").strip()

        if not title or not url:
            return render_template("index.html", error="Both Title and URL are required!")

        domain = extract_domain(url)  # Extract domain from the URL

        # Vectorize the inputs
        title_vectorized = text_vectorizer.transform([title])
        domain_vectorized = domain_vectorizer.transform([domain])

        # Combine features
        from scipy.sparse import hstack
        X_combined = hstack([title_vectorized, domain_vectorized])

        # Predict
        prediction = model.predict(X_combined)[0]
        label = "Real News ✅" if prediction == 1 else "Fake News ❌"

        return render_template("index.html", title=title, url=url, result=label)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
