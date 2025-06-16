
# SMART HEALTH NEWS CLASSIFIER
# Classifies news as authentic or misleading
# Incorporates content filtering and medical policy checks


import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Required resources
nltk.download("stopwords")
nltk.download("wordnet")

# DATA COLLECTION

articles = {
    'headline': [
        # Genuine articles
        "CDC approves new flu vaccine amid rising cases",
        "New study links smoking to lung cancer increase",
        "FDA issues alert on unsafe food supplement",
        "Exercise reduces risk of stroke in elderly, study finds",
        "Hospitals report success with new cancer therapy",
        
        # Misleading articles
        "5G signals are causing mental illness, claims study",
        "Homeopathy proven to cure cancer by secret research",
        "Drinking silver water eliminates all viruses",
        "New miracle oil cures diabetes overnight",
        "Vaccines contain microchips, insider reveals"
    ],
    'tag': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = real, 1 = fake
}

news_df = pd.DataFrame(articles)

#TEXT SANITIZATION

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|[^a-z\s]", "", text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    cleaned = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(cleaned)

news_df["processed"] = news_df["headline"].apply(clean_text)

#MODEL PREPARATION

vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X = vectorizer.fit_transform(news_df["processed"])
y = news_df["tag"]

classifier = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced", C=0.7)
classifier.fit(X, y)

# CLASSIFICATION FUNCTION

def classify_article(article_text):
    normalized = clean_text(article_text)
    vector_input = vectorizer.transform([normalized])
    prediction = classifier.predict(vector_input)[0]
    confidence_score = classifier.predict_proba(vector_input)[0][prediction]
    
    # Health policy red flag check
    health_flags = {
        'silver': {'triggers': ['virus', 'cure'], 'source': 'FDA'},
        'vaccine': {'triggers': ['chip', 'tracking'], 'source': 'CDC'},
        'homeopathy': {'triggers': ['cancer'], 'source': 'WHO'}
    }

    reason_notes = []
    for keyword, rule in health_flags.items():
        if keyword in normalized:
            if any(t in normalized for t in rule['triggers']):
                prediction = 1
                confidence_score = max(confidence_score, 0.97)
                reason_notes.append(f"Conflicts with {rule['source']} guidelines")
            else:
                prediction = 0
                confidence_score = max(confidence_score, 0.97)
                reason_notes.append(f"Supports {rule['source']} publications")

    important_terms = [w for w in normalized.split() if w in vectorizer.get_feature_names_out()]

    return {
        "original_text": article_text,
        "result": "Fake" if prediction else "Real",
        "confidence": f"{confidence_score:.0%}",
        "insight": " | ".join(reason_notes) if reason_notes else "Standard linguistic classification",
        "terms": important_terms[:5]
    }

#  USER-DRIVEN INTERACTION

def start_classifier():
    print("🧠 HEALTH NEWS VERIFIER")
    print("-" * 50)
    print("Type a health-related news headline below.")
    print("Enter 'exit' to stop the program.")
    
    while True:
        headline = input("\n🗞️  Headline: ")
        if headline.strip().lower() == "exit":
            print("\n🔚 Session ended.")
            break
        result = classify_article(headline)
        print("\n" + "-" * 50)
        print(f"📰 HEADLINE: {result['original_text']}")
        print(f"✅ CLASSIFICATION: {result['result']} ({result['confidence']} confident)")
        print(f"💬 INSIGHT: {result['insight']}")
        print(f"🔎 TERMS IDENTIFIED: {', '.join(result['terms'])}")
        print("-" * 50)

# Correct entry point
if __name__ == "__main__":
    start_classifier()
