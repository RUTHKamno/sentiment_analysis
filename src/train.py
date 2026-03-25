"""
src/train.py
"""

import os
import pickle
import string

import nltk
import pandas as pd
import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_PATH = os.environ.get("DATA_PATH", "../data/b2w_train.csv")
MODEL_DIR = os.environ.get("MODEL_DIR", "../models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
dataset = pd.read_csv(DATA_PATH)
# ──────────────────────────────────────────────
# 2. CLEANING & BALANCING
# ──────────────────────────────────────────────
dataset["polarity_label"] = dataset["polarity"].map({0.0: "negative", 1.0: "positive"})

positive = dataset[dataset["polarity_label"] == "positive"].sample(
    28589, random_state=42
)
negative = dataset[dataset["polarity_label"] == "negative"]
dataset = pd.concat([positive, negative]).reset_index(drop=True)

# ──────────────────────────────────────────────
# 3. TEXT PRE-PROCESSING
# ──────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("rslp", quiet=True)

puncts = list(string.punctuation)
stopwords_list = list(
    set([unidecode.unidecode(sw) for sw in nltk.corpus.stopwords.words("portuguese")])
)
stopwords_puncts = set(stopwords_list + puncts)


def preprocess(text: str) -> str:
    # 1. Gestion des valeurs nulles (NaN)
    if not isinstance(text, str) or text.strip() == "":
        return ""

    tokenizer = nltk.tokenize.WordPunctTokenizer()
    stemmer = nltk.RSLPStemmer()

    tokens = tokenizer.tokenize(text)
    # Nettoyage (minuscules + suppression accents)
    tokens = [unidecode.unidecode(t.lower()) for t in tokens]

    # FILTRE CRUCIAL : On vérifie que 't' n'est pas vide ET pas dans les stopwords
    tokens = [t for t in tokens if t.strip() and t not in stopwords_puncts]

    # Stemming sécurisé
    tokens = [stemmer.stem(t) for t in tokens if len(t) > 0]

    return " ".join(tokens)


print("Pre-processing texts...")
dataset["text_clean"] = dataset["review_text"].apply(preprocess)

# ──────────────────────────────────────────────
# 4. VECTORIZATION
# ──────────────────────────────────────────────
vectorizer = TfidfVectorizer(lowercase=False, max_features=100, ngram_range=(1, 2))
X = vectorizer.fit_transform(dataset["text_clean"])
y = dataset["polarity"]

# ──────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.7,
    test_size=0.3,
    stratify=y,
    random_state=42,
)

# ──────────────────────────────────────────────
# 6. TRAINING
# ──────────────────────────────────────────────
print("Training model...")
model = LogisticRegression(solver="lbfgs", max_iter=1000)
model.fit(X_train, y_train)

# ──────────────────────────────────────────────
# 7. EVALUATION
# ──────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n===== EVALUATION =====")
print(f"Accuracy : {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ──────────────────────────────────────────────
# 8. SAVE MODEL
# ──────────────────────────────────────────────
pickle.dump(model, open(os.path.join(MODEL_DIR, "model.pkl"), "wb"))
pickle.dump(vectorizer, open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb"))
print(f"\nModel saved      → {MODEL_DIR}/model.pkl")
print(f"Vectorizer saved → {MODEL_DIR}/vectorizer.pkl")
