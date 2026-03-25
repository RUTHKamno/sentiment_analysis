"""
tests/test_model.py

- Charge le modèle depuis models/  (pas de réentraînement)
- Relit data/buscape.csv et refait le même split que train.py
  pour obtenir X_test / y_test
"""

import os
import pickle
import string

import nltk
import unidecode
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score

# ──────────────────────────────────────────────
# CHEMINS
# ──────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.environ.get("DATA_PATH", os.path.join(ROOT, "data", "b2w_test.csv"))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(ROOT, "models"))

# ──────────────────────────────────────────────
# CHARGEMENT DU MODÈLE ET DU VECTORISEUR
# ──────────────────────────────────────────────
model = pickle.load(open(os.path.join(MODEL_DIR, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb"))

# ──────────────────────────────────────────────
# RECONSTRUCTION DU JEU DE TEST
# Même pipeline que train.py pour obtenir le même X_test / y_test
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


dataset = pd.read_csv(DATA_PATH)
dataset["polarity_label"] = dataset["polarity"].map({0.0: "negative", 1.0: "positive"})

positive = dataset[dataset["polarity_label"] == "positive"].sample(
    28589, random_state=42
)
negative = dataset[dataset["polarity_label"] == "negative"]
dataset = pd.concat([positive, negative]).reset_index(drop=True)

dataset["text_clean"] = dataset["review_text"].apply(preprocess)

X = vectorizer.transform(dataset["text_clean"])  # transform uniquement, pas fit
y = dataset["polarity"]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    train_size=0.7,
    test_size=0.3,
    stratify=y,
    random_state=42,  # même seed → même split que train.py
)

# ──────────────────────────────────────────────
# TESTS
# ──────────────────────────────────────────────


class TestModele:

    def test_accuracy_superieure_a_80_pourcent(self):
        accuracy = accuracy_score(y_test, model.predict(X_test))
        assert (
            accuracy >= 0.80
        ), f"Accuracy trop basse : {accuracy:.4f} < 0.80 — déploiement bloqué."

    def test_precision_negative_superieure_a_75_pourcent(self):
        precision = precision_score(y_test, model.predict(X_test), pos_label=0.0)
        assert (
            precision >= 0.75
        ), f"Précision (négatif) trop basse : {precision:.4f} < 0.75"

    def test_precision_positive_superieure_a_75_pourcent(self):
        precision = precision_score(y_test, model.predict(X_test), pos_label=1.0)
        assert (
            precision >= 0.75
        ), f"Précision (positif) trop basse : {precision:.4f} < 0.75"
