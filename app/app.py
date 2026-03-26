<<<<<<< develop
import os
import pickle
import string

import gradio as gr
import nltk
import unidecode

MODEL_DIR = os.environ.get("MODEL_DIR", "models")


def build_preprocess():
    nltk.download("stopwords", quiet=True)
    nltk.download("rslp", quiet=True)

    puncts = list(string.punctuation)
    stopwords_list = list(
        set(
            [
                unidecode.unidecode(sw)
                for sw in nltk.corpus.stopwords.words("portuguese")
            ]
        )
    )
    stopwords_puncts = set(stopwords_list + puncts)

    tokenizer = nltk.tokenize.WordPunctTokenizer()
    stemmer = nltk.RSLPStemmer()

    def preprocess(text: str) -> str:
        if not isinstance(text, str) or text.strip() == "":
            return ""
        tokens = tokenizer.tokenize(text)
        tokens = [unidecode.unidecode(t.lower()) for t in tokens]
        tokens = [t for t in tokens if t.strip() and t not in stopwords_puncts]
        tokens = [stemmer.stem(t) for t in tokens if len(t) > 0]
        return " ".join(tokens)

    return preprocess


def load_model_and_vectorizer(model_dir: str):
    model_path = os.path.join(model_dir, "model.pkl")
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
    model = pickle.load(open(model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))
    return model, vectorizer


def create_demo():
    preprocess = build_preprocess()
    model, vectorizer = load_model_and_vectorizer(MODEL_DIR)

    def predict(review_text: str):
        clean = preprocess(review_text)
        X = vectorizer.transform([clean])
        pred = model.predict(X)[0]

        proba = None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)[0]
            proba = {"negative": float(p[0]), "positive": float(p[1])}

        label = "positive" if float(pred) == 1.0 else "negative"
        return label, proba

    return gr.Interface(
        fn=predict,
        inputs=gr.Textbox(lines=6, label="Review text (pt-BR)"),
        outputs=[
            gr.Label(label="Prediction"),
            gr.JSON(label="Probabilities"),
        ],
        title="B2W Sentiment Analysis (PT-BR)",
        description="Logistic Regression + TF‑IDF (1-2 grams) avec preprocessing NLTK (RSLP).",
        allow_flagging="never",
    )


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
=======
import gradio as gr
import pickle
import os
from huggingface_hub import hf_hub_download

# Configuration
HF_REPO = os.environ.get("HF_REPO", "DefaultUser/sentiment-model")
MODEL_FILES = ["model.pkl", "vectorizer.pkl"]

# Télécharger le modèle et vectorizer depuis HF Hub
print(f"Téléchargement depuis {HF_REPO}...")
try:
    model_path = hf_hub_download(repo_id=HF_REPO, filename="model.pkl")
    vectorizer_path = hf_hub_download(repo_id=HF_REPO, filename="vectorizer.pkl")
except Exception as e:
    print(f"Erreur : {e}")
    # Fallback local (pour tests)
    model_path = "models/model.pkl"
    vectorizer_path = "models/vectorizer.pkl"

print("Chargement du vectorizer...")
with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

print("Chargement du modèle...")
with open(model_path, "rb") as f:
    model = pickle.load(f)

def predict(text):
    if not text.strip():
        return "⚠️ Veuillez entrer un texte."

    # Vectorisation du texte puis prédiction
    X = vectorizer.transform([text])
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    score = round(max(proba) * 100, 1)

    label = "🟢 Positif" if pred == 1 else "🔴 Négatif"
    return f"**{label}** — {score}% de confiance"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Ex: Ce produit est vraiment excellent !",
        label="Avis sur un produit tech"
    ),
    outputs=gr.Markdown(label="Sentiment détecté"),
    title="🎯 Analyse de Sentiments — Produits Tech",
    description="Entrez un avis client. Le modèle détecte s'il est positif ou négatif.",
    examples=[
        ["Ce smartphone est absolument fantastique !"],
        ["Très déçu, produit en panne après 2 jours."],
        ["Correct sans plus, rapport qualité-prix moyen."],
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()
>>>>>>> main
