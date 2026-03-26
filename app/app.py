import os
import pickle
import string

import gradio as gr
import nltk
import unidecode
from huggingface_hub import hf_hub_download

# Configuration pour le téléchargement depuis Hugging Face Hub
HF_REPO = os.environ.get("HF_REPO", "DefaultUser/sentiment-model")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")

def build_preprocess():
    """Préparation du préprocessing NLTK de la branche develop."""
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


def get_model_and_vectorizer():
    """Charge le modèle depuis HF Hub ou utilise le fallback local."""
    print(f"Tentative de téléchargement depuis {HF_REPO}...")
    try:
        model_path = hf_hub_download(repo_id=HF_REPO, filename="model.pkl")
        vectorizer_path = hf_hub_download(repo_id=HF_REPO, filename="vectorizer.pkl")
    except Exception as e:
        print(f"Erreur de téléchargement HF ({e}). Utilisation du fallback local.")
        model_path = os.path.join(MODEL_DIR, "model.pkl")
        vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")

    print(f"Chargement depuis: {model_path} et {vectorizer_path}")
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model, vectorizer


def create_demo():
    preprocess = build_preprocess()
    model, vectorizer = get_model_and_vectorizer()

    def predict(text: str):
        if not text.strip():
            return "⚠️ Veuillez entrer un texte.", None

        clean = preprocess(text)
        X = vectorizer.transform([clean])
        pred = model.predict(X)[0]

        proba_dict = None
        score = 0
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)[0]
            proba_dict = {"Négatif": float(p[0]), "Positif": float(p[1])}
            score = round(max(p) * 100, 1)

        label = "🟢 Positif" if float(pred) == 1.0 else "🔴 Négatif"
        return f"**{label}** — {score}% de confiance", proba_dict

    return gr.Interface(
        fn=predict,
        inputs=gr.Textbox(
            lines=4,
            placeholder="Ex: Super produto! Funciona perfeitamente.",
            label="Avis client (Portugais)",
        ),
        outputs=[
            gr.Markdown(label="Sentiment détecté"),
            gr.JSON(label="Probabilités (Détails)"),
        ],
        title="B2W Sentiment Analysis (PT-BR)",
        description="Logistic Regression + TF-IDF avec NLTK sur les avis B2W.",
        examples=[
            ["O smartphone é fantástico e a bateria dura muito!"],
            ["A entrega atrasou e o produto veio com defeito. Horrível."],
            ["Produto normal, atende as expectativas, mas não surpreende."],
        ],
        theme=gr.themes.Soft(),
        allow_flagging="never",
    )


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
