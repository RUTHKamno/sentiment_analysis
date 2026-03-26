"""
app/app.py
Interface Gradio — télécharge le modèle depuis Hugging Face Hub au démarrage.
"""

import os
import pickle
import string

import gradio as gr
import nltk
import unidecode
from huggingface_hub import hf_hub_download

# ──────────────────────────────────────────────
# Téléchargement du modèle depuis HF Hub
# La variable HF_REPO est définie dans les secrets du Space HF
# ──────────────────────────────────────────────
HF_REPO = os.environ.get("HF_REPO", "kamcheruth/b2w_sentiment_analysis")

model_path      = hf_hub_download(repo_id=HF_REPO, filename="model.pkl")
vectorizer_path = hf_hub_download(repo_id=HF_REPO, filename="vectorizer.pkl")

model      = pickle.load(open(model_path,      "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

print(f"Modèle chargé depuis : {HF_REPO}")

# ──────────────────────────────────────────────
# Pré-traitement (identique à src/train.py)
# ──────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("rslp",      quiet=True)

puncts         = list(string.punctuation)
stopwords_list = list(set([
    unidecode.unidecode(sw)
    for sw in nltk.corpus.stopwords.words("portuguese")
]))
stopwords_puncts = set(stopwords_list + puncts)


def preprocess(text: str) -> str:
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    stemmer   = nltk.RSLPStemmer()
    tokens = tokenizer.tokenize(text)
    tokens = [unidecode.unidecode(t.lower()) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords_puncts]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


# ──────────────────────────────────────────────
# Fonction de prédiction
# ──────────────────────────────────────────────
def predict(text: str):
    if not text.strip():
        return "Veuillez entrer un avis.", {}
    clean  = preprocess(text)
    X      = vectorizer.transform([clean])
    proba  = model.predict_proba(X)[0]
    label  = model.predict(X)[0]
    result = "Positif ✅" if label == 1.0 else "Négatif ❌"
    scores = {"Positif": float(proba[1]), "Négatif": float(proba[0])}
    return result, scores


# ──────────────────────────────────────────────
# Interface Gradio
# ──────────────────────────────────────────────
with gr.Blocks(title="B2W Sentiment Analysis") as demo:
    gr.Markdown("# 🇧🇷 B2W Sentiment Analysis")
    gr.Markdown("Entrez un avis produit en **portugais brésilien** pour analyser son sentiment.")

    text_input = gr.Textbox(
        label="Avis produit",
        placeholder="Ex: Produto excelente, chegou rápido!",
        lines=3,
    )

    btn = gr.Button("Analyser", variant="primary")

    with gr.Row():
        label_output = gr.Label(label="Sentiment")
        proba_output = gr.Label(label="Probabilités")

    btn.click(fn=predict, inputs=text_input, outputs=[label_output, proba_output])

    gr.Examples(
        examples=[
            ["Produto excelente, recomendo para todos!"],
            ["Péssima qualidade, quebrou em dois dias."],
            ["Entrega rápida, produto conforme descrito."],
            ["Não gostei, veio com defeito."],
        ],
        inputs=text_input,
    )

if __name__ == "__main__":
    demo.launch()
