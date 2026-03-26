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