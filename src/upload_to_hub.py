"""
src/upload_to_hub.py
Upload du modèle et du vectoriseur sur Hugging Face Hub.
"""

import os
import pickle

from huggingface_hub import HfApi, login

HF_TOKEN = os.environ["HF_TOKEN"]
HF_REPO = os.environ["HF_REPO"]
MODEL_DIR = os.environ.get("MODEL_DIR", "../models")

# Connexion à Hugging Face
login(token=HF_TOKEN)
api = HfApi()

# Crée le repo s'il n'existe pas encore
api.create_repo(
    repo_id=HF_REPO,
    repo_type="model",
    exist_ok=True,
)
print(f"Repo prêt : https://huggingface.co/{HF_REPO}")

# Upload model.pkl et vectorizer.pkl
for filename in ["model.pkl", "vectorizer.pkl"]:
    local_path = os.path.join(MODEL_DIR, filename)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=filename,
        repo_id=HF_REPO,
        repo_type="model",
        commit_message=f"CI/CD: update {filename}",
    )
    print(f"✅ {filename} uploadé sur {HF_REPO}")

print(f"\nModèle disponible sur : https://huggingface.co/{HF_REPO}")
