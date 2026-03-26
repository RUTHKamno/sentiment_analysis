"""
src/update_space.py
Déploie l'interface Gradio sur Hugging Face Spaces.
Le modèle n'est PAS uploadé ici — l'app le télécharge
elle-même depuis HF Hub au démarrage via hf_hub_download.
"""

import os

from huggingface_hub import HfApi, login

HF_TOKEN = os.environ["HF_TOKEN"]
HF_SPACE_REPO = os.environ["HF_SPACE_REPO"]  # ex: kamcheruth/b2w_sentiment_analysis_app
HF_REPO = os.environ["HF_REPO"]  # ex: kamcheruth/b2w_sentiment_analysis

login(token=HF_TOKEN)
api = HfApi()

# Crée le Space s'il n'existe pas encore
api.create_repo(
    repo_id=HF_SPACE_REPO,
    repo_type="space",
    space_sdk="gradio",
    exist_ok=True,
)
print(f"Space prêt : https://huggingface.co/spaces/{HF_SPACE_REPO}")

# Définir HF_REPO comme variable d'environnement du Space
# (utilisée par app.py pour savoir où télécharger le modèle)
api.add_space_variable(
    repo_id=HF_SPACE_REPO,
    key="HF_REPO",
    value=HF_REPO,
)
print(f"Variable HF_REPO={HF_REPO} configurée sur le Space")

# Uploader uniquement app.py et requirements.txt
# (pas de model.pkl ni vectorizer.pkl — l'app les télécharge depuis HF Hub)
files = {
    "app/app.py": "app.py",
    "requirements.txt": "requirements.txt",
}

for local_path, remote_path in files.items():
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=remote_path,
        repo_id=HF_SPACE_REPO,
        repo_type="space",
        commit_message=f"CI/CD: update {remote_path}",
    )
    print(f"✅ {local_path} → {remote_path}")

print(f"\nSpace disponible sur : https://huggingface.co/spaces/{HF_SPACE_REPO}")
