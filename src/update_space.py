from huggingface_hub import HfApi, login
import os

def update_space():
    """Mise à jour du Space Gradio sur Hugging Face"""
    
    # Configuration
    hf_token = os.environ.get("HF_TOKEN")
    space_repo = os.environ.get("HF_SPACE_REPO")  # ex: username/sentiment-space
    
    if not hf_token or not space_repo:
        raise ValueError("HF_TOKEN et HF_SPACE_REPO doivent être définis")
    
    print(f"🚀 Mise à jour du Space : {space_repo}")
    
    # Authentification
    login(token=hf_token)
    api = HfApi(token=hf_token)
    
    # Créer le Space s'il n'existe pas
    api.create_repo(
        repo_id=space_repo,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True
    )
    print(f"  ✅ Space créé/existant")
    
    # Upload app.py depuis app/app.py
    print(f"  ⬆️  Upload de app.py...")
    api.upload_file(
        path_or_fileobj="app/app.py",
        path_in_repo="app.py",
        repo_id=space_repo,
        repo_type="space",
        commit_message="[CI/CD] Update app.py"
    )
    print(f"  ✅ app.py uploadé")
    
    # Upload requirements.txt
    print(f"  ⬆️  Upload de requirements.txt...")
    api.upload_file(
        path_or_fileobj="requirements.txt",
        path_in_repo="requirements.txt",
        repo_id=space_repo,
        repo_type="space",
        commit_message="[CI/CD] Update requirements.txt"
    )
    print(f"  ✅ requirements.txt uploadé")
    
    print(f"\n✅ Space mis à jour : https://huggingface.co/spaces/{space_repo}")

if __name__ == "__main__":
    update_space()