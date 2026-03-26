"""
src/update_space.py
Met à jour automatiquement un Hugging Face Space (Gradio) avec :
- app.py (depuis app/app.py)
- model.pkl + vectorizer.pkl (depuis models/)
"""

import os

from huggingface_hub import HfApi, login


def main():
    hf_token = os.environ["HF_TOKEN"]
    hf_space_repo = os.environ[
        "HF_SPACE_REPO"
    ]  # ex: kamcheruth/b2w_sentiment_analysis_app

    model_dir = os.environ.get("MODEL_DIR", "models")
    app_src_path = os.environ.get("APP_PATH", os.path.join("app", "app.py"))

    login(token=hf_token)
    api = HfApi()

    api.create_repo(
        repo_id=hf_space_repo,
        repo_type="space",
        exist_ok=True,
        space_sdk="gradio",
    )

    uploads = [
        (app_src_path, "app.py"),
        (os.path.join(model_dir, "model.pkl"), "model.pkl"),
        (os.path.join(model_dir, "vectorizer.pkl"), "vectorizer.pkl"),
    ]

    for local_path, path_in_repo in uploads:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=hf_space_repo,
            repo_type="space",
            commit_message=f"CI/CD: update {path_in_repo}",
        )

    print(f"✅ Space mis à jour : https://huggingface.co/spaces/{hf_space_repo}")


if __name__ == "__main__":
    main()
