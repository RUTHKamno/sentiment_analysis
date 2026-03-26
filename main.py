import subprocess
import sys

def main():
    """Lance l'application Gradio"""
    print("🚀 Démarrage de l'application Gradio...")
    try:
        subprocess.run(
            [sys.executable, "-m", "gradio", "app/app.py"],
            cwd="."
        )
    except Exception as e:
        print(f"Erreur : {e}")
        print("Alternative : python app/app.py")

if __name__ == "__main__":
    main()
