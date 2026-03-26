# B2W Sentiment Analysis — Pipeline MLOpsMaster 2 IABD — Projet Final Git/GitHubÉquipe : 

projet qui implémente une pipeline complète d'analyse de sentiments (NLP) sur les avis produits, intégrant les pratiques MLOps : automatisation CI/CD, tests de performance, et déploiement continu sur Hugging Face.

# 1. Structure du Projet.
├── .github/workflows/  # Pipelines CI/CD (GitHub Actions)
├── app/                # Interface utilisateur (Gradio)
├── data/               # Dataset (Suivi via Git LFS)
├── hooks/              # Scripts de pre-commit partagés
├── models/             # Modèles et Vectoriseurs sauvegardés (LFS)
├── src/                # Scripts d'entraînement et utilitaires (train.py, etc.)
├── tests/              # Tests unitaires et validation (Pytest)
├── .gitattributes      # Configuration Git LFS
├── .gitignore          # Fichiers exclus du suivi Git
└── requirements.txt    # Dépendances du projet

# 2. Installation et Configuration
# 2.1 PrérequisPython 3.10  +  Git LFS (Indispensable pour les fichiers > 5Mo) UV ou Pip (Gestionnaire de paquets)
# 2.2 Installation de l'environnement

# 3. Cloner le dépôt
git clone [https://github.com/kamcheruth/b2w-sentiment-analysis.git](https://github.com/kamcheruth/b2w-sentiment-analysis.git)
cd b2w-sentiment-analysis

# 4. Installer Git LFS sur votre machine (une seule fois)
git lfs install
git lfs pull

# 5. Créer l'environnement et installer les dépendances
pip install -r requirements.txt

# 5. Télécharger les ressources NLP obligatoires
python -m nltk.downloader stopwords wordnet

# 5.1 Activation des Git Hooks (IMPORTANT)Pour garantir que tout le monde respecte les normes de code (Flake8, Black) avant de commit, lancez cette commande :

# Copier et activer le script de vérification
cp hooks/pre-commit.sh .git/hooks/pre-commit

# 6. Cycle de Développement

## Étape 1 : 
EntraînementLe script src/train.py nettoie les données, entraîne un modèle de classification et sauvegarde les artefacts dans le dossier models/.python src/train.py

## Étape 2 : 
Tests LocauxAvant de pousser votre code (Push), vérifiez que l'accuracy est toujours > 80% :pytest tests/test_model.py -v

## Étape 3 : 
Workflow Git (Git Flow)Créez une branche : git checkout -b feature/nom-de-votre-tacheTravaillez et commitez (les hooks vérifieront votre syntaxe automatiquement).Poussez vers GitHub : git push origin feature/nom-de-votre-tacheOuvrez une Pull Request vers la branche develop.

# 6. Automatisation CI/CD (Rôle DevOps)À chaque Pull Request, GitHub Actions lance automatiquement :

Linter : Vérification de la syntaxe avec flake8.
Tests : Validation de l'accuracy du modèle avec pytest.
Notification : Un email est envoyé à l'équipe avec le rapport de validation.
Déploiement : Si fusionné sur main, le modèle est poussé sur Hugging Face Hub et l'interface est mise à jour sur Hugging Face Spaces.

# 7. Sécurité et SecretsNe commitez jamais de fichiers .env ou .secrets.Les tokens HF_TOKEN et les identifiants SMTP doivent être configurés dans les GitHub Secrets du dépôt.

# 8. Interface de Démo Une fois le déploiement réussi, l'interface est accessible 

Note aux collaborateurs : Merci de toujours vérifier que vos fichiers CSV et PKL sont bien suivis par LFS via la commande git lfs status avant de push.