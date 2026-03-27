# 🎭 B2W Sentiment Analysis — Pipeline MLOps

**Master 2 IABD — Projet Final Git/GitHub**

> Pipeline complète d'analyse de sentiments (NLP) sur des avis produits, intégrant les pratiques MLOps : automatisation CI/CD, tests de performance, et déploiement continu sur Hugging Face.

---

## 📁 1. Structure du Projet

```
├── .github/workflows/   # Pipelines CI/CD (GitHub Actions)
├── app/                 # Interface utilisateur (Gradio)
├── data/                # Dataset (Suivi via Git LFS)
├── hooks/               # Scripts de pre-commit partagés
├── models/              # Modèles et Vectoriseurs sauvegardés (LFS)
├── src/                 # Scripts d'entraînement et utilitaires (train.py, etc.)
├── tests/               # Tests unitaires et validation (Pytest)
├── .gitattributes       # Configuration Git LFS
├── .gitignore           # Fichiers exclus du suivi Git
└── requirements.txt     # Dépendances du projet
```

---

## ⚙️ 2. Installation et Configuration

### 2.1 Prérequis

| Outil | Version minimale | Remarque |
|---|---|---|
| Python | 3.10+ | [python.org](https://www.python.org/downloads/) |
| Git | Récente | [git-scm.com](https://git-scm.com/) |
| Git LFS | Récente | **Indispensable** pour les fichiers > 5 Mo |
| pip | Inclus avec Python | Gestionnaire de paquets |

---

### 2.2 Cloner le dépôt

```bash
git clone https://github.com/kamcheruth/b2w-sentiment-analysis.git
cd b2w-sentiment-analysis
```

---

### 2.3 Installer Git LFS et récupérer les fichiers volumineux

> ⚠️ **Git LFS est obligatoire.** Sans lui, les fichiers `.pkl`, `.csv` et les modèles ne seront pas téléchargés correctement.

#### 🐧 Linux

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install git-lfs -y

# Arch Linux
sudo pacman -S git-lfs

# Fedora / RHEL
sudo dnf install git-lfs

# Activer LFS et récupérer les fichiers
git lfs install
git lfs pull
```

#### 🪟 Windows

```powershell
# Option 1 — Via winget (Windows 10/11)
winget install Git.GitLFS

# Option 2 — Via Chocolatey
choco install git-lfs

# Option 3 — Téléchargement manuel
# Aller sur https://git-lfs.com et installer le binaire

# Activer LFS et récupérer les fichiers (PowerShell ou Git Bash)
git lfs install
git lfs pull
```

---

### 2.4 Créer l'environnement virtuel et installer les dépendances

#### 🐧 Linux

```bash
# Créer l'environnement virtuel
python3 -m venv .venv

# Activer l'environnement
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

#### 🪟 Windows

```powershell
# Créer l'environnement virtuel
python -m venv .venv

# Activer l'environnement (PowerShell)
.venv\Scripts\Activate.ps1

# Si l'exécution de scripts est bloquée, exécuter d'abord :
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Activer l'environnement (CMD)
.venv\Scripts\activate.bat

# Installer les dépendances
pip install -r requirements.txt
```

---

### 2.5 Télécharger les ressources NLP obligatoires

> Même commande sur Linux et Windows (dans l'environnement virtuel activé) :

```bash
python -m nltk.downloader stopwords wordnet
```

---

### 2.6 Activation des Git Hooks ⚠️ IMPORTANT

Les hooks garantissent que tout le monde respecte les normes de code (Flake8, Black) avant chaque commit.

#### 🐧 Linux

```bash
# Copier le script et lui donner les droits d'exécution
cp hooks/pre-commit.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

#### 🪟 Windows

```powershell
# Via PowerShell
Copy-Item hooks\pre-commit.sh .git\hooks\pre-commit

# Via Git Bash (recommandé pour les hooks shell)
cp hooks/pre-commit.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

> 💡 **Conseil Windows** : Il est recommandé d'utiliser **Git Bash** (inclus avec Git for Windows) pour exécuter les scripts shell `.sh`.

---

## 🔄 3. Cycle de Développement

### Étape 1 — Entraînement du modèle

Le script `src/train.py` nettoie les données, entraîne un modèle de classification et sauvegarde les artefacts dans le dossier `models/`.

```bash
python src/train.py
```

### Étape 2 — Tests locaux

Avant de pousser votre code, vérifiez que l'accuracy est toujours > 80% :

```bash
pytest tests/test_model.py -v
```

### Étape 3 — Workflow Git (Git Flow)

```bash
# 1. Créer une branche de fonctionnalité
git checkout -b feature/nom-de-votre-tache

# 2. Travailler et committer
#    (les hooks vérifient votre syntaxe automatiquement)
git add .
git commit -m "feat: description de votre modification"

# 3. Pousser vers GitHub
git push origin feature/nom-de-votre-tache

# 4. Ouvrir une Pull Request vers la branche develop sur GitHub
```

---

## 🤖 4. Automatisation CI/CD

À chaque Pull Request, **GitHub Actions** lance automatiquement :

| Étape | Outil | Description |
|---|---|---|
| 🔍 Linter | `flake8` | Vérification de la syntaxe |
| ✅ Tests | `pytest` | Validation de l'accuracy du modèle |
| 📧 Notification | SMTP | Email de rapport envoyé à l'équipe |
| 🚀 Déploiement | Hugging Face Hub | Déclenché uniquement lors d'une fusion sur `main` |

---

## 🔐 5. Sécurité et Secrets

- ❌ Ne commitez **jamais** de fichiers `.env` ou `.secrets`
- ✅ Les tokens `HF_TOKEN` et les identifiants SMTP doivent être configurés dans les **GitHub Secrets** du dépôt
- ✅ Vérifiez que `.gitignore` contient bien `.env` avant tout push

---

## 🌐 6. Interface de Démo

Une fois le déploiement réussi sur `main`, l'interface Gradio est accessible sur **Hugging Face Spaces**.

---

## 📝 Note aux collaborateurs

> Merci de toujours vérifier que vos fichiers `.csv` et `.pkl` sont bien suivis par LFS **avant de push** :

```bash
git lfs status
```

Si un fichier volumineux n'apparaît pas dans cette liste, ajoutez-le manuellement :

```bash
git lfs track "*.pkl"
git lfs track "*.csv"
git add .gitattributes
```
