import os
import pickle
import string

import gradio as gr
import nltk
import unidecode

MODEL_DIR = os.environ.get("MODEL_DIR", "models")


def build_preprocess():
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


def load_model_and_vectorizer(model_dir: str):
    model_path = os.path.join(model_dir, "model.pkl")
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
    model = pickle.load(open(model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))
    return model, vectorizer


def create_demo():
    preprocess = build_preprocess()
    model, vectorizer = load_model_and_vectorizer(MODEL_DIR)

    def predict(review_text: str):
        clean = preprocess(review_text)
        X = vectorizer.transform([clean])
        pred = model.predict(X)[0]

        proba = None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)[0]
            proba = {"negative": float(p[0]), "positive": float(p[1])}

        label = "positive" if float(pred) == 1.0 else "negative"
        return label, proba

    return gr.Interface(
        fn=predict,
        inputs=gr.Textbox(lines=6, label="Review text (pt-BR)"),
        outputs=[
            gr.Label(label="Prediction"),
            gr.JSON(label="Probabilities"),
        ],
        title="B2W Sentiment Analysis (PT-BR)",
        description="Logistic Regression + TF‑IDF (1-2 grams) avec preprocessing NLTK (RSLP).",
        allow_flagging="never",
    )


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
