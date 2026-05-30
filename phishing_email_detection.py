import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "smsspamcollection.zip")
DATA_FILE = os.path.join(DATA_DIR, "SMSSpamCollection")


def download_dataset():
    """Download the SMS Spam Collection dataset from UCI."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_FILE):
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(DATA_DIR)


def load_data():
    """Load and return the dataset as a DataFrame."""
    df = pd.read_csv(DATA_FILE, sep="\t", header=None, names=["label", "text"])
    df["label"] = df["label"].map({"spam": "Phishing", "ham": "Safe"})
    return df

def extract_features(df):
    """Extract additional email-style features from text."""
    df = df.copy()
    df["has_url"] = df["text"].apply(
        lambda x: int(bool(re.search(r"http[s]?://|www\.", x, re.IGNORECASE)))
    )
    df["has_urgent_words"] = df["text"].apply(
        lambda x: int(bool(re.search(
            r"\b(urgent|verify|confirm|account|login|click|free|winner|prize|password|bank|update)\b",
            x, re.IGNORECASE
        )))
    )
    df["text_length"] = df["text"].apply(len)
    df["num_digits"] = df["text"].apply(lambda x: sum(c.isdigit() for c in x))
    df["num_special_chars"] = df["text"].apply(
        lambda x: sum(not c.isalnum() and not c.isspace() for c in x)
    )
    return df


def preprocess_text(text):
    """Basic text cleaning."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_pipeline(classifier):
    """TF-IDF + classifier pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=preprocess_text,
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )),
        ("clf", classifier)
    ])


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train both models and return the best one."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    trained_pipelines = {}

    for name, clf in models.items():
        pipeline = build_pipeline(clf)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        trained_pipelines[name] = (pipeline, y_pred)

        print(f"\n{name}")
        print(f"Accuracy: {acc * 100:.2f}%")
        print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))

    best_name = max(results, key=results.get)
    print(f"Best Model: {best_name} ({results[best_name] * 100:.2f}% accuracy)")
    return best_name, trained_pipelines[best_name]

def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred, labels=["Safe", "Phishing"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe", "Phishing"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()


def plot_top_features(pipeline, top_n=20):
    """Plot top TF-IDF features for the Logistic Regression model."""
    try:
        tfidf = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["clf"]
        if not hasattr(clf, "coef_"):
            return

        feature_names = np.array(tfidf.get_feature_names_out())
        coef = clf.coef_[0]
        top_phishing_idx = np.argsort(coef)[-top_n:]
        top_safe_idx = np.argsort(coef)[:top_n]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Top Discriminating Words (TF-IDF Weights)", fontsize=14, fontweight="bold")

        for ax, idx, title, color in [
            (axes[0], top_phishing_idx, "Phishing Indicators", "salmon"),
            (axes[1], top_safe_idx[::-1], "Safe Indicators", "mediumseagreen"),
        ]:
            ax.barh(feature_names[idx], coef[idx], color=color)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("TF-IDF Coefficient")
            ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150)
        plt.show()
    except Exception as e:
        pass


def plot_label_distribution(df):
    """Bar chart of class distribution."""
    counts = df["label"].value_counts()
    colors = ["mediumseagreen", "salmon"]
    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", ax=ax, color=colors, edgecolor="black", width=0.5)
    ax.set_title("Dataset Label Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.set_xticklabels(counts.index, rotation=0)
    for i, v in enumerate(counts):
        ax.text(i, v + 20, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("label_distribution.png", dpi=150)
    plt.show()

SAMPLE_EMAILS = [
    "Congratulations! You have won a $1,000 gift card. Click http://win-now.com to claim your prize immediately!",
    "Hey, are we still on for lunch tomorrow? Let me know!",
    "URGENT: Your bank account has been compromised. Verify your password at http://secure-login.xyz NOW.",
    "Hi team, please find the meeting notes from today's standup attached.",
    "FREE entry to our lottery! Text WIN to 80080 to claim your free iPhone now!",
    "Your order #45231 has been shipped and will arrive by Thursday.",
]


def run_predictions(pipeline):
    """Run predictions on sample emails and display results."""
    for i, email in enumerate(SAMPLE_EMAILS, 1):
        prediction = pipeline.predict([email])[0]
        proba = pipeline.predict_proba([email])[0]
        phishing_conf = proba[list(pipeline.classes_).index("Phishing")] * 100
        print(f"\nEmail {i}: {email[:80]}{'...' if len(email) > 80 else ''}")
        print(f"  Prediction : {prediction}")
        print(f"  Confidence : {phishing_conf:.1f}% phishing probability")


def main():
    download_dataset()
    df = load_data()
    df = extract_features(df)

    plot_label_distribution(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    best_name, (best_pipeline, y_pred) = train_and_evaluate(X_train, X_test, y_train, y_test)

    plot_confusion_matrix(y_test, y_pred, best_name)
    plot_top_features(best_pipeline)

    run_predictions(best_pipeline)


if __name__ == "__main__":
    main()
