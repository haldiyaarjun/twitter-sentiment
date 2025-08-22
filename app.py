# app.py
import os
import pathlib
import numpy as np
import streamlit as st
import joblib

MODEL_PATH = pathlib.Path("models/baseline_pipeline.joblib")

st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="🐦")

st.title("🐦 Twitter Sentiment Analysis")
st.caption("TF-IDF + Logistic Regression (baseline)")

# ---------- Helpers ----------
@st.cache_resource
def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. "
            "Make sure you trained and saved the pipeline, or added it to the repo."
        )
    return joblib.load(MODEL_PATH)

def predict_label_and_conf(pipe, texts):
    """
    Returns labels (0/1) and a confidence score per text.
    Confidence is probability for the predicted class if available,
    otherwise a sigmoid of the decision_function margin (for ranking only).
    """
    labels = pipe.predict(texts)

    # Try probabilities if the classifier supports it
    clf = pipe.named_steps.get("clf", None)
    vec = pipe.named_steps.get("tfidf", None)
    conf = np.full(len(texts), np.nan, dtype=float)

    try:
        if hasattr(clf, "predict_proba"):
            X = vec.transform(texts)
            proba = clf.predict_proba(X)
            conf = proba[np.arange(len(texts)), labels]  # prob of predicted class
        elif hasattr(clf, "decision_function"):
            X = vec.transform(texts)
            margin = clf.decision_function(X)
            if margin.ndim == 1:
                # map signed distance -> (0,1) via sigmoid of absolute margin
                conf = 1.0 / (1.0 + np.exp(-np.abs(margin)))
            else:
                # multiclass: take margin of predicted class
                chosen = margin[np.arange(len(texts)), labels]
                conf = 1.0 / (1.0 + np.exp(-np.abs(chosen)))
    except Exception:
        # If anything goes wrong, keep NaNs; we can still show labels
        pass

    return labels, conf

def format_label(y):
    return "Positive" if int(y) == 1 else "Negative"

# ---------- Load model ----------
try:
    pipe = load_pipeline()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# ---------- UI ----------
tab_single, tab_batch = st.tabs(["Single input", "Batch input"])

with tab_single:
    text = st.text_area(
        "Enter a tweet or any short text:",
        placeholder="e.g., I love this phone! The battery lasts forever.",
        height=120,
    )
    if st.button("Analyze sentiment", type="primary"):
        t = text.strip()
        if not t:
            st.warning("Please enter some text.")
        else:
            labels, conf = predict_label_and_conf(pipe, [t])
            label = format_label(labels[0])
            st.markdown(f"### Prediction: **{label}**")
            if not np.isnan(conf[0]):
                st.caption(f"Confidence: {conf[0]:.2f}")

with tab_batch:
    st.write("Paste multiple lines below (one sentence per line).")
    bulk = st.text_area(
        "Batch texts",
        placeholder="I love this!\nThis update is terrible.\nNot bad at all.",
        height=180,
        key="bulk",
    )
    if st.button("Analyze batch"):
        lines = [ln.strip() for ln in bulk.splitlines() if ln.strip()]
        if not lines:
            st.warning("Please add at least one non-empty line.")
        else:
            labels, conf = predict_label_and_conf(pipe, lines)
            for i, line in enumerate(lines):
                label = format_label(labels[i])
                has_conf = not np.isnan(conf[i])
                conf_str = f" — {conf[i]:.2f}" if has_conf else ""
                st.write(f"- **{label}**{conf_str}: {line}")

st.divider()
with st.expander("About this app"):
    st.markdown(
        """
        **How it works:** This app loads a scikit-learn **Pipeline** that includes
        a TF-IDF vectorizer and a Logistic Regression classifier. Your input text is
        vectorized and classified as **Positive (1)** or **Negative (0)**.

        - If available, we display a **confidence** (predicted class probability or a scaled margin).
        - Trained on the Sentiment140 dataset (Positive/Negative).
        """
    )

