from __future__ import annotations

import logging
from typing import Any, Dict, List

from .utils import load_json, resolve_path


def predict_sentiment(config: Dict, text: str) -> Dict[str, Any]:
    """
    Prediksi sentimen untuk teks review menggunakan model baseline terbaik.

    Returns:
        dict dengan key: sentiment, confidence, probabilities, model_used
    """
    import joblib
    import numpy as np

    baseline_dir = resolve_path(config, "sentiment_dir")
    metrics_path = baseline_dir / "metrics.json"

    if not metrics_path.exists():
        raise FileNotFoundError(
            "Sentiment model belum dilatih. "
            "Jalankan: main.py --step train_sentiment_baseline --allow-training"
        )

    metrics = load_json(metrics_path)
    best_model_name = metrics.get("best_model", "")

    if not best_model_name:
        raise ValueError("Tidak ditemukan best_model di metrics.json")

    model_path = baseline_dir / f"{best_model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file tidak ditemukan: {model_path}")

    pipeline = joblib.load(model_path)
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba([text])[0]
        pred_label = int(np.argmax(proba))
        confidence = float(proba[pred_label])
        probabilities = {label_map[i]: float(p) for i, p in enumerate(proba)}
    else:
        # LinearSVC: gunakan decision_function → softmax pseudo-probabilities
        scores = pipeline.decision_function([text])[0]
        exp_scores = np.exp(scores - np.max(scores))
        proba = exp_scores / exp_scores.sum()
        pred_label = int(np.argmax(proba))
        confidence = float(proba[pred_label])
        probabilities = {label_map[i]: float(p) for i, p in enumerate(proba)}

    logging.info("Sentiment inference: '%s...' → %s (%.4f)", text[:50], label_map[pred_label], confidence)

    return {
        "sentiment": label_map[pred_label],
        "confidence": round(confidence, 4),
        "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
        "model_used": best_model_name,
    }


def recommend_for_user(config: Dict, user_id: str, top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Ambil rekomendasi buku untuk user_id dari artifact hybrid recommender.

    Returns:
        List of dicts: user_id, rank, item_id, final_score, collaborative_score,
                       sentiment_score, popularity_score, content_score
    """
    import pandas as pd

    rec_dir = resolve_path(config, "recommender_dir")
    hybrid_csv = rec_dir / "hybrid" / "hybrid_recommendations.csv"

    if not hybrid_csv.exists():
        raise FileNotFoundError(
            "Recommender belum dilatih. "
            "Jalankan: main.py --step train_recommender --allow-training"
        )

    df = pd.read_csv(hybrid_csv)
    user_recs = df[df["user_id"] == user_id].sort_values("rank").head(top_n)

    if user_recs.empty:
        return []

    return user_recs.to_dict(orient="records")
