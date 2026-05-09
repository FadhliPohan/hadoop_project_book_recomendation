from __future__ import annotations

from contextlib import nullcontext
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from .mlflow_tracker import log_artifact, log_metrics, log_params, start_run
from .utils import append_model_registry, ensure_dir, resolve_path, save_json


@dataclass
class CollabModel:
    user_to_idx: Dict[str, int]
    item_to_idx: Dict[str, int]
    idx_to_item: Dict[int, str]
    us: np.ndarray
    vt: np.ndarray
    rating_min: float
    rating_max: float
    global_mean: float


def precision_at_k(recommended: Sequence[str], relevant: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    recommended_k = recommended[:k]
    if not recommended_k:
        return 0.0
    hits = len(set(recommended_k) & relevant)
    return hits / k


def recall_at_k(recommended: Sequence[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: Sequence[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0

    dcg = 0.0
    for idx, item_id in enumerate(recommended[:k], start=1):
        if item_id in relevant:
            dcg += 1.0 / np.log2(idx + 1)

    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _iterative_filter(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    filtered = df.copy()
    while True:
        before = len(filtered)
        user_counts = filtered.groupby("user_id").size()
        item_counts = filtered.groupby("item_id").size()
        filtered = filtered[filtered["user_id"].isin(user_counts[user_counts >= min_user].index)]
        filtered = filtered[filtered["item_id"].isin(item_counts[item_counts >= min_item].index)]
        if len(filtered) == before:
            return filtered


def _build_popularity_scores(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("item_id").agg(avg_rating=("rating", "mean"), review_count=("rating", "count"))
    c = agg["avg_rating"].mean()
    m = agg["review_count"].quantile(0.75)

    v = agg["review_count"]
    r = agg["avg_rating"]
    agg["popularity_score_raw"] = (v / (v + m)) * r + (m / (v + m)) * c

    min_val = agg["popularity_score_raw"].min()
    max_val = agg["popularity_score_raw"].max()
    if max_val > min_val:
        agg["popularity_score"] = (agg["popularity_score_raw"] - min_val) / (max_val - min_val)
    else:
        agg["popularity_score"] = 0.0

    return agg.sort_values(["popularity_score", "review_count"], ascending=False)


def _build_collaborative_model(train_df: pd.DataFrame, latent_factors: int) -> CollabModel:
    users = sorted(train_df["user_id"].unique())
    items = sorted(train_df["item_id"].unique())

    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {it: i for i, it in enumerate(items)}
    idx_to_item = {i: it for it, i in item_to_idx.items()}

    row = train_df["user_id"].map(user_to_idx).to_numpy()
    col = train_df["item_id"].map(item_to_idx).to_numpy()
    val = train_df["rating"].to_numpy(dtype=np.float32)

    matrix = csr_matrix((val, (row, col)), shape=(len(users), len(items)), dtype=np.float32)

    k_max = min(matrix.shape) - 1
    k = min(latent_factors, max(2, k_max))
    if k < 2:
        raise ValueError("Matrix terlalu kecil untuk SVD collaborative filtering.")

    u, s, vt = svds(matrix, k=k)

    # Ensure consistent descending singular values for stable behavior.
    order = np.argsort(s)[::-1]
    s = s[order]
    u = u[:, order]
    vt = vt[order, :]

    us = u * s

    return CollabModel(
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
        us=us,
        vt=vt,
        rating_min=float(train_df["rating"].min()),
        rating_max=float(train_df["rating"].max()),
        global_mean=float(train_df["rating"].mean()),
    )


def _predict_collab_raw(model: CollabModel, user_id: str, item_id: str) -> float:
    u_idx = model.user_to_idx.get(user_id)
    i_idx = model.item_to_idx.get(item_id)
    if u_idx is None or i_idx is None:
        return model.global_mean

    score = float(model.us[u_idx, :].dot(model.vt[:, i_idx]))
    # Bound prediction to rating scale to avoid unrealistic extreme values.
    return float(np.clip(score, model.rating_min, model.rating_max))


def _normalize_rating(score: float, rating_min: float, rating_max: float) -> float:
    if rating_max <= rating_min:
        return 0.0
    norm = (score - rating_min) / (rating_max - rating_min)
    return float(np.clip(norm, 0.0, 1.0))


def _build_content_model(train_df: pd.DataFrame, out_dir: Path) -> Tuple[TfidfVectorizer, csr_matrix, Dict[str, int]]:
    # pandas >=2.0 compatible: reset_index() + explicit rename
    grouped = (
        train_df.groupby("item_id")["review_text_processed"]
        .apply(lambda s: " ".join(s.astype(str).head(5)))
        .reset_index()
        .rename(columns={"review_text_processed": "item_text"})
    )

    vectorizer = TfidfVectorizer(max_features=15000, min_df=2, ngram_range=(1, 2))
    item_matrix = vectorizer.fit_transform(grouped["item_text"].fillna(""))
    item_matrix = normalize(item_matrix)

    item_to_idx = {item: idx for idx, item in enumerate(grouped["item_id"].tolist())}

    ensure_dir(out_dir)
    save_npz(out_dir / "item_text_tfidf.npz", item_matrix)
    with (out_dir / "item_index.json").open("w", encoding="utf-8") as f:
        json.dump(item_to_idx, f)

    return vectorizer, item_matrix.tocsr(), item_to_idx


def _content_user_profile_scores(
    user_items: Iterable[str],
    candidate_items: Sequence[str],
    item_matrix: csr_matrix,
    item_to_idx: Mapping[str, int],
) -> Dict[str, float]:
    hist_indices = [item_to_idx[it] for it in user_items if it in item_to_idx]
    if not hist_indices:
        return {item: 0.0 for item in candidate_items}

    profile = item_matrix[hist_indices].mean(axis=0)
    profile_norm = np.linalg.norm(profile)
    if profile_norm == 0:
        return {item: 0.0 for item in candidate_items}

    candidate_indices = [item_to_idx[it] for it in candidate_items if it in item_to_idx]
    if not candidate_indices:
        return {item: 0.0 for item in candidate_items}

    cand_mat = item_matrix[candidate_indices]
    scores_array = cand_mat.dot(profile.T)
    scores_array = np.asarray(scores_array).reshape(-1)

    score_map = {item: 0.0 for item in candidate_items}
    present_items = [it for it in candidate_items if it in item_to_idx]
    for item, score in zip(present_items, scores_array):
        score_map[item] = float(np.clip(score, 0.0, 1.0))

    return score_map


def _recommend_for_user(
    user_id: str,
    train_df: pd.DataFrame,
    pop_scores: Dict[str, float],
    sentiment_scores: Dict[str, float],
    collab_model: CollabModel,
    item_matrix: csr_matrix,
    item_to_idx: Dict[str, int],
    candidate_pool_size: int,
) -> List[Tuple[str, float, float, float, float, float]]:
    seen_items = set(train_df.loc[train_df["user_id"] == user_id, "item_id"].tolist())
    candidate_items = [it for it in pop_scores.keys() if it not in seen_items][:candidate_pool_size]

    content_scores = _content_user_profile_scores(seen_items, candidate_items, item_matrix, item_to_idx)

    ranked = []
    for item_id in candidate_items:
        pop = pop_scores.get(item_id, 0.0)
        sent = sentiment_scores.get(item_id, 0.0)
        collab_raw = _predict_collab_raw(collab_model, user_id, item_id)
        collab = _normalize_rating(collab_raw, collab_model.rating_min, collab_model.rating_max)
        content = content_scores.get(item_id, 0.0)

        final_score = 0.5 * collab + 0.3 * sent + 0.2 * pop
        ranked.append((item_id, final_score, collab, sent, pop, content))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def _diversity_at_k(
    recommendations: Dict[str, List[str]],
    item_matrix: csr_matrix,
    item_to_idx: Dict[str, int],
    k: int,
) -> float:
    diversities = []
    for recs in recommendations.values():
        items = recs[:k]
        idx = [item_to_idx[it] for it in items if it in item_to_idx]
        if len(idx) < 2:
            continue

        mat = item_matrix[idx]
        sim = (mat @ mat.T).toarray()
        upper = sim[np.triu_indices_from(sim, k=1)]
        if upper.size == 0:
            continue
        diversities.append(float(1.0 - upper.mean()))

    return float(np.mean(diversities)) if diversities else 0.0


def _load_processed_interactions(processed_path: Path) -> pd.DataFrame:
    usecols = ["user_id", "item_id", "rating", "sentiment_label", "review_text_processed"]
    read_kwargs = {
        "usecols": usecols,
        "dtype": {
            "user_id": "string",
            "item_id": "string",
            "rating": "float32",
            "sentiment_label": "int8",
            "review_text_processed": "string",
        },
        "low_memory": True,
    }

    def _read_chunked(engine: str, chunksize: int) -> pd.DataFrame:
        reader = pd.read_csv(
            processed_path,
            engine=engine,
            chunksize=chunksize,
            **read_kwargs,
        )
        chunks: list[pd.DataFrame] = []
        for chunk in reader:
            if chunk.empty:
                continue
            chunk["user_id"] = chunk["user_id"].fillna("")
            chunk["item_id"] = chunk["item_id"].fillna("")
            chunk["review_text_processed"] = chunk["review_text_processed"].fillna("")
            chunks.append(chunk)

        if not chunks:
            return pd.DataFrame(columns=usecols)
        return pd.concat(chunks, ignore_index=True, copy=False)

    last_exc: Exception | None = None
    for engine, chunksize in [("c", 20000), ("python", 10000)]:
        try:
            return _read_chunked(engine=engine, chunksize=chunksize)
        except (pd.errors.ParserError, MemoryError) as exc:
            last_exc = exc
            logging.warning(
                "Gagal baca interactions dari %s dengan engine='%s' (chunksize=%s): %s. Coba strategi berikutnya.",
                processed_path,
                engine,
                chunksize,
                exc.__class__.__name__,
            )

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Gagal memuat interactions dari {processed_path}")


def train_recommenders(config: Dict) -> Dict:
    processed_path = resolve_path(config, "processed_reviews_csv")
    rec_root = resolve_path(config, "recommender_dir")
    ensure_dir(rec_root)

    popularity_dir = rec_root / "popularity"
    collab_dir = rec_root / "collaborative_filtering"
    content_dir = rec_root / "content_based"
    hybrid_dir = rec_root / "hybrid"

    for path in [popularity_dir, collab_dir, content_dir, hybrid_dir]:
        ensure_dir(path)

    run_ctx = start_run(config, "hybrid_recommender_v1") or nullcontext()
    with run_ctx:
        interactions = _load_processed_interactions(processed_path)

        min_user = config["recommender"]["min_user_interactions"]
        min_item = config["recommender"]["min_item_interactions"]

        filtered = _iterative_filter(interactions, min_user=min_user, min_item=min_item)
        logging.info("Recommender filtered interactions: %s rows", len(filtered))

        if len(filtered) < 2000:
            raise ValueError("Data interaksi terlalu kecil setelah filtering untuk training recommender.")

        train_df, test_df = train_test_split(
            filtered,
            test_size=0.2,
            random_state=config["project"]["random_state"],
        )

        # A. Popularity
        popularity = _build_popularity_scores(train_df)
        popularity_csv = popularity_dir / "top_books_popularity.csv"
        popularity.to_csv(popularity_csv)
        pop_scores = popularity["popularity_score"].to_dict()

        # B. Collaborative filtering (SVD)
        collab_model = _build_collaborative_model(
            train_df=train_df,
            latent_factors=config["recommender"]["latent_factors"],
        )

        test_known = test_df[
            test_df["user_id"].isin(collab_model.user_to_idx)
            & test_df["item_id"].isin(collab_model.item_to_idx)
        ].copy()

        test_known["pred"] = test_known.apply(
            lambda r: _predict_collab_raw(collab_model, r["user_id"], r["item_id"]), axis=1
        )

        rmse = float(np.sqrt(mean_squared_error(test_known["rating"], test_known["pred"])))
        mae = float(mean_absolute_error(test_known["rating"], test_known["pred"]))

        collab_metrics_path = collab_dir / "metrics.json"
        save_json({"rmse": rmse, "mae": mae, "n_eval": int(len(test_known))}, collab_metrics_path)

        factors_path = collab_dir / "factors.npz"
        np.savez(
            factors_path,
            us=collab_model.us,
            vt=collab_model.vt,
            rating_min=collab_model.rating_min,
            rating_max=collab_model.rating_max,
            global_mean=collab_model.global_mean,
        )

        user_index_path = collab_dir / "user_index.json"
        item_index_path = collab_dir / "item_index.json"
        with user_index_path.open("w", encoding="utf-8") as f:
            json.dump(collab_model.user_to_idx, f)
        with item_index_path.open("w", encoding="utf-8") as f:
            json.dump(collab_model.item_to_idx, f)

        # C. Content-based
        _vectorizer, item_matrix, item_to_idx = _build_content_model(train_df, content_dir)

        # D. Hybrid
        sentiment_scores = (train_df.groupby("item_id")["sentiment_label"].mean() / 2.0).to_dict()
        candidate_pool_size = config["recommender"]["candidate_pool_size"]

        relevant_test = (
            test_df[test_df["rating"] >= 4]
            .groupby("user_id")["item_id"]
            .apply(lambda s: set(s.tolist()))
            .to_dict()
        )

        eval_users = list(relevant_test.keys())[:200]
        recommendations: Dict[str, List[str]] = {}
        recommendation_rows = []

        for user_id in eval_users:
            ranked = _recommend_for_user(
                user_id=user_id,
                train_df=train_df,
                pop_scores=pop_scores,
                sentiment_scores=sentiment_scores,
                collab_model=collab_model,
                item_matrix=item_matrix,
                item_to_idx=item_to_idx,
                candidate_pool_size=candidate_pool_size,
            )

            top10 = ranked[:10]
            recommendations[user_id] = [item for item, *_ in top10]

            for rank, (item, final_s, collab_s, sent_s, pop_s, content_s) in enumerate(top10, start=1):
                recommendation_rows.append(
                    {
                        "user_id": user_id,
                        "rank": rank,
                        "item_id": item,
                        "final_score": final_s,
                        "collaborative_score": collab_s,
                        "sentiment_score": sent_s,
                        "popularity_score": pop_s,
                        "content_score": content_s,
                    }
                )

        hybrid_csv = hybrid_dir / "hybrid_recommendations.csv"
        pd.DataFrame(recommendation_rows).to_csv(hybrid_csv, index=False)

        metric_payload = {
            "rmse": rmse,
            "mae": mae,
            "evaluated_users": len(eval_users),
            "precision_at_k": {},
            "recall_at_k": {},
            "ndcg_at_k": {},
            "coverage_at_k": {},
            "diversity_at_k": {},
        }

        all_items = set(pop_scores.keys())
        for k in config["recommender"]["top_k_values"]:
            p_scores = []
            r_scores = []
            n_scores = []
            rec_items_k = set()

            for user_id in eval_users:
                rec_list = recommendations.get(user_id, [])
                rel_items = relevant_test.get(user_id, set())
                p_scores.append(precision_at_k(rec_list, rel_items, k))
                r_scores.append(recall_at_k(rec_list, rel_items, k))
                n_scores.append(ndcg_at_k(rec_list, rel_items, k))
                rec_items_k.update(rec_list[:k])

            metric_payload["precision_at_k"][str(k)] = float(np.mean(p_scores)) if p_scores else 0.0
            metric_payload["recall_at_k"][str(k)] = float(np.mean(r_scores)) if r_scores else 0.0
            metric_payload["ndcg_at_k"][str(k)] = float(np.mean(n_scores)) if n_scores else 0.0
            metric_payload["coverage_at_k"][str(k)] = float(len(rec_items_k) / max(1, len(all_items)))
            metric_payload["diversity_at_k"][str(k)] = _diversity_at_k(recommendations, item_matrix, item_to_idx, k)

        report_path = resolve_path(config, "recommender_report_json")
        save_json(metric_payload, report_path)

        append_model_registry(
            config,
            {
                "name": "svd_recommender_v1",
                "task": "recommender_collaborative",
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": str(processed_path),
                "metrics": {"rmse": rmse, "mae": mae},
                "version": "v1",
                "path": str(collab_dir / "factors.npz"),
                "hyperparameters": {
                    "latent_factors": config["recommender"]["latent_factors"],
                    "min_user_interactions": min_user,
                    "min_item_interactions": min_item,
                },
            },
        )

        append_model_registry(
            config,
            {
                "name": "hybrid_recommender_v1",
                "task": "recommender_hybrid",
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": str(processed_path),
                "metrics": metric_payload,
                "version": "v1",
                "path": str(hybrid_dir / "hybrid_recommendations.csv"),
                "hyperparameters": {
                    "formula": "0.5*collaborative + 0.3*sentiment + 0.2*popularity",
                    "candidate_pool_size": candidate_pool_size,
                },
            },
        )

        log_params(
            config,
            {
                "model_name": "hybrid_recommender_v1",
                "latent_factors": config["recommender"]["latent_factors"],
                "min_user_interactions": min_user,
                "min_item_interactions": min_item,
                "candidate_pool_size": candidate_pool_size,
            },
        )
        log_metrics(
            config,
            {
                "rmse": rmse,
                "mae": mae,
                "precision_at_5": metric_payload["precision_at_k"].get("5", 0.0),
                "recall_at_5": metric_payload["recall_at_k"].get("5", 0.0),
                "ndcg_at_5": metric_payload["ndcg_at_k"].get("5", 0.0),
            },
        )
        log_artifact(config, factors_path, artifact_path="recommender/collaborative_filtering")
        log_artifact(config, collab_metrics_path, artifact_path="recommender/collaborative_filtering")
        log_artifact(config, popularity_csv, artifact_path="recommender/popularity")
        log_artifact(config, hybrid_csv, artifact_path="recommender/hybrid")
        log_artifact(config, report_path, artifact_path="recommender/hybrid")

        return metric_payload
