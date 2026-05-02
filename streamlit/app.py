"""
Amazon Books ML Dashboard — Full Featured Web App
Tabs: Overview | EDA | Pipeline | Reports | Inference | Cluster
"""
from __future__ import annotations

import getpass
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# ── path setup ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
ML_DIR   = ROOT_DIR / "machine_learning"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

# ── helpers ──────────────────────────────────────────────────────────────────
def _python_bin() -> str:
    p = ROOT_DIR / ".venv" / "bin" / "python"
    return str(p) if p.exists() else "python3"


def _default_hdfs_dataset_path() -> str:
    return f"/user/{getpass.getuser()}/amazon_books"


def _default_hdfs_output_path() -> str:
    return f"/user/{getpass.getuser()}/output/amazon_books_ml"


def _load_config() -> Dict:
    try:
        from src.utils import load_config
        return load_config()
    except Exception as e:
        st.error(f"Gagal load config.yaml: {e}")
        return {}


def _load_cluster_cfg() -> Dict:
    import yaml
    p = ROOT_DIR / "config" / "cluster.yaml"
    if p.exists():
        return yaml.safe_load(p.read_text()) or {}
    default_dataset = _default_hdfs_dataset_path()
    default_output = _default_hdfs_output_path()
    return {
        "cluster": {"master": "fadhli", "workers": ["fadhli@worker1", "fadhli@worker2"], "ssh_timeout": 5},
        "hdfs":    {
            "namenode": "hdfs://fadhli:9000",
            "dataset_path": default_dataset,
            "output_path": default_output,
            "ui_host": "fadhli",
            "ui_port": 9870,
        },
        "yarn":    {"ui_host": "fadhli", "ui_port": 8088},
        "spark":   {"master": "yarn", "deploy_mode": "client", "num_executors": 2,
                    "executor_cores": 2, "executor_memory": "2G", "driver_memory": "2G"},
    }


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return None


def _run(cmd: List[str], cwd: Path = ROOT_DIR, timeout: int = 300) -> Dict:
    try:
        r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
        return {"cmd": " ".join(cmd), "rc": r.returncode, "out": r.stdout, "err": r.stderr}
    except subprocess.TimeoutExpired:
        return {"cmd": " ".join(cmd), "rc": -1, "out": "", "err": "Timeout."}
    except Exception as e:
        return {"cmd": " ".join(cmd), "rc": -1, "out": "", "err": str(e)}


def _show_result(res: Dict) -> None:
    st.code(res["cmd"], language="bash")
    if res["rc"] == 0:
        st.success("✅ Berhasil")
    else:
        st.error(f"❌ Gagal (exit {res['rc']})")
    c1, c2 = st.columns(2)
    with c1:
        with st.expander("📄 Stdout", expanded=res["rc"] == 0):
            st.text(res["out"] or "(kosong)")
    with c2:
        with st.expander("⚠️ Stderr", expanded=res["rc"] != 0):
            st.text(res["err"] or "(kosong)")


def _ssh_ok(worker: str, timeout: int = 5) -> bool:
    try:
        r = subprocess.run(
            ["ssh", "-o", f"ConnectTimeout={timeout}", "-o", "BatchMode=yes",
             "-o", "StrictHostKeyChecking=no", worker, "echo OK"],
            capture_output=True, text=True, timeout=timeout + 2
        )
        return r.returncode == 0 and "OK" in r.stdout
    except Exception:
        return False


# ── TAB: OVERVIEW ────────────────────────────────────────────────────────────
def tab_overview(cfg: Dict, ccfg: Dict) -> None:
    st.subheader("🗂️ Status Artifact")
    checks = {
        "📁 Dataset (Books_rating.csv)":      ML_DIR / "dataset" / "Books_rating.csv",
        "⚙️ Processed (train.csv)":           ML_DIR / "data" / "processed" / "train.csv",
        "🤖 Baseline Sentiment metrics.json": ML_DIR / "models" / "sentiment" / "baseline" / "metrics.json",
        "🔥 Transformer metrics.json":        ML_DIR / "models" / "sentiment" / "transformer" / "distilbert_v1" / "metrics.json",
        "📚 Recommender metrics.json":        ML_DIR / "reports" / "recommender_metrics.json",
        "📋 Model Registry":                  ML_DIR / "models" / "model_registry.json",
        "📝 Final Report":                    ML_DIR / "reports" / "final_report.json",
    }
    rows = []
    for name, path in checks.items():
        ex = path.exists()
        sz = f"{path.stat().st_size/1024:.1f} KB" if ex else "—"
        rows.append({"Artifact": name, "Status": "✅ Ada" if ex else "❌ Belum", "Ukuran": sz})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Quick metrics from final report
    report = _load_json(ML_DIR / "reports" / "final_report.json")
    if report:
        st.subheader("📈 Ringkasan Hasil Training")
        c1, c2, c3 = st.columns(3)
        bl = report.get("sentiment_baseline") or {}
        tr = report.get("sentiment_transformer") or {}
        rc = report.get("recommender") or {}
        if bl:
            c1.metric("🏆 Best Baseline", bl.get("best_model", "—"),
                      f"F1={bl.get('best_validation_f1_weighted', 0):.4f}")
        if tr:
            c2.metric("🔥 Transformer F1", f"{tr.get('test', {}).get('f1_weighted', 0):.4f}",
                      f"Acc={tr.get('test', {}).get('accuracy', 0):.4f}")
        if rc:
            c3.metric("📚 Recommender RMSE", f"{rc.get('rmse', 0):.4f}",
                      f"NDCG@10={rc.get('ndcg_at_k', {}).get('10', 0):.4f}")
    else:
        st.info("Belum ada final report. Jalankan pipeline terlebih dahulu.")


# ── TAB: EDA ─────────────────────────────────────────────────────────────────
def tab_eda(cfg: Dict) -> None:
    eda_dir = ML_DIR / "reports" / "eda"
    if not eda_dir.exists() or not list(eda_dir.iterdir()):
        st.warning("⚠️ Belum ada hasil EDA. Jalankan step `eda` di tab Pipeline.")
        return

    # Stats
    stats_path = eda_dir / "basic_stats.csv"
    if stats_path.exists():
        st.subheader("📊 Statistik Dasar")
        df_s = pd.read_csv(stats_path)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(df_s, use_container_width=True, hide_index=True)
        with c2:
            for _, row in df_s.iterrows():
                if row["metric"] in ["rows", "unique_users", "unique_items"]:
                    st.metric(row["metric"].replace("_", " ").title(), f"{int(row['value']):,}")
                elif row["metric"] == "avg_rating":
                    st.metric("Avg Rating", f"{row['value']:.3f}")

    # Charts
    c1, c2 = st.columns(2)
    rating_png  = eda_dir / "rating_distribution.png"
    review_png  = eda_dir / "review_length_distribution.png"
    with c1:
        st.subheader("⭐ Distribusi Rating")
        if rating_png.exists():
            st.image(str(rating_png), use_container_width=True)
        else:
            st.info("Grafik belum ada.")
    with c2:
        st.subheader("📝 Panjang Review")
        if review_png.exists():
            st.image(str(review_png), use_container_width=True)
        else:
            st.info("Grafik belum ada.")

    # Rating table
    rating_csv = eda_dir / "rating_distribution.csv"
    if rating_csv.exists():
        st.subheader("📋 Tabel Distribusi Rating")
        st.dataframe(pd.read_csv(rating_csv), use_container_width=True, hide_index=True)

    # Sample reviews
    ex_csv = eda_dir / "sentiment_examples.csv"
    if ex_csv.exists():
        st.subheader("💬 Contoh Review per Sentimen")
        df_ex = pd.read_csv(ex_csv)
        for sent, icon in [("Positive", "🟢"), ("Neutral", "🟡"), ("Negative", "🔴")]:
            subset = df_ex[df_ex["sentiment_bucket"] == sent]
            if not subset.empty:
                with st.expander(f"{icon} {sent} ({len(subset)} contoh)"):
                    st.dataframe(subset[["rating", "review_text"]], use_container_width=True, hide_index=True)

    # Summary text
    summ = eda_dir / "eda_summary.txt"
    if summ.exists():
        st.subheader("💡 Insight EDA")
        st.info(summ.read_text("utf-8"))


# ── TAB: PIPELINE ─────────────────────────────────────────────────────────────
def tab_pipeline(cfg: Dict) -> None:
    st.subheader("🚀 Pipeline Runner")
    st.caption("Training diblokir secara default. Centang 'Izinkan Training' untuk step training.")

    STEPS = [
        ("eda",                        "🔍 EDA",                            False),
        ("preprocess",                 "⚙️ Preprocess (Cleaning & Split)",  False),
        ("train_sentiment_baseline",   "🤖 Train Baseline Sentiment",       True),
        ("train_sentiment_transformer","🔥 Train Transformer (DistilBERT)", True),
        ("train_recommender",          "📚 Train Recommender System",       True),
        ("evaluate",                   "📊 Evaluate (Compile Final Report)", False),
        ("all",                        "🔄 All Steps",                      False),
    ]

    c1, c2 = st.columns([3, 1])
    with c1:
        idx = st.selectbox("Pilih Step", range(len(STEPS)), format_func=lambda i: STEPS[i][1])
    step_key, step_label, needs_train = STEPS[idx]

    with c2:
        allow = st.checkbox("☑️ Izinkan Training", value=False,
                            help="Wajib untuk step training.", disabled=not needs_train and step_key != "all")

    if needs_train and not allow:
        st.warning(f"⚠️ Step `{step_key}` memerlukan flag `--allow-training`.")

    if st.button("▶️ Jalankan", type="primary", use_container_width=True):
        cmd = [_python_bin(), "machine_learning/main.py", "--step", step_key]
        if allow:
            cmd.append("--allow-training")
        with st.spinner(f"Menjalankan: {step_label}..."):
            res = _run(cmd)
        _show_result(res)


# ── TAB: REPORTS ──────────────────────────────────────────────────────────────
def tab_reports(cfg: Dict) -> None:
    # ---- Baseline Sentiment ----
    st.subheader("🤖 Sentiment Baseline")
    bl_path = ML_DIR / "models" / "sentiment" / "baseline" / "metrics.json"
    bl_data = _load_json(bl_path)
    if bl_data:
        best = bl_data.get("best_model", "—")
        st.info(f"🏆 Best: `{best}` — Val F1 = `{bl_data.get('best_validation_f1_weighted', 0):.4f}`")
        rows = []
        for mn, md in bl_data.get("metrics", {}).items():
            vm = md.get("validation", {})
            tm = md.get("test", {})
            rows.append({
                "Model": mn,
                "Val Acc": f"{vm.get('accuracy', 0):.4f}",
                "Val F1":  f"{vm.get('f1_weighted', 0):.4f}",
                "Test Acc": f"{tm.get('accuracy', 0):.4f}",
                "Test F1":  f"{tm.get('f1_weighted', 0):.4f}",
                "Precision": f"{tm.get('precision_weighted', 0):.4f}",
                "Recall":    f"{tm.get('recall_weighted', 0):.4f}",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Confusion matrices
        bl_dir = ML_DIR / "models" / "sentiment" / "baseline"
        model_names = list(bl_data.get("metrics", {}).keys())
        if model_names:
            st.markdown("**Confusion Matrices:**")
            cols = st.columns(min(len(model_names), 3))
            for i, mn in enumerate(model_names):
                cm_path = bl_dir / f"{mn}_confusion_matrix_test.png"
                if cm_path.exists():
                    cols[i % 3].image(str(cm_path), caption=mn, use_container_width=True)
    else:
        st.info("Baseline model belum dilatih.")

    st.divider()

    # ---- Transformer ----
    st.subheader("🔥 Transformer (DistilBERT)")
    tr_path = ML_DIR / "models" / "sentiment" / "transformer" / "distilbert_v1" / "metrics.json"
    tr_data = _load_json(tr_path)
    if tr_data:
        c1, c2 = st.columns(2)
        vm = tr_data.get("validation", {})
        tm = tr_data.get("test", {})
        with c1:
            st.markdown("**Validation**")
            for k, v in vm.items():
                if isinstance(v, float):
                    st.metric(k, f"{v:.4f}")
        with c2:
            st.markdown("**Test**")
            for k, v in tm.items():
                if isinstance(v, float):
                    st.metric(k, f"{v:.4f}")
        cm_tr = ML_DIR / "models" / "sentiment" / "transformer" / "distilbert_v1" / "confusion_matrix_test.png"
        if cm_tr.exists():
            st.image(str(cm_tr), caption="Confusion Matrix (Test)", width=450)
    else:
        st.info("Transformer belum dilatih.")

    st.divider()

    # ---- Recommender ----
    st.subheader("📚 Recommender System")
    rec_data = _load_json(ML_DIR / "reports" / "recommender_metrics.json")
    if rec_data:
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{rec_data.get('rmse', 0):.4f}")
        c2.metric("MAE",  f"{rec_data.get('mae', 0):.4f}")
        c3.metric("Evaluated Users", rec_data.get("evaluated_users", 0))

        for metric_name in ["precision_at_k", "recall_at_k", "ndcg_at_k", "coverage_at_k"]:
            mdata = rec_data.get(metric_name, {})
            if mdata:
                label = metric_name.replace("_", " ").upper()
                st.markdown(f"**{label}:**")
                df_m = pd.DataFrame([{"K": f"@{k}", "Value": round(v, 4)} for k, v in mdata.items()])
                st.dataframe(df_m, use_container_width=False, hide_index=True)
    else:
        st.info("Recommender belum dilatih.")

    st.divider()

    # ---- Model Registry ----
    st.subheader("📋 Model Registry")
    reg_data = _load_json(ML_DIR / "models" / "model_registry.json")
    if reg_data and reg_data.get("models"):
        rows = []
        for m in reg_data["models"]:
            rows.append({
                "Name": m.get("name", "—"),
                "Task": m.get("task", "—"),
                "Version": m.get("version", "—"),
                "Trained At": m.get("trained_at_utc", "—")[:19],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada model yang terdaftar.")


# ── TAB: INFERENCE ────────────────────────────────────────────────────────────
def tab_inference(cfg: Dict) -> None:
    # Sentiment
    st.subheader("🔮 Sentiment Inference")
    review_text = st.text_area("Teks Review", value="This book is very helpful and well written.", height=120)
    if st.button("Prediksi Sentiment", type="primary"):
        try:
            from src.inference import predict_sentiment
            with st.spinner("Memproses..."):
                res = predict_sentiment(cfg, review_text)
            sent = res["sentiment"]
            icon = {"Positive": "🟢", "Neutral": "🟡", "Negative": "🔴"}.get(sent, "⚪")
            c1, c2, c3 = st.columns(3)
            c1.metric("Sentimen", f"{icon} {sent}")
            c2.metric("Confidence", f"{res['confidence']:.4f}")
            c3.metric("Model", res.get("model_used", "—"))
            st.markdown("**Probabilitas per Kelas:**")
            prob_df = pd.DataFrame([{"Label": k, "Probability": v} for k, v in res["probabilities"].items()])
            st.dataframe(prob_df, use_container_width=False, hide_index=True)
        except FileNotFoundError as e:
            st.warning(str(e))
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()

    # Recommender
    st.subheader("📚 Recommender Inference")
    user_id = st.text_input("User ID", placeholder="Masukkan user_id dari dataset...")
    top_n   = st.slider("Top-N", 1, 20, 10)
    if st.button("Ambil Rekomendasi"):
        if not user_id.strip():
            st.warning("Masukkan User ID terlebih dahulu.")
        else:
            try:
                from src.inference import recommend_for_user
                with st.spinner("Mengambil rekomendasi..."):
                    recs = recommend_for_user(cfg, user_id.strip(), top_n)
                if not recs:
                    st.warning("Tidak ada rekomendasi untuk user tersebut.")
                else:
                    st.success(f"Ditemukan {len(recs)} rekomendasi.")
                    df_rec = pd.DataFrame(recs)
                    display_cols = [c for c in ["rank", "item_id", "final_score",
                                                "collaborative_score", "sentiment_score",
                                                "popularity_score"] if c in df_rec.columns]
                    st.dataframe(df_rec[display_cols], use_container_width=True, hide_index=True)
            except FileNotFoundError as e:
                st.warning(str(e))
            except Exception as e:
                st.error(f"Error: {e}")


# ── TAB: CLUSTER ──────────────────────────────────────────────────────────────
def tab_cluster(cfg: Dict, ccfg: Dict) -> None:
    workers  = ccfg.get("cluster", {}).get("workers", ["fadhli@worker1", "fadhli@worker2"])
    timeout  = ccfg.get("cluster", {}).get("ssh_timeout", 5)
    hdfs_cfg = ccfg.get("hdfs", {})
    yarn_cfg = ccfg.get("yarn", {})
    sp_cfg   = ccfg.get("spark", {})

    # ---- Node Status ----
    st.subheader("🖥️ Status Node")
    if st.button("🔄 Cek Status SSH Worker"):
        status_rows = []
        for w in workers:
            with st.spinner(f"Mengecek {w}..."):
                ok = _ssh_ok(w, timeout)
            status_rows.append({"Node": w, "SSH": "✅ Online" if ok else "❌ Offline"})
        st.dataframe(pd.DataFrame(status_rows), use_container_width=False, hide_index=True)

    # ---- Web UIs ----
    st.subheader("🌐 Web UI Hadoop")
    h_host = hdfs_cfg.get("ui_host", "fadhli")
    h_port = hdfs_cfg.get("ui_port", 9870)
    y_host = yarn_cfg.get("ui_host", "fadhli")
    y_port = yarn_cfg.get("ui_port", 8088)
    c1, c2 = st.columns(2)
    c1.link_button("🗄️ HDFS NameNode UI", f"http://{h_host}:{h_port}", use_container_width=True)
    c2.link_button("⚡ YARN ResourceManager UI", f"http://{y_host}:{y_port}", use_container_width=True)

    # ---- Cluster Control ----
    st.subheader("⚙️ Cluster Control")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶️ Start Cluster", use_container_width=True):
            with st.spinner("Starting..."):
                res = _run(["bash", str(ROOT_DIR / "scripts" / "start_cluster.sh")])
            _show_result(res)
    with c2:
        if st.button("⏹️ Stop Cluster", use_container_width=True):
            with st.spinner("Stopping..."):
                res = _run(["bash", str(ROOT_DIR / "scripts" / "stop_cluster.sh")])
            _show_result(res)

    # ---- Permission Fix ----
    st.subheader("🔐 Perbaiki Permission HDFS")
    st.caption("Mode ini bersifat permisif (world-writable) untuk mempermudah lab/testing cluster.")
    permission_target_user = st.text_input("Target user HDFS", value=getpass.getuser(), key="perm_target_user")
    if st.button("Fix Permission Worker1 & Worker2", use_container_width=True):
        with st.spinner("Memperbaiki permission HDFS..."):
            res = _run(
                ["bash", str(ROOT_DIR / "scripts" / "fix_hdfs_permissions.sh"), permission_target_user],
                timeout=600,
            )
        _show_result(res)

    # ---- HDFS Upload ----
    st.subheader("📤 Upload Dataset ke HDFS")
    hdfs_target = st.text_input("HDFS Target Path", value=hdfs_cfg.get("dataset_path", _default_hdfs_dataset_path()))
    if st.button("Upload ke HDFS", use_container_width=True):
        with st.spinner("Mengupload..."):
            res = _run(["bash", str(ROOT_DIR / "scripts" / "upload_to_hdfs.sh"), hdfs_target])
        _show_result(res)

    # ---- Spark Submit ----
    st.subheader("⚡ Spark Submit (Distributed Training / Preprocessing)")
    with st.expander("Konfigurasi Spark"):
        col1, col2, col3, col4 = st.columns(4)
        n_exec  = col1.number_input("num-executors",   value=int(sp_cfg.get("num_executors", 2)),   min_value=1, max_value=10)
        e_cores = col2.number_input("executor-cores",  value=int(sp_cfg.get("executor_cores", 2)),  min_value=1, max_value=8)
        e_mem   = col3.text_input("executor-memory", value=sp_cfg.get("executor_memory", "2G"))
        d_mem   = col4.text_input("driver-memory",   value=sp_cfg.get("driver_memory", "2G"))

    spark_step = st.selectbox("Step Spark", [
        "preprocess_spark — Distributed Preprocessing via YARN",
    ])
    spark_step_key = spark_step.split("—")[0].strip()

    if st.button("🚀 Submit Spark Job", type="primary", use_container_width=True):
        cmd = [
            "bash", str(ROOT_DIR / "scripts" / "spark_submit_training.sh"),
            spark_step_key,
            str(int(n_exec)), str(int(e_cores)), e_mem, d_mem,
        ]
        with st.spinner("Submitting Spark job ke YARN..."):
            res = _run(cmd, timeout=600)
        _show_result(res)

    # ---- HDFS file browser ----
    st.subheader("🗂️ Browse HDFS")
    default_hdfs_path = hdfs_cfg.get("dataset_path", _default_hdfs_dataset_path())
    hdfs_path = st.text_input("HDFS Path", value=default_hdfs_path)
    if st.button("List HDFS"):
        res = _run(["hdfs", "dfs", "-ls", "-h", hdfs_path])
        _show_result(res)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Amazon Books ML Dashboard",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Header
    st.title("📚 Amazon Books ML Dashboard")
    st.caption("Sentiment Analysis & Recommender System — Hadoop Multi-node Edition")
    st.divider()

    cfg  = _load_config()
    ccfg = _load_cluster_cfg()

    tabs = st.tabs(["🏠 Overview", "📊 EDA", "🚀 Pipeline", "📈 Reports", "🔮 Inference", "🖥️ Cluster"])

    with tabs[0]:
        tab_overview(cfg, ccfg)
    with tabs[1]:
        tab_eda(cfg)
    with tabs[2]:
        tab_pipeline(cfg)
    with tabs[3]:
        tab_reports(cfg)
    with tabs[4]:
        tab_inference(cfg)
    with tabs[5]:
        tab_cluster(cfg, ccfg)


if __name__ == "__main__":
    main()
