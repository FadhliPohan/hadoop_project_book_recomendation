from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import streamlit as st

if not hasattr(st, "set_page_config"):
    raise RuntimeError(
        "Package Streamlit tidak terdeteksi dengan benar. "
        "Jalankan via command `streamlit run streamlit/app.py` setelah install package streamlit."
    )

ROOT_DIR = Path(__file__).resolve().parents[1]
ML_DIR = ROOT_DIR / "machine_learning"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src.inference import predict_sentiment, recommend_for_user  # noqa: E402
from src.utils import load_config  # noqa: E402


def _python_bin() -> str:
    venv_python = ROOT_DIR / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return "python3"


def _run_command(cmd: List[str], cwd: Path = ROOT_DIR) -> Dict:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _run_pipeline_step(step: str, allow_training: bool) -> Dict:
    cmd = [_python_bin(), "machine_learning/main.py", "--step", step]
    if allow_training:
        cmd.append("--allow-training")
    return _run_command(cmd)


def _run_script(script_name: str, args: List[str] | None = None) -> Dict:
    args = args or []
    script_path = ROOT_DIR / "scripts" / script_name
    cmd = ["bash", str(script_path), *args]
    return _run_command(cmd)


def _show_command_result(result: Dict) -> None:
    st.code(result["command"], language="bash")
    if result["returncode"] == 0:
        st.success("Command selesai tanpa error.")
    else:
        st.error(f"Command gagal (exit code={result['returncode']}).")

    with st.expander("Stdout", expanded=True):
        st.text(result["stdout"] or "(kosong)")
    with st.expander("Stderr"):
        st.text(result["stderr"] or "(kosong)")


def _pipeline_page(config: Dict) -> None:
    st.subheader("Pipeline Runner")
    st.caption("Training diblokir default. Aktifkan hanya jika memang ingin training.")

    step = st.selectbox(
        "Pilih step",
        [
            "eda",
            "preprocess",
            "train_sentiment_baseline",
            "train_sentiment_transformer",
            "train_recommender",
            "evaluate",
            "all",
        ],
        index=0,
    )
    allow_training = st.checkbox("Izinkan training untuk step ini (--allow-training)", value=False)

    if st.button("Jalankan Step", type="primary"):
        result = _run_pipeline_step(step=step, allow_training=allow_training)
        _show_command_result(result)


def _cluster_page() -> None:
    st.subheader("Cluster Control")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Cluster"):
            result = _run_script("start_cluster.sh")
            _show_command_result(result)

    with col2:
        if st.button("Stop Cluster"):
            result = _run_script("stop_cluster.sh")
            _show_command_result(result)

    st.markdown("### Upload Dataset ke HDFS")
    hdfs_target = st.text_input("HDFS target", value="/data/amazon_books")
    if st.button("Upload ke HDFS"):
        result = _run_script("upload_to_hdfs.sh", [hdfs_target])
        _show_command_result(result)


def _inference_page(config: Dict) -> None:
    st.subheader("Inference")

    with st.expander("Sentiment Inference", expanded=True):
        review_text = st.text_area(
            "Review text",
            value="This book is very helpful and well written.",
            height=120,
        )
        if st.button("Prediksi Sentiment"):
            try:
                result = predict_sentiment(config, review_text)
                st.json(result)
            except Exception as exc:
                st.error(f"Gagal menjalankan sentiment inference: {exc}")

    with st.expander("Recommender Inference", expanded=True):
        user_id = st.text_input("User ID")
        top_n = st.slider("Top-N", min_value=1, max_value=20, value=10)
        if st.button("Ambil Rekomendasi"):
            try:
                recs = recommend_for_user(config, user_id=user_id, top_n=top_n)
                if not recs:
                    st.warning("Tidak ada rekomendasi untuk user_id tersebut.")
                else:
                    st.dataframe(recs, use_container_width=True)
            except Exception as exc:
                st.error(f"Gagal menjalankan recommender inference: {exc}")


def _status_page(config: Dict) -> None:
    st.subheader("Status Artefak")

    checks = {
        "Dataset reviews": ML_DIR / "dataset" / "Books_rating.csv",
        "Processed train.csv": ML_DIR / "data" / "processed" / "train.csv",
        "Sentiment metrics": ML_DIR / "models" / "sentiment" / "baseline" / "metrics.json",
        "Recommender metrics": ML_DIR / "reports" / "recommender_metrics.json",
        "Model registry": ML_DIR / "models" / "model_registry.json",
    }

    rows = []
    for name, path in checks.items():
        rows.append(
            {
                "artifact": name,
                "path": str(path),
                "exists": path.exists(),
            }
        )

    st.dataframe(rows, use_container_width=True)

    final_report = ML_DIR / "reports" / "final_report.json"
    if final_report.exists():
        st.markdown("### final_report.json")
        try:
            st.json(json.loads(final_report.read_text(encoding="utf-8")))
        except Exception as exc:
            st.error(f"Gagal membaca final report: {exc}")


def main() -> None:
    st.set_page_config(page_title="Amazon Books ML Dashboard", layout="wide")
    st.title("Amazon Books ML Dashboard")
    st.caption("Workflow aman: coding-first, training by explicit permission.")

    config = load_config()

    tabs = st.tabs(["Pipeline", "Cluster", "Inference", "Status"])
    with tabs[0]:
        _pipeline_page(config)
    with tabs[1]:
        _cluster_page()
    with tabs[2]:
        _inference_page(config)
    with tabs[3]:
        _status_page(config)


if __name__ == "__main__":
    main()
