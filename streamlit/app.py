"""
Amazon Books ML Dashboard — Full Featured Web App
Tabs: Overview | EDA | Pipeline | Reports | Inference | Cluster
"""
from __future__ import annotations

import getpass
import json
import mimetypes
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path, PurePosixPath
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


def _load_recommender_user_ids() -> List[str]:
    hybrid_csv = ML_DIR / "models" / "recommender" / "hybrid" / "hybrid_recommendations.csv"
    if not hybrid_csv.exists():
        return []

    try:
        df = pd.read_csv(hybrid_csv, usecols=["user_id"])
    except Exception:
        return []

    return sorted(df["user_id"].dropna().astype(str).unique().tolist())


def _format_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run(cmd: List[str], cwd: Path = ROOT_DIR, timeout: int = 300) -> Dict:
    display_cmd = _format_cmd(cmd)
    started_at = time.monotonic()
    try:
        r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
        return {
            "cmd": display_cmd,
            "rc": r.returncode,
            "out": r.stdout,
            "err": r.stderr,
            "duration_sec": time.monotonic() - started_at,
        }
    except subprocess.TimeoutExpired:
        return {
            "cmd": display_cmd,
            "rc": -1,
            "out": "",
            "err": "Timeout.",
            "duration_sec": time.monotonic() - started_at,
            "timed_out": True,
        }
    except Exception as e:
        return {
            "cmd": display_cmd,
            "rc": -1,
            "out": "",
            "err": str(e),
            "duration_sec": time.monotonic() - started_at,
        }


def _enqueue_stream_output(name: str, stream: Any, sink: queue.Queue) -> None:
    try:
        for line in iter(stream.readline, ""):
            sink.put((name, line))
    finally:
        sink.put((name, None))
        stream.close()


def _run_live(cmd: List[str], cwd: Path = ROOT_DIR, timeout: int = 300,
              title: str = "Menjalankan command", max_visible_lines: int = 200) -> Dict:
    display_cmd = _format_cmd(cmd)
    stdout_lines: List[str] = []
    stderr_lines: List[str] = []
    stdout_tail = deque(maxlen=max_visible_lines)
    stderr_tail = deque(maxlen=max_visible_lines)
    started_at = time.monotonic()

    with st.status(title, state="running", expanded=True) as live_status:
        st.code(display_cmd, language="bash")
        meta_placeholder = st.empty()

        with st.expander("📄 Live Stdout", expanded=True):
            stdout_placeholder = st.empty()

        with st.expander("⚠️ Live Stderr", expanded=False):
            stderr_placeholder = st.empty()

        def render(state_label: str) -> None:
            duration = time.monotonic() - started_at
            meta_placeholder.caption(
                f"Status: {state_label} | Durasi: {duration:.1f}s | "
                f"Stdout: {len(stdout_lines)} baris | Stderr: {len(stderr_lines)} baris"
            )
            stdout_placeholder.code(
                "".join(stdout_tail).rstrip() or "(menunggu stdout...)",
                language="text",
            )
            stderr_placeholder.code(
                "".join(stderr_tail).rstrip() or "(belum ada stderr)",
                language="text",
            )

        render("menyiapkan proses")

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            err_text = str(e)
            stderr_lines.append(err_text)
            stderr_tail.append(err_text)
            render("gagal memulai")
            live_status.update(label=f"{title} gagal dimulai", state="error", expanded=True)
            return {
                "cmd": display_cmd,
                "rc": -1,
                "out": "",
                "err": err_text,
                "duration_sec": time.monotonic() - started_at,
            }

        stream_queue: queue.Queue = queue.Queue()
        stream_open = {"stdout": proc.stdout is not None, "stderr": proc.stderr is not None}

        for name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr)):
            if stream is not None:
                threading.Thread(
                    target=_enqueue_stream_output,
                    args=(name, stream, stream_queue),
                    daemon=True,
                ).start()

        timed_out = False
        last_render = 0.0

        while True:
            now = time.monotonic()
            if proc.poll() is None and timeout and now - started_at > timeout and not timed_out:
                timed_out = True
                proc.kill()
                timeout_msg = f"[ERROR] Timeout setelah {timeout} detik."
                stderr_lines.append(timeout_msg)
                stderr_tail.append(timeout_msg)

            saw_new_line = False
            while True:
                wait_time = 0.2 if not saw_new_line else 0.0
                try:
                    stream_name, line = stream_queue.get(timeout=wait_time)
                except queue.Empty:
                    break

                saw_new_line = True
                if line is None:
                    stream_open[stream_name] = False
                    continue

                if stream_name == "stdout":
                    stdout_lines.append(line)
                    stdout_tail.append(line)
                else:
                    stderr_lines.append(line)
                    stderr_tail.append(line)

            if saw_new_line or now - last_render >= 0.5:
                phase = "sedang berjalan" if proc.poll() is None and not timed_out else "menyelesaikan output"
                render(phase)
                last_render = now

            if proc.poll() is not None and not any(stream_open.values()) and stream_queue.empty():
                break

        rc = proc.wait() if proc.poll() is None else proc.returncode
        render("selesai" if rc == 0 and not timed_out else "gagal")
        live_status.update(
            label=f"{title} selesai" if rc == 0 and not timed_out else f"{title} gagal",
            state="complete" if rc == 0 and not timed_out else "error",
            expanded=True,
        )
        return {
            "cmd": display_cmd,
            "rc": 0 if rc == 0 and not timed_out else -1 if timed_out else rc,
            "out": "".join(stdout_lines),
            "err": "".join(stderr_lines),
            "duration_sec": time.monotonic() - started_at,
            "timed_out": timed_out,
        }


def _show_result(res: Dict, show_command: bool = True) -> None:
    if show_command:
        st.code(res["cmd"], language="bash")
    if res["rc"] == 0:
        st.success("✅ Berhasil")
    elif res.get("timed_out"):
        st.error("❌ Gagal karena timeout")
    else:
        st.error(f"❌ Gagal (exit {res['rc']})")
    if res.get("duration_sec") is not None:
        st.caption(f"Durasi eksekusi: {res['duration_sec']:.1f} detik")
    c1, c2 = st.columns(2)
    with c1:
        with st.expander("📄 Stdout", expanded=res["rc"] == 0):
            st.text(res["out"] or "(kosong)")
    with c2:
        with st.expander("⚠️ Stderr", expanded=res["rc"] != 0):
            st.text(res["err"] or "(kosong)")


def _format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_nested(data: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _hdfs_basename(hdfs_path: str) -> str:
    normalized = hdfs_path.strip().rstrip("/")
    if not normalized:
        return "hdfs_export"
    name = PurePosixPath(normalized).name
    return name or "hdfs_export"


def _prepare_hdfs_download(hdfs_path: str, timeout: int = 600) -> Dict[str, Any]:
    source_path = hdfs_path.strip()
    if not source_path:
        return {
            "cmd": "hdfs download",
            "rc": -1,
            "out": "",
            "err": "HDFS path tidak boleh kosong.",
            "duration_sec": 0.0,
        }

    started_at = time.monotonic()

    exists = _run(["hdfs", "dfs", "-test", "-e", source_path], timeout=timeout)
    if exists["rc"] != 0:
        return {
            "cmd": f"hdfs dfs -test -e {source_path}",
            "rc": exists["rc"],
            "out": exists["out"],
            "err": exists["err"] or f"Path HDFS tidak ditemukan: {source_path}",
            "duration_sec": time.monotonic() - started_at,
        }

    is_dir = _run(["hdfs", "dfs", "-test", "-d", source_path], timeout=timeout)["rc"] == 0

    temp_root = Path(tempfile.mkdtemp(prefix="hdfs_download_", dir="/tmp"))
    base_name = _hdfs_basename(source_path)

    if is_dir:
        local_parent = temp_root / "payload"
        local_parent.mkdir(parents=True, exist_ok=True)
        fetch_cmd = ["hdfs", "dfs", "-get", "-f", source_path, str(local_parent)]
        fetch_res = _run(fetch_cmd, timeout=timeout)
        if fetch_res["rc"] != 0:
            return fetch_res

        local_target = local_parent / base_name
        archive_path = shutil.make_archive(
            str(temp_root / base_name),
            "zip",
            root_dir=str(local_parent),
            base_dir=base_name,
        )
        archive_file = Path(archive_path)
        payload = archive_file.read_bytes()
        return {
            "cmd": fetch_res["cmd"],
            "rc": 0,
            "out": (
                f"Folder HDFS berhasil diambil dan di-zip.\n"
                f"Source: {source_path}\n"
                f"Local temp: {local_target}\n"
                f"Archive: {archive_file}"
            ),
            "err": fetch_res["err"],
            "duration_sec": time.monotonic() - started_at,
            "download_name": f"{base_name}.zip",
            "download_mime": "application/zip",
            "download_bytes": payload,
            "download_size": len(payload),
            "download_kind": "directory",
            "source_path": source_path,
        }

    local_target = temp_root / base_name
    fetch_cmd = ["hdfs", "dfs", "-get", "-f", source_path, str(local_target)]
    fetch_res = _run(fetch_cmd, timeout=timeout)
    if fetch_res["rc"] != 0:
        return fetch_res

    payload = local_target.read_bytes()
    mime_type, _ = mimetypes.guess_type(local_target.name)
    return {
        "cmd": fetch_res["cmd"],
        "rc": 0,
        "out": (
            f"File HDFS berhasil diambil.\n"
            f"Source: {source_path}\n"
            f"Local temp: {local_target}"
        ),
        "err": fetch_res["err"],
        "duration_sec": time.monotonic() - started_at,
        "download_name": local_target.name,
        "download_mime": mime_type or "application/octet-stream",
        "download_bytes": payload,
        "download_size": len(payload),
        "download_kind": "file",
        "source_path": source_path,
    }

# ── TAB: OVERVIEW ────────────────────────────────────────────────────────────
def tab_overview(cfg: Dict, ccfg: Dict) -> None:
    st.subheader("🗂️ Status Artifact")
    checks = {
        "📁 Dataset (Books_rating.csv)":      ML_DIR / "dataset" / "Books_rating.csv",
        "⚙️ Processed (train.csv)":           ML_DIR / "data" / "processed" / "train.csv",
        "🤖 Baseline Sentiment metrics.json": ML_DIR / "models" / "sentiment" / "baseline" / "metrics.json",
        "🔥 Transformer metrics.json":        ML_DIR / "models" / "sentiment" / "transformer" / "distilbert_v1" / "metrics.json",
        "📚 Recommender metrics.json":        ML_DIR / "reports" / "recommender_metrics.json",
        "⚖️ Comparison report":               ML_DIR / "reports" / "training_mode_comparison.json",
        "📋 Model Registry":                  ML_DIR / "models" / "model_registry.json",
        "📝 Final Report":                    ML_DIR / "reports" / "final_report.json",
    }
    rows = []
    for name, path in checks.items():
        ex = path.exists()
        sz = f"{path.stat().st_size/1024:.1f} KB" if ex else "—"
        rows.append({"Artifact": name, "Status": "✅ Ada" if ex else "❌ Belum", "Ukuran": sz})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    preprocess_meta = _load_json(ML_DIR / "data" / "processed" / "preprocess_metadata.json")
    if preprocess_meta:
        source = preprocess_meta.get("source", "unknown")
        source_label = (
            "Hasil Spark Submit di HDFS"
            if source == "spark_hdfs"
            else "Dataset lokal master"
            if source == "local_dataset"
            else source
        )
        st.caption(
            "Sumber preprocess terakhir: "
            f"{source_label} | rows={preprocess_meta.get('rows_processed', '—')} | "
            f"generated_at={preprocess_meta.get('generated_at_utc', '—')}"
        )

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

    compare = _load_json(ML_DIR / "reports" / "training_mode_comparison.json")
    if compare:
        st.subheader("⚖️ Ringkasan Perbandingan Mode")
        status = compare.get("status", "unknown")
        st.caption(f"Status komparasi: `{status}` | generated_at={compare.get('generated_at_utc', '—')}")
        comp = compare.get("comparison", {})
        if comp:
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Δ Durasi (with-without)",
                f"{comp.get('total_duration_delta_sec_with_minus_without', 0.0):.2f} s",
            )
            c2.metric(
                "Δ Sentiment F1 Test",
                f"{comp.get('sentiment_test_f1_delta_with_minus_without', 0.0):.4f}",
            )
            c3.metric(
                "Δ Recommender RMSE",
                f"{comp.get('recommender_rmse_delta_with_minus_without', 0.0):.4f}",
            )


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
    default_ram_limit = float(cfg.get("training", {}).get("master_ram_limit_gb", 3.0))
    default_run_worker_preprocess = bool(cfg.get("training", {}).get("auto_run_worker_preprocess", False))
    default_include_transformer = bool(cfg.get("training", {}).get("include_transformer_by_default", False))

    STEPS = [
        ("eda",                        "🔍 EDA",                            False),
        ("preprocess",                 "⚙️ Preprocess (Cleaning & Split)",  False),
        ("train_sentiment_baseline",   "🤖 Train Baseline Sentiment",       True),
        ("train_sentiment_transformer","🔥 Train Transformer (DistilBERT)", True),
        ("train_recommender",          "📚 Train Recommender System",       True),
        ("train_pipeline",             "🏁 Train Pipeline (Single Mode)",    True),
        ("compare_training_modes",     "⚖️ Compare: With Worker vs Without", True),
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

    preprocessing_applicable = step_key in {"eda", "preprocess", "all"}
    training_controls_applicable = step_key in {"train_pipeline", "compare_training_modes"} or (step_key == "all" and allow)

    preprocess_source = "local_dataset"
    if preprocessing_applicable:
        preprocess_source = st.selectbox(
            "Sumber Data",
            ["local_dataset", "spark_hdfs"],
            format_func=lambda v: (
                "Dataset Lokal Master — machine_learning/dataset/Books_rating.csv"
                if v == "local_dataset"
                else "Hasil Spark Submit di HDFS — output distributed preprocessing"
            ),
        )
        if preprocess_source == "spark_hdfs":
            st.info(
                "Mode ini memakai hasil `preprocess_spark` dari HDFS. Untuk step `preprocess` dan `all`, "
                "hasil tersebut akan dimaterialisasikan kembali ke `machine_learning/data/processed/` "
                "agar step training lokal bisa memakainya."
            )
        else:
            st.caption(
                "Mode ini membaca dataset mentah lokal di master dan menjalankan EDA/preprocessing dari dataset lokal."
            )

    training_mode = "without_worker"
    include_transformer = False
    run_worker_preprocess = False
    ram_limit_gb = default_ram_limit
    if training_controls_applicable:
        st.markdown("**Konfigurasi Training Mode**")
        training_mode = st.selectbox(
            "Training Mode",
            ["without_worker", "with_worker"],
            format_func=lambda v: (
                "without_worker — preprocess + training full di master"
                if v == "without_worker"
                else "with_worker — preprocess awal via Spark worker, training di master"
            ),
        )
        include_transformer = st.checkbox(
            "Sertakan Transformer (DistilBERT)",
            value=default_include_transformer,
            help="Jika aktif, mode training pipeline juga akan melatih model transformer.",
        )
        run_worker_preprocess = st.checkbox(
            "Jalankan Spark preprocess otomatis (khusus with_worker)",
            value=default_run_worker_preprocess,
            help=(
                "Jika aktif, pipeline akan memanggil scripts/spark_submit_training.sh preprocess_spark "
                "sebelum membaca hasil HDFS."
            ),
        )
        ram_limit_gb = st.number_input(
            "Batas RAM master (GB)",
            min_value=1.0,
            max_value=16.0,
            value=float(default_ram_limit),
            step=0.5,
            help="Batas memori proses training di master. Sesuai requirement eksperimen, gunakan 3GB.",
        )

        if training_mode == "with_worker":
            st.info(
                "Mode with_worker: worker hanya dipakai untuk preprocessing/cleaning distributed. "
                "Training model tetap berjalan di master."
            )

    st.markdown("**Konfigurasi Eksekusi**")
    default_timeout_sec = int(cfg.get("streamlit", {}).get("pipeline_timeout_sec", 10800))
    no_timeout_default = bool(step_key in {"all", "train_pipeline", "compare_training_modes"} and allow)
    col_exec_1, col_exec_2 = st.columns([2, 2])
    with col_exec_1:
        disable_timeout = st.checkbox(
            "Tanpa timeout (recommended untuk training panjang)",
            value=no_timeout_default,
        )
    with col_exec_2:
        pipeline_timeout = st.number_input(
            "Timeout Pipeline (detik)",
            min_value=60,
            max_value=86400,
            step=60,
            value=default_timeout_sec,
            disabled=disable_timeout,
            help="Batas tunggu di dashboard. Nonaktifkan timeout untuk training yang lama.",
        )

    st.caption(
        "Output terminal proses akan tampil live di bawah saat command berjalan (stdout & stderr)."
    )

    if st.button("▶️ Jalankan", type="primary", use_container_width=True):
        cmd = [_python_bin(), "machine_learning/main.py", "--step", step_key]
        if preprocessing_applicable:
            cmd.extend(["--preprocess-source", preprocess_source])
        if allow:
            cmd.append("--allow-training")
        if training_controls_applicable:
            cmd.extend(["--training-mode", training_mode])
            cmd.extend(["--ram-limit-gb", str(float(ram_limit_gb))])
            if include_transformer:
                cmd.append("--include-transformer")
            if run_worker_preprocess:
                cmd.append("--run-worker-preprocess")
        effective_timeout = 0 if disable_timeout else int(pipeline_timeout)
        res = _run_live(
            cmd,
            timeout=effective_timeout,
            title=f"Menjalankan: {step_label}",
            max_visible_lines=500,
        )
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

    # ---- Worker vs Tanpa Worker ----
    st.subheader("⚖️ Perbandingan Training Mode")
    comp_data = _load_json(ML_DIR / "reports" / "training_mode_comparison.json")
    if comp_data:
        runs = comp_data.get("runs") or {}
        run_count = len(runs)
        error_count = len(comp_data.get("errors") or {})
        warning_count = len(comp_data.get("warnings") or {})
        st.caption(
            f"Status: `{comp_data.get('status', 'unknown')}` | "
            f"Generated: {comp_data.get('generated_at_utc', '—')} | "
            f"Runs: {run_count} | Errors: {error_count} | Warnings: {warning_count}"
        )
        st.caption(
            f"run_worker_preprocess={comp_data.get('run_worker_preprocess', False)} | "
            f"master_ram_limit_gb={comp_data.get('master_ram_limit_gb', '—')}"
        )

        run_rows = []
        for mode in ["without_worker", "with_worker"]:
            run = runs.get(mode) or {}
            sent = run.get("sentiment_baseline_summary", {})
            rec = run.get("recommender_summary", {})
            worker_submit = run.get("worker_preprocess_submit") or {}
            run_rows.append(
                {
                    "Mode": mode,
                    "Status": run.get("status", "error"),
                    "Preprocess Source": run.get("preprocess_source", "—"),
                    "Worker Submit Enabled": bool(run.get("worker_preprocess_submit_enabled", False)),
                    "Worker Submit Duration (s)": round(_as_float(worker_submit.get("duration_sec")), 3) if worker_submit else None,
                    "Total Duration (s)": round(_as_float(run.get("total_duration_sec")), 3),
                    "Peak Memory (MB)": round(_as_float((run.get("peak_memory_mb") or {}).get("pipeline_peak")), 3),
                    "Sentiment Best Model": sent.get("best_model", "—"),
                    "Sentiment Val F1": round(_as_float(sent.get("best_validation_f1_weighted")), 6),
                    "Sentiment Test F1": round(_as_float(sent.get("best_test_f1_weighted")), 6),
                    "Sentiment Test Acc": round(_as_float(sent.get("best_test_accuracy")), 6),
                    "Rec RMSE": round(_as_float(rec.get("rmse")), 6),
                    "Rec MAE": round(_as_float(rec.get("mae")), 6),
                    "Rec Precision@5": round(_as_float(rec.get("precision_at_5")), 6),
                    "Rec Recall@5": round(_as_float(rec.get("recall_at_5")), 6),
                    "Rec NDCG@10": round(_as_float(rec.get("ndcg_at_10")), 6),
                }
            )
        st.markdown("**Ringkasan KPI per Mode:**")
        st.dataframe(pd.DataFrame(run_rows), use_container_width=True, hide_index=True)

        stage_rows = []
        for mode in ["without_worker", "with_worker"]:
            run = runs.get(mode) or {}
            stage_timings = run.get("timings_sec") or {}
            stage_mem = ((run.get("peak_memory_mb") or {}).get("stages") or {})
            for stage_name, duration in stage_timings.items():
                stage_rows.append(
                    {
                        "Mode": mode,
                        "Stage": stage_name,
                        "Duration (s)": round(_as_float(duration), 3),
                        "Peak Memory (MB)": round(_as_float(stage_mem.get(stage_name)), 3),
                    }
                )
        if stage_rows:
            st.markdown("**Detail Stage Timing & Peak Memory:**")
            stage_df = pd.DataFrame(stage_rows).sort_values(["Mode", "Duration (s)"], ascending=[True, False])
            st.dataframe(stage_df, use_container_width=True, hide_index=True)

        without_run = runs.get("without_worker") or {}
        with_run = runs.get("with_worker") or {}
        if without_run and with_run:
            metric_defs = [
                ("Total Duration (s)", ["total_duration_sec"]),
                ("Peak Memory (MB)", ["peak_memory_mb", "pipeline_peak"]),
                ("Sentiment Test F1", ["sentiment_baseline_summary", "best_test_f1_weighted"]),
                ("Sentiment Test Accuracy", ["sentiment_baseline_summary", "best_test_accuracy"]),
                ("Recommender RMSE", ["recommender_summary", "rmse"]),
                ("Recommender MAE", ["recommender_summary", "mae"]),
                ("Recommender Precision@5", ["recommender_summary", "precision_at_5"]),
                ("Recommender Recall@5", ["recommender_summary", "recall_at_5"]),
                ("Recommender NDCG@10", ["recommender_summary", "ndcg_at_10"]),
            ]
            delta_rows = []
            for metric_name, path in metric_defs:
                without_val = _as_float(_get_nested(without_run, path, 0.0))
                with_val = _as_float(_get_nested(with_run, path, 0.0))
                delta_val = with_val - without_val
                delta_pct = (delta_val / without_val * 100.0) if without_val != 0 else None
                delta_rows.append(
                    {
                        "Metric": metric_name,
                        "without_worker": round(without_val, 6),
                        "with_worker": round(with_val, 6),
                        "Delta (with - without)": round(delta_val, 6),
                        "Delta % vs without": f"{delta_pct:+.2f}%" if delta_pct is not None else "—",
                    }
                )
            st.markdown("**Perbandingan Detail (with_worker - without_worker):**")
            st.dataframe(pd.DataFrame(delta_rows), use_container_width=True, hide_index=True)

        with_worker_run = runs.get("with_worker") or {}
        worker_submit = with_worker_run.get("worker_preprocess_submit") or {}
        if worker_submit:
            st.markdown("**Detail Worker Preprocess Submit (with_worker):**")
            worker_rows = [
                {"Field": "Command", "Value": worker_submit.get("cmd", "—")},
                {"Field": "Return Code", "Value": worker_submit.get("rc", "—")},
                {"Field": "Duration (s)", "Value": round(_as_float(worker_submit.get("duration_sec")), 3)},
                {"Field": "Stdout Tail", "Value": (worker_submit.get("stdout_tail", "") or "—")[:1200]},
                {"Field": "Stderr Tail", "Value": (worker_submit.get("stderr_tail", "") or "—")[:1200]},
            ]
            st.dataframe(pd.DataFrame(worker_rows), use_container_width=True, hide_index=True)

        warnings = comp_data.get("warnings", {})
        if warnings:
            st.warning("Terdapat warning pada sebagian mode:")
            for mode, detail in warnings.items():
                with st.expander(f"Warning mode: {mode}", expanded=False):
                    st.text(detail.get("warning", "(tanpa warning)"))
                    if detail.get("original_error"):
                        st.caption("Original error:")
                        st.text(detail.get("original_error"))

        errors = comp_data.get("errors", {})
        if errors:
            st.warning("Sebagian mode gagal dijalankan. Detail error:")
            for mode, detail in errors.items():
                with st.expander(f"Error mode: {mode}", expanded=False):
                    st.text(detail.get("error", "(tanpa pesan error)"))
                    if detail.get("traceback"):
                        st.caption("Traceback:")
                        st.code(detail.get("traceback", ""), language="text")
    else:
        st.info("Belum ada laporan komparasi. Jalankan step `compare_training_modes` dari tab Pipeline.")

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
    available_user_ids = _load_recommender_user_ids()
    if available_user_ids:
        st.caption(f"Tersedia {len(available_user_ids)} user_id dari artifact recommender yang siap dipakai.")
        user_id = st.selectbox("Pilih User ID", available_user_ids, index=0)
    else:
        st.info("Belum ada daftar user_id. Latih recommender dulu agar pilihan user muncul otomatis.")
        user_id = st.text_input("User ID", placeholder="Masukkan user_id dari hasil training recommender...")

    top_n   = st.slider("Top-N", 1, 20, 10)
    if st.button("Ambil Rekomendasi"):
        if not str(user_id).strip():
            st.warning("Masukkan User ID terlebih dahulu.")
        else:
            try:
                from src.inference import recommend_for_user
                with st.spinner("Mengambil rekomendasi..."):
                    recs = recommend_for_user(cfg, str(user_id).strip(), top_n)
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
    spp_cfg  = cfg.get("spark_preprocess", {})

    # ---- Node Status ----
    st.subheader("🖥️ Status Node")
    if st.button("🔄 Cek Status SSH Worker"):
        status_rows = []
        for w in workers:
            res = _run_live(
                [
                    "ssh",
                    "-o", f"ConnectTimeout={timeout}",
                    "-o", "BatchMode=yes",
                    "-o", "StrictHostKeyChecking=no",
                    w,
                    "echo OK",
                ],
                timeout=timeout + 2,
                title=f"Mengecek SSH {w}",
            )
            ok = res["rc"] == 0 and "OK" in res["out"]
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
            res = _run_live(
                ["bash", str(ROOT_DIR / "scripts" / "start_cluster.sh")],
                title="Menjalankan start cluster",
            )
            _show_result(res, show_command=False)
    with c2:
        if st.button("⏹️ Stop Cluster", use_container_width=True):
            res = _run_live(
                ["bash", str(ROOT_DIR / "scripts" / "stop_cluster.sh")],
                title="Menjalankan stop cluster",
            )
            _show_result(res, show_command=False)

    # ---- Reset Training State ----
    st.subheader("🧹 Reset Training State")
    st.caption(
        "Menghapus artifact training lokal agar simulasi bisa dimulai dari awal lagi. "
        "Dataset mentah di `machine_learning/dataset/` tetap aman."
    )
    st.code(
        "\n".join(
            [
                "Yang dibersihkan:",
                "- machine_learning/data/processed",
                "- machine_learning/models/sentiment",
                "- machine_learning/models/recommender",
                "- machine_learning/reports",
                "- machine_learning/logs",
                "- machine_learning/mlruns",
                "- machine_learning/models/model_registry.json",
            ]
        ),
        language="text",
    )
    include_hadoop_reset = st.checkbox(
        "Sertakan reset Hadoop/HDFS/history di master dan semua worker",
        key="include_hadoop_reset",
    )
    if include_hadoop_reset:
        st.warning(
            "Mode ini sangat destruktif: cluster akan dihentikan, state NameNode/DataNode/YARN/log Hadoop "
            "dibersihkan, NameNode diformat ulang, dan dataset di HDFS harus di-upload kembali."
        )
        st.code(
            "\n".join(
                [
                    "Tambahan yang dibersihkan saat mode Hadoop aktif:",
                    "- /data/hadoop/hdfs/namenode (master)",
                    "- /data/hadoop/hdfs/datanode (master + worker)",
                    "- /data/hadoop/tmp (master + worker)",
                    "- /usr/local/hadoop/logs (master + worker)",
                    "- histori HDFS/YARN yang tersimpan pada state directory di atas",
                ]
            ),
            language="text",
        )
    confirm_reset = st.checkbox(
        "Saya paham reset ini akan menghapus history training lokal",
        key="confirm_reset_training",
    )
    confirm_hadoop_reset = True
    if include_hadoop_reset:
        confirm_hadoop_reset = st.checkbox(
            "Saya paham reset ini juga akan menghapus state Hadoop/HDFS/history cluster",
            key="confirm_reset_hadoop",
        )
    if st.button(
        "♻️ Reset Training",
        use_container_width=True,
        disabled=not confirm_reset or not confirm_hadoop_reset,
    ):
        cmd = ["bash", str(ROOT_DIR / "scripts" / "reset_training_state.sh")]
        if include_hadoop_reset:
            cmd.append("--include-hadoop-state")
        res = _run_live(
            cmd,
            timeout=1800 if include_hadoop_reset else 600,
            title="Membersihkan artifact training dan Hadoop" if include_hadoop_reset else "Membersihkan artifact training",
        )
        _show_result(res, show_command=False)

    # ---- Permission Fix ----
    st.subheader("🔐 Perbaiki Permission HDFS")
    st.caption("Mode ini bersifat permisif (world-writable) untuk mempermudah lab/testing cluster.")
    permission_target_user = st.text_input("Target user HDFS", value=getpass.getuser(), key="perm_target_user")
    if st.button("Fix Permission Worker1 & Worker2", use_container_width=True):
        res = _run_live(
            ["bash", str(ROOT_DIR / "scripts" / "fix_hdfs_permissions.sh"), permission_target_user],
            timeout=600,
            title="Memperbaiki permission HDFS",
        )
        _show_result(res, show_command=False)

    # ---- HDFS Upload ----
    st.subheader("📤 Upload Dataset ke HDFS")
    hdfs_target = st.text_input("HDFS Target Path", value=hdfs_cfg.get("dataset_path", _default_hdfs_dataset_path()))
    if st.button("Upload ke HDFS", use_container_width=True):
        res = _run_live(
            ["bash", str(ROOT_DIR / "scripts" / "upload_to_hdfs.sh"), hdfs_target],
            title="Mengupload dataset ke HDFS",
        )
        _show_result(res, show_command=False)

    # ---- Spark Submit ----
    st.subheader("⚡ Spark Submit (Distributed Preprocessing via YARN)")
    st.info(
        "Tab ini saat ini hanya menjalankan `preprocess_spark`. Output Spark ditulis ke HDFS, "
        "sedangkan training model (`train_sentiment_baseline`, `train_sentiment_transformer`, "
        "`train_recommender`) dan `final_report.json` tetap dibuat lewat pipeline lokal di master."
    )
    with st.expander("Konfigurasi Spark"):
        col1, col2, col3, col4 = st.columns(4)
        n_exec  = col1.number_input("num-executors",   value=int(sp_cfg.get("num_executors", 2)),   min_value=1, max_value=10)
        e_cores = col2.number_input("executor-cores",  value=int(sp_cfg.get("executor_cores", 2)),  min_value=1, max_value=8)
        e_mem   = col3.text_input("executor-memory", value=sp_cfg.get("executor_memory", "2G"))
        d_mem   = col4.text_input("driver-memory",   value=sp_cfg.get("driver_memory", "2G"))
        col5, col6, col7 = st.columns(3)
        sample_fraction = col5.number_input(
            "sample-fraction",
            value=float(spp_cfg.get("sample_fraction", 1.0)),
            min_value=0.01,
            max_value=1.0,
            step=0.05,
            format="%.2f",
        )
        output_partitions = col6.number_input(
            "output-partitions",
            value=int(spp_cfg.get("output_partitions", 0)),
            min_value=0,
            max_value=1000,
            step=1,
        )
        max_rows = col7.number_input(
            "max-rows (0 = full)",
            value=int(spp_cfg.get("max_rows", 0)),
            min_value=0,
            max_value=100000000,
            step=1000,
        )
        col8, col9 = st.columns(2)
        log_row_counts = col8.checkbox(
            "log row counts",
            value=bool(spp_cfg.get("log_row_counts", False)),
        )
        show_label_distribution = col9.checkbox(
            "show label distribution",
            value=bool(spp_cfg.get("show_label_distribution", False)),
        )
        spark_timeout = st.number_input(
            "submit-timeout (detik)",
            value=int(sp_cfg.get("submit_timeout_sec", 1800)),
            min_value=60,
            max_value=14400,
            step=60,
        )
        st.caption(
            "Timeout ini hanya batas tunggu dashboard. Wrapper `spark_submit_training.sh` "
            "sekarang menjalankan preflight YARN dan warning hostname sebelum submit. "
            "Gunakan `max-rows` untuk membatasi ukuran output Spark saat jaringan cluster lambat. "
            "Job ini tidak otomatis menyalin hasil ke folder `machine_learning/` di master."
        )

    spark_step = st.selectbox("Step Spark", [
        "preprocess_spark — Distributed Preprocessing via YARN",
    ])
    spark_step_key = spark_step.split("—")[0].strip()

    if st.button("🚀 Submit Spark Job", type="primary", use_container_width=True):
        cmd = [
            "env",
            f"YARN_PREFLIGHT_TIMEOUT={int(sp_cfg.get('preflight_timeout_sec', 20))}",
            f"SPARK_SAMPLE_FRACTION={float(sample_fraction):.4f}",
            f"SPARK_OUTPUT_PARTITIONS={int(output_partitions)}",
            f"SPARK_MAX_ROWS={int(max_rows)}",
            f"SPARK_LOG_ROW_COUNTS={'1' if log_row_counts else '0'}",
            f"SPARK_SHOW_LABEL_DISTRIBUTION={'1' if show_label_distribution else '0'}",
            "bash", str(ROOT_DIR / "scripts" / "spark_submit_training.sh"),
            spark_step_key,
            str(int(n_exec)), str(int(e_cores)), e_mem, d_mem,
        ]
        res = _run_live(
            cmd,
            timeout=int(spark_timeout),
            title="Submitting Spark job ke YARN",
        )
        _show_result(res, show_command=False)

    # ---- HDFS file browser ----
    st.subheader("🗂️ Browse HDFS")
    default_hdfs_path = hdfs_cfg.get("dataset_path", _default_hdfs_dataset_path())
    hdfs_path = st.text_input("HDFS Path", value=default_hdfs_path)
    st.caption(
        "Ambil data HDFS secara manual dengan `hdfs dfs -get <hdfs_path> <lokasi_lokal>`. "
        "Untuk output Spark berupa Parquet folder, ambil seluruh foldernya, bukan hanya satu part file."
    )
    st.code(
        "\n".join(
            [
                f"hdfs dfs -ls -h {hdfs_path}",
                f"hdfs dfs -get {hdfs_path} ./hasil_dari_hdfs",
            ]
        ),
        language="bash",
    )
    if st.button("List HDFS"):
        res = _run_live(
            ["hdfs", "dfs", "-ls", "-h", hdfs_path],
            title="Membaca isi path HDFS",
        )
        _show_result(res, show_command=False)

    st.subheader("⬇️ Download Dari HDFS")
    st.caption(
        "Masukkan file atau folder HDFS. Jika target adalah folder seperti output Parquet Spark, "
        "dashboard akan mengunduh lalu mengemasnya menjadi `.zip`."
    )
    download_hdfs_path = st.text_input(
        "HDFS Path untuk Download",
        value=hdfs_cfg.get("output_path", _default_hdfs_output_path()),
        key="hdfs_download_path",
    )
    download_timeout = st.number_input(
        "download-timeout (detik)",
        value=600,
        min_value=60,
        max_value=14400,
        step=60,
        key="hdfs_download_timeout",
    )
    if st.button("Siapkan Download HDFS", use_container_width=True):
        download_res = _prepare_hdfs_download(download_hdfs_path, timeout=int(download_timeout))
        if download_res["rc"] == 0:
            st.session_state["hdfs_download_payload"] = download_res
        else:
            st.session_state.pop("hdfs_download_payload", None)
        _show_result(download_res, show_command=False)

    prepared_download = st.session_state.get("hdfs_download_payload")
    if prepared_download and prepared_download.get("source_path") == download_hdfs_path:
        st.success(
            "File siap diunduh: "
            f"{prepared_download['download_name']} "
            f"({_format_size(int(prepared_download['download_size']))})"
        )
        if prepared_download.get("download_kind") == "directory":
            st.info("Target HDFS berupa folder. Dashboard mengirimkannya sebagai arsip `.zip`.")
        st.download_button(
            "Download HDFS Sekarang",
            data=prepared_download["download_bytes"],
            file_name=prepared_download["download_name"],
            mime=prepared_download["download_mime"],
            use_container_width=True,
            key="download_hdfs_button",
        )


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
