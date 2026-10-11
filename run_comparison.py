# Companion benchmark exported from the executed comparison notebook.
# Run from the repository root; existing results are replaced.

import os
from pathlib import Path
ROOT = Path.cwd()
OUT = ROOT / "results"
IMAGES = ROOT / "images"
for folder in [OUT, IMAGES, ROOT / ".model_cache"]:
    folder.mkdir(exist_ok=True)
os.environ["TABPFN_MODEL_CACHE_DIR"] = str(ROOT / ".model_cache")
os.environ["HF_HOME"] = str(ROOT / ".model_cache" / "huggingface")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
os.environ["SKB_DATA_DIRECTORY"] = str(ROOT / ".model_cache" / "skrub")
ALLOW_PENDING_TABPFN = False  # Require all four models; never report an incomplete comparison as complete.
THREADS = 4
for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[key] = str(THREADS)

import gc, json, time, platform, hashlib, importlib.metadata as metadata
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, log_loss
from threadpoolctl import threadpool_limits
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from tabpfn.errors import TabPFNLicenseError
torch.set_num_threads(THREADS)
torch.set_num_interop_threads(1)

SEEDS = [42, 43, 44, 45, 46]
TEST_SIZE = 0.25
MODEL_NAMES = ["Logistic Regression", "Random Forest", "CatBoost", "TabPFN-3.5"]
CHECKPOINT = "tabpfn-v3.5-20260909.safetensors"
def processor_name():
    if platform.system() == "Windows":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, "HARDWARE" + chr(92) + "DESCRIPTION" + chr(92) + "System" + chr(92) + "CentralProcessor" + chr(92) + "0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor()
environment = {
    "run_date_jakarta": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
    "python": platform.python_version(), "os": platform.platform(),
    "processor": processor_name(), "logical_cpu_count": os.cpu_count(),
    "threads": THREADS, "device": "cpu", "cuda_available": torch.cuda.is_available(),
    "checkpoint": CHECKPOINT, "seeds": SEEDS, "test_size": TEST_SIZE,
    "versions": {p: metadata.version(p) for p in ["tabpfn", "torch", "catboost", "scikit-learn", "numpy", "pandas", "scipy", "matplotlib", "nbformat", "nbclient", "ipykernel", "huggingface-hub", "safetensors", "skrub"]},
}
(OUT / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
display(pd.Series(environment["versions"], name="version").to_frame())
print("Device: CPU | compute threads:", THREADS)

datasets = {"Breast Cancer": load_breast_cancer(as_frame=True), "Wine": load_wine(as_frame=True)}
dataset_rows = []
for name, data in datasets.items():
    assert data.data.isna().sum().sum() == 0
    digest = hashlib.sha256(np.ascontiguousarray(data.data.to_numpy()).tobytes() + np.ascontiguousarray(data.target.to_numpy()).tobytes()).hexdigest()
    dataset_rows.append({"dataset": name, "rows": len(data.data), "features": data.data.shape[1], "classes": data.target.nunique(), "class_counts": data.target.value_counts().sort_index().to_dict(), "sha256_values": digest})
dataset_info = pd.DataFrame(dataset_rows)
dataset_info.to_json(OUT / "datasets.json", orient="records", indent=2)
display(dataset_info.drop(columns="sha256_values"))

def make_model(name, seed):
    if name == "Logistic Regression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=seed))
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=THREADS)
    if name == "CatBoost":
        return CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                                 random_seed=seed, thread_count=THREADS,
                                 verbose=False, allow_writing_files=False, task_type="CPU")
    if name == "TabPFN-3.5":
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V3_5, model_path=str(ROOT / ".model_cache" / CHECKPOINT),
            n_estimators=4, device="cpu", random_state=seed,
            n_preprocessing_jobs=1, show_progress_bar=False)
    raise ValueError(name)

settings = {name: make_model(name, 42).get_params(deep=False) for name in MODEL_NAMES}
(OUT / "model_settings.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
display(pd.DataFrame({"model": MODEL_NAMES, "settings": [str(settings[n]) for n in MODEL_NAMES]}))

warm_data = datasets["Wine"]
warm_ix, _ = train_test_split(np.arange(len(warm_data.data)), train_size=36, stratify=warm_data.target, random_state=100)
warmups = []
available_models = MODEL_NAMES.copy()
model_status = {name: "measured" for name in MODEL_NAMES}
with threadpool_limits(limits=THREADS):
    for name in MODEL_NAMES:
        start = time.perf_counter()
        model = make_model(name, 100)
        try:
            model.fit(warm_data.data.iloc[warm_ix], warm_data.target.iloc[warm_ix])
            model.predict_proba(warm_data.data.iloc[warm_ix[:3]])
        except TabPFNLicenseError:
            if name != "TabPFN-3.5" or not ALLOW_PENDING_TABPFN:
                raise
            available_models.remove(name)
            model_status[name] = "pending: official weight download requires license acceptance and authentication"
            print("TabPFN-3.5 PENDING: enable licensed access and rerun. No TabPFN scores will be produced.")
            del model
            continue
        warmups.append({"model": name, "setup_seconds": time.perf_counter() - start})
        del model
        gc.collect()
pd.DataFrame(warmups).to_csv(OUT / "setup_timings.csv", index=False)
checkpoint_path = ROOT / ".model_cache" / CHECKPOINT
if "TabPFN-3.5" in available_models:
    assert checkpoint_path.is_file(), "Requested 3.5 checkpoint was not downloaded."
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    environment["checkpoint_sha256"] = digest.hexdigest()
    environment["checkpoint_bytes"] = checkpoint_path.stat().st_size
environment["model_status"] = model_status
(OUT / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
display(pd.DataFrame(warmups))
print("Checkpoint:", CHECKPOINT)
print("Status:", model_status)
print("SHA256:", environment.get("checkpoint_sha256", "pending access; no weights downloaded"))

rows, split_indices = [], {}
for dataset_name, data in datasets.items():
    X, y = data.data, data.target
    for seed in SEEDS:
        train_ix, test_ix = train_test_split(np.arange(len(X)), test_size=TEST_SIZE, stratify=y, random_state=seed)
        assert not set(train_ix) & set(test_ix)
        split_indices[f"{dataset_name}_{seed}"] = {"train": train_ix.tolist(), "test": test_ix.tolist()}
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
        order = np.random.default_rng(seed).permutation(available_models)
        for name in order:
            gc.collect()
            with threadpool_limits(limits=THREADS):
                start = time.perf_counter()
                model = make_model(name, seed)
                constructed = time.perf_counter()
                model.fit(X_train, y_train)
                fitted = time.perf_counter()
                probabilities = model.predict_proba(X_test)
                predicted = time.perf_counter()
            classes = np.asarray(model.classes_)
            assert probabilities.shape == (len(test_ix), len(classes))
            assert np.isfinite(probabilities).all()
            assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
            labels = classes[probabilities.argmax(axis=1)]
            auc = roc_auc_score(y_test, probabilities[:, 1]) if len(classes) == 2 else roc_auc_score(y_test, probabilities, labels=classes, multi_class="ovr", average="macro")
            row = {"dataset": dataset_name, "seed": seed, "model": name,
                   "n_train": len(train_ix), "n_test": len(test_ix),
                   "accuracy": accuracy_score(y_test, labels),
                   "balanced_accuracy": balanced_accuracy_score(y_test, labels),
                   "macro_f1": f1_score(y_test, labels, average="macro"),
                   "roc_auc": auc, "log_loss": log_loss(y_test, probabilities, labels=classes),
                   "construct_seconds": constructed-start, "fit_seconds": fitted-constructed,
                   "predict_seconds": predicted-fitted, "total_seconds": predicted-start}
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT / "per_split_results.csv", index=False)
            pred_frame = pd.DataFrame({"row_index": test_ix, "actual": y_test.to_numpy(), "prediction": labels})
            for col, cls in enumerate(classes):
                pred_frame[f"probability_class_{cls}"] = probabilities[:, col]
            stem = f"{dataset_name.lower().replace(' ', '_')}_{seed}_{name.lower().replace(' ', '_')}"
            pred_frame.to_csv(OUT / f"predictions_{stem}.csv", index=False)
            print(f"{dataset_name} | seed {seed} | {name}: BA={row['balanced_accuracy']:.4f}, total={row['total_seconds']:.3f}s", flush=True)
            del model
        (OUT / "split_indices.json").write_text(json.dumps(split_indices, indent=2), encoding="utf-8")
results = pd.DataFrame(rows)
assert len(results) == 2 * len(SEEDS) * len(available_models)
assert results.groupby(["dataset", "seed"])["model"].nunique().eq(len(available_models)).all()
display(results)

metrics = ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "log_loss"]
summary = results.groupby(["dataset", "model"])[metrics].agg(["mean", "std"])
summary.columns = ["_".join(c) for c in summary.columns]
timing = results.groupby(["dataset", "model"])[["construct_seconds", "fit_seconds", "predict_seconds", "total_seconds"]].median().add_suffix("_median")
summary = summary.join(timing).reset_index()
summary.to_csv(OUT / "summary.csv", index=False)
score_table = summary[["dataset", "model"]].copy()
for metric in metrics:
    score_table[metric] = summary.apply(lambda r: f"{r[metric+'_mean']:.4f} ± {r[metric+'_std']:.4f}", axis=1)
display(score_table)
display(summary[["dataset", "model", *timing.columns]].round(4))
if "TabPFN-3.5" in available_models:
    wide = results.pivot(index=["dataset", "seed"], columns="model", values="balanced_accuracy")
    deltas = wide.drop(columns="TabPFN-3.5").apply(lambda c: wide["TabPFN-3.5"] - c)
    deltas.to_csv(OUT / "paired_balanced_accuracy_deltas.csv")
    display(deltas.groupby(level="dataset").agg(["mean", "std"]))
else:
    print("TabPFN-3.5 performance and runtime: PENDING. No head-to-head verdict is possible.")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
colors = ["#52616B", "#3975A8", "#D69038", "#168477"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
for ax, dataset_name in zip(axes, datasets):
    sub = summary[summary.dataset.eq(dataset_name)].set_index("model").reindex(MODEL_NAMES)
    ax.barh(MODEL_NAMES, sub.balanced_accuracy_mean, xerr=sub.balanced_accuracy_std, color=colors, capsize=4)
    if "TabPFN-3.5" not in available_models:
        ax.text(0.02, 3, "Pending licensed access", va="center", color="#168477")
    ax.set_xlim(0, 1.04)
    ax.set_xlabel("Balanced accuracy (mean ± split SD)")
    ax.set_title(dataset_name)
    ax.invert_yaxis()
    ax.set_ylim(3.5, -0.5)
fig.suptitle("Fixed configurations on two small numeric datasets", fontsize=15)
fig.savefig(IMAGES / "performance.png", dpi=200, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
for ax, dataset_name in zip(axes, datasets):
    sub = summary[summary.dataset.eq(dataset_name)].set_index("model").reindex(MODEL_NAMES)
    ax.barh(MODEL_NAMES, sub.total_seconds_median, color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("Median total runtime (seconds, log scale)")
    ax.set_title(dataset_name)
    ax.invert_yaxis()
    ax.set_ylim(3.5, -0.5)
    ax.margins(x=0.22)
    for i, val in enumerate(sub.total_seconds_median):
        if np.isfinite(val):
            ax.text(val*1.08, i, f"{val:.3f}s", va="center", fontsize=10)
    if "TabPFN-3.5" not in available_models:
        ax.text(0.03, 3, "Pending licensed access", transform=ax.get_yaxis_transform(), va="center", color="#168477")
fig.suptitle("CPU runtime • 4 compute threads • 5 splits", fontsize=15)
fig.savefig(IMAGES / "runtime.png", dpi=200, bbox_inches="tight")
plt.show()