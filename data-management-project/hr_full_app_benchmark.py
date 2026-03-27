# %pip install --upgrade pip
# %pip install --upgrade pandas numpy sqlalchemy psycopg2-binary scikit-learn xgboost tabm rtdl_num_embeddings joblib psutil nvidia-ml-py optuna matplotlib
# %pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# ---------------------------------------------------------
# 1. GLOBAL CONFIGURATION
# 2. DEVICE DETECTION GPU/CPU
# 3. RESOURCE MONITORING
# 4. DATABASE 
# 5. DATA PREPARATION
# 6. METRICS AND PLOTS
# 7. DEFINE XGBOOST RUN
# 8. DEFINE TABM RUN
# 9. MAIN FUNCTION
#       a. Shared transformed data
#       b. XGBoost optuna tuning 
#       c. Final XGBoost training + monitoring
#	    d. TabM tuning
#	    e. Final TabM training + monitoring
#	    f. Final comparison / export
# ---------------------------------------------------------
import copy, json, os, random, urllib.parse, warnings, joblib, optuna, psutil
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event, Thread
from time import perf_counter, sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlalchemy as sa
import torch
import torch.nn.functional as F
import xgboost as xgb
from rtdl_num_embeddings import LinearReLUEmbeddings
from sklearn.metrics import (accuracy_score,average_precision_score,f1_score,log_loss,precision_recall_curve,precision_score,recall_score,roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from sqlalchemy.sql.elements import quoted_name
from tabm import TabM
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

try:
    import pynvml
    PYNVML_AVAILABLE = True
except Exception:
    pynvml = None
    PYNVML_AVAILABLE = False

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =========================================================
# 1. PATHS / GLOBAL CONFIG
# =========================================================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

INPUT_DIR = BASE_DIR / "Input"
MODEL_DIR = BASE_DIR / "model_save" / "model"
OUTPUT_DIR = BASE_DIR / "Output"
PLOTS_DIR = OUTPUT_DIR / "plots"
OPTUNA_DIR = OUTPUT_DIR / "optuna"
THRESHOLD_DIR = OUTPUT_DIR / "thresholds"

DATA_ENV_PATH = INPUT_DIR / "data.env"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
THRESHOLD_DIR.mkdir(parents=True, exist_ok=True)

HR_TABLE = "HR_Synth_Data"
COMPARISON_CSV_PATH = OUTPUT_DIR / "model_comparison_tabm_vs_xgboost.csv"
RESOURCE_SUMMARY_CSV_PATH = OUTPUT_DIR / "benchmark_resource_summary.csv"
RESOURCE_SAMPLES_CSV_PATH = OUTPUT_DIR / "benchmark_resource_samples.csv"
XGB_PREDICTIONS_CSV_PATH = OUTPUT_DIR / "predictions_xgboost.csv"
TABM_PREDICTIONS_CSV_PATH = OUTPUT_DIR / "predictions_tabm.csv"
XGB_IMPORTANCE_CSV_PATH = OUTPUT_DIR / "xgboost_feature_importance.csv"

SEED = 42
TEST_SIZE = 0.20
VALID_SIZE = 0.25  # z train casti -> 60/20/20 split
RESOURCE_SAMPLE_INTERVAL_SEC = 0.1
USE_SQL_EXPORT = False

RUN_XGB_OPTUNA = True
RUN_TABM_OPTUNA = True
XGB_OPTUNA_TRIALS = 100
TABM_OPTUNA_TRIALS = 100

DEFAULT_FIXED_THRESHOLD = 0.50
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01
THRESHOLD_SELECTION_METRIC = "f1"  # options: accuracy, f1, precision, recall, specificity, balanced_accuracy, youden_j

XGB_EARLY_STOPPING_ROUNDS = 25
TABM_FINAL_EPOCHS = 250
TABM_FINAL_PATIENCE = 50
TABM_TUNE_EPOCHS = 100
TABM_TUNE_PATIENCE = 25

# =========================================================
# 2. DEVICE DETECTION GPU/CPU
# =========================================================
def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_global_seed(SEED)

def get_torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_xgboost_device() -> str:
    try:
        X_test = np.array([[1, 2], [3, 4]], dtype=np.float32)
        y_test = np.array([0, 1], dtype=np.int32)
        test_model = XGBClassifier(device="cuda", n_estimators=1, tree_method="hist")
        test_model.fit(X_test, y_test)
        print("GPU na CUDA bol detekovany pre XGBoost.")
        return "cuda"
    except Exception:
        print("GPU na CUDA nebol detekovany pre XGBoost. Prepinam na CPU mode.")
        return "cpu"


TORCH_DEVICE = get_torch_device()
XGB_DEVICE = get_xgboost_device()
print(f"TORCH_DEVICE = {TORCH_DEVICE}")
print(f"XGB_DEVICE   = {XGB_DEVICE}")
print(f"PYNVML_AVAILABLE = {PYNVML_AVAILABLE}")

# =========================================================
# 3. RESOURCE MONITORING
# =========================================================
def bytes_to_gb(value: float | int) -> float:
    return round(float(value) / (1024 ** 3), 4)

class ResourceMonitor:
    def __init__(self, label: str, sample_interval: float = RESOURCE_SAMPLE_INTERVAL_SEC):
        self.label = label
        self.sample_interval = sample_interval
        self.process = psutil.Process(os.getpid())
        self._stop_event = Event()
        self._thread = None
        self._nvml_initialized_here = False
        self._gpu_handle = None
        self.start_ts = None
        self.end_ts = None
        self.samples: list[dict] = []

    def _init_gpu(self) -> None:
        if not torch.cuda.is_available() or not PYNVML_AVAILABLE:
            return
        try:
            pynvml.nvmlInit()
            self._nvml_initialized_here = True
            gpu_index = torch.cuda.current_device()
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception:
            self._gpu_handle = None
            self._nvml_initialized_here = False

    def _shutdown_gpu(self) -> None:
        if self._nvml_initialized_here and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _sample_once(self) -> None:
        vm = psutil.virtual_memory()
        sample = {
                    "label": self.label,
                    "ts": perf_counter(),
                    "process_cpu_percent": self.process.cpu_percent(interval=None),
                    "system_cpu_percent": psutil.cpu_percent(interval=None),
                    "process_rss_gb": bytes_to_gb(self.process.memory_info().rss),
                    "system_used_ram_gb": bytes_to_gb(vm.used),
                    "system_ram_percent": float(vm.percent),
                    "gpu_util_percent": np.nan,
                    "gpu_mem_util_percent": np.nan,
                    "gpu_used_vram_gb": np.nan,
                    "gpu_total_vram_gb": np.nan,
                    }

        if self._gpu_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                sample["gpu_util_percent"] = float(util.gpu)
                sample["gpu_mem_util_percent"] = float(util.memory)
                sample["gpu_used_vram_gb"] = bytes_to_gb(mem.used)
                sample["gpu_total_vram_gb"] = bytes_to_gb(mem.total)
            except Exception:
                pass

        self.samples.append(sample)

    def _run(self) -> None:
        self.process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        while not self._stop_event.is_set():
            self._sample_once()
            sleep(self.sample_interval)

    def start(self) -> None:
        self.start_ts = perf_counter()
        self._init_gpu()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.end_ts = perf_counter()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval * 4))
        self._shutdown_gpu()

    def summary(self) -> dict[str, float | str]:
        df = pd.DataFrame(self.samples)
        duration_sec = round((self.end_ts - self.start_ts), 4) if self.start_ts and self.end_ts else np.nan

        result = {
                    "label": self.label,
                    "duration_sec": duration_sec,
                    "samples": int(len(df)),
                    "cpu_process_avg_pct": np.nan,
                    "cpu_process_peak_pct": np.nan,
                    "cpu_system_avg_pct": np.nan,
                    "cpu_system_peak_pct": np.nan,
                    "rss_avg_gb": np.nan,
                    "rss_peak_gb": np.nan,
                    "system_ram_avg_pct": np.nan,
                    "system_ram_peak_pct": np.nan,
                    "gpu_util_avg_pct": np.nan,
                    "gpu_util_peak_pct": np.nan,
                    "gpu_mem_util_avg_pct": np.nan,
                    "gpu_mem_util_peak_pct": np.nan,
                    "gpu_used_vram_avg_gb": np.nan,
                    "gpu_used_vram_peak_gb": np.nan,
                    "torch_peak_allocated_gb": np.nan,
                    "torch_peak_reserved_gb": np.nan,
                    }

        if not df.empty:
            result.update({
                        "cpu_process_avg_pct": round(float(df["process_cpu_percent"].mean()), 4),
                        "cpu_process_peak_pct": round(float(df["process_cpu_percent"].max()), 4),
                        "cpu_system_avg_pct": round(float(df["system_cpu_percent"].mean()), 4),
                        "cpu_system_peak_pct": round(float(df["system_cpu_percent"].max()), 4),
                        "rss_avg_gb": round(float(df["process_rss_gb"].mean()), 4),
                        "rss_peak_gb": round(float(df["process_rss_gb"].max()), 4),
                        "system_ram_avg_pct": round(float(df["system_ram_percent"].mean()), 4),
                        "system_ram_peak_pct": round(float(df["system_ram_percent"].max()), 4),
            })

            if df["gpu_util_percent"].notna().any():
                result.update({
                    "gpu_util_avg_pct": round(float(df["gpu_util_percent"].mean()), 4),
                    "gpu_util_peak_pct": round(float(df["gpu_util_percent"].max()), 4),
                    "gpu_mem_util_avg_pct": round(float(df["gpu_mem_util_percent"].mean()), 4),
                    "gpu_mem_util_peak_pct": round(float(df["gpu_mem_util_percent"].max()), 4),
                    "gpu_used_vram_avg_gb": round(float(df["gpu_used_vram_gb"].mean()), 4),
                    "gpu_used_vram_peak_gb": round(float(df["gpu_used_vram_gb"].max()), 4),
                })

        if torch.cuda.is_available():
            try:
                result["torch_peak_allocated_gb"] = round(bytes_to_gb(torch.cuda.max_memory_allocated()), 4)
                result["torch_peak_reserved_gb"] = round(bytes_to_gb(torch.cuda.max_memory_reserved()), 4)
            except Exception:
                pass

        return result

    def samples_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.samples)
        if not df.empty:
            df["seconds_from_start"] = df["ts"] - df["ts"].min()
        return df

# =========================================================
# 4. DATABASE
# =========================================================
def load_data_env(env_path: Path | None = None) -> None:
    env_path = env_path or DATA_ENV_PATH

    print(f"Nahravam data.env zo suboru: {env_path}")
    if not env_path.exists():
        raise FileNotFoundError(f"Subor data.env nebol najdeny: {env_path}")

    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()

def qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'

def postgre_schema() -> str:
    return os.getenv("PG_SCHEMA", "public")

def postgre_engine() -> sa.Engine:
    load_data_env(DATA_ENV_PATH)

    required = ["PG_HOST", "PG_PORT", "PG_DATABASE", "PG_USERNAME", "PG_PASSWORD"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Chybaju premenne prostredia: {missing}")

    host = os.getenv("PG_HOST")
    port = os.getenv("PG_PORT", "5432")
    database = os.getenv("PG_DATABASE")
    username = os.getenv("PG_USERNAME")
    password = urllib.parse.quote_plus(os.getenv("PG_PASSWORD"))

    return sa.create_engine(
        f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}",pool_pre_ping=True)

def load_hr_dataframe() -> pd.DataFrame:
    engine = postgre_engine()
    schema = postgre_schema()
    query = text(f"SELECT * FROM {qident(schema)}.{qident(HR_TABLE)}")
    df = pd.read_sql(query, engine)
    print(f"Bolo nacitanych {len(df)} riadkov z {schema}.{HR_TABLE}")
    return df

def save_to_sql(df: pd.DataFrame, table_name: str) -> None:
    engine = postgre_engine()
    schema = postgre_schema()
    print(f"Zapisujem {len(df)} riadkov do {schema}.{table_name}")
    df.to_sql(
            quoted_name(table_name, True),
            engine,
            schema=schema,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=1000,
            )

# =========================================================
# 5. DATA PREPARATION
# =========================================================
def get_drop_columns() -> list[str]:
    return [
            "EmployeeNumber",
            "EmployeeCount",
            "StandardHours",
            "Over18",
            "FirstName",
            "LastName",
            "FullName",
            "Email",
            "Username",
            "DailyRate",
            "HourlyRate",
            "MonthlyRate",
            "EnvironmentSatisfaction",
            "RelationshipSatisfaction",
            "JobSatisfaction",
            "JobLevel",
            "WorkLifeBalance",
            "JobInvolvement",
            ]

def target_to_binary(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).fillna(0).astype(int)

def prepare_base_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if "Attrition" not in df.columns:
        raise ValueError("V datasete chyba cielovy stlpec 'Attrition'.")

    y = target_to_binary(df["Attrition"])
    meta_cols = [c for c in ["EmployeeNumber", "FullName", "Department", "JobRole", "Attrition"] if c in df.columns]
    meta_df = df[meta_cols].copy()

    X = df.drop(columns=["Attrition"], errors="ignore").copy()
    X = X.drop(columns=[c for c in get_drop_columns() if c in X.columns], errors="ignore")

    return X, y, meta_df

@dataclass
class TabularPreprocessor:
    numeric_cols: list[str]
    categorical_cols: list[str]
    numeric_fill_values: dict[str, float]
    categorical_mappings: dict[str, dict[str, int]]
    categorical_cardinalities: list[int]
    scaler: StandardScaler

    @classmethod
    def fit(cls, X_train: pd.DataFrame) -> "TabularPreprocessor":
        numeric_cols = X_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

        numeric_fill_values: dict[str, float] = {}
        for col in numeric_cols:
            numeric_fill_values[col] = float(pd.to_numeric(X_train[col], errors="coerce").median())

        X_num_train = pd.DataFrame(index=X_train.index)
        for col in numeric_cols:
            X_num_train[col] = pd.to_numeric(X_train[col], errors="coerce").fillna(numeric_fill_values[col])

        scaler = StandardScaler()
        if numeric_cols:
            scaler.fit(X_num_train[numeric_cols])

        categorical_mappings: dict[str, dict[str, int]] = {}
        categorical_cardinalities: list[int] = []
        for col in categorical_cols:
            values = X_train[col].fillna("<NA>").astype(str).unique().tolist()
            mapping = {v: i + 1 for i, v in enumerate(sorted(values))}
            categorical_mappings[col] = mapping
            categorical_cardinalities.append(len(mapping) + 1)

        return cls(
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                    numeric_fill_values=numeric_fill_values,
                    categorical_mappings=categorical_mappings,
                    categorical_cardinalities=categorical_cardinalities,
                    scaler=scaler,
                    )

    def transform_for_xgboost(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)

        for col in self.numeric_cols:
            out[col] = pd.to_numeric(X[col], errors="coerce").fillna(self.numeric_fill_values[col]).astype(float)

        for col in self.categorical_cols:
            mapping = self.categorical_mappings[col]
            out[col] = X[col].fillna("<NA>").astype(str).map(mapping).fillna(0).astype(int)

        ordered_cols = self.numeric_cols + self.categorical_cols
        return out[ordered_cols].copy()

    def transform_for_tabm(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X_num = pd.DataFrame(index=X.index)
        for col in self.numeric_cols:
            X_num[col] = pd.to_numeric(X[col], errors="coerce").fillna(self.numeric_fill_values[col]).astype(float)

        if self.numeric_cols:
            x_num = self.scaler.transform(X_num[self.numeric_cols]).astype(np.float32)
        else:
            x_num = np.zeros((len(X), 0), dtype=np.float32)

        x_cat_parts = []
        for col in self.categorical_cols:
            mapping = self.categorical_mappings[col]
            x_cat_parts.append(X[col].fillna("<NA>").astype(str).map(mapping).fillna(0).astype(np.int64).to_numpy())

        if x_cat_parts:
            x_cat = np.column_stack(x_cat_parts).astype(np.int64)
        else:
            x_cat = np.zeros((len(X), 0), dtype=np.int64)

        return x_num, x_cat

# =========================================================
# 6. METRICS AND PLOTS
# =========================================================
def evaluate_binary_model(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = DEFAULT_FIXED_THRESHOLD) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "pr_auc": float(average_precision_score(y_true, y_prob)),
                "logloss": float(log_loss(y_true, y_prob, labels=[0, 1])),
                }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["roc_auc"] = float("nan")

    return metrics

def print_metric_block(model_name: str, metrics: dict[str, float]) -> None:
    print(f"{model_name}")
    for key in ["accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc", "logloss"]:
        value = metrics.get(key)
        print(f"{key:>10}: {value:.6f}")

def save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_threshold_grid() -> np.ndarray:
    return np.round(np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP / 2, THRESHOLD_STEP), 4)

def compute_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn

def build_threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    dataset_name: str,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    thresholds = thresholds if thresholds is not None else get_threshold_grid()
    rows = []
    positive_rate = float(np.mean(y_true))

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tp, tn, fp, fn = compute_confusion_counts(y_true, y_pred)

        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        accuracy = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        balanced_accuracy = float((recall + specificity) / 2.0)
        youden_j = float(recall + specificity - 1.0)
        predicted_positive_rate = float(np.mean(y_pred))

        rows.append({
                    "Model": model_name,
                    "Dataset": dataset_name,
                    "threshold": float(threshold),
                    "positive_rate": positive_rate,
                    "predicted_positive_rate": predicted_positive_rate,
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "specificity": specificity,
                    "balanced_accuracy": balanced_accuracy,
                    "youden_j": youden_j,
                    })

    return pd.DataFrame(rows)


def select_best_threshold(sweep_df: pd.DataFrame, metric: str = THRESHOLD_SELECTION_METRIC) -> dict:
    if metric not in sweep_df.columns:
        raise ValueError(f"Metric '{metric}' neexistuje v threshold sweep DataFrame.")

    ranked = sweep_df.sort_values(
        by=[metric, "recall", "precision", "threshold"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return ranked.iloc[0].to_dict()


def plot_xgb_training_curves(evals_result: dict, save_path: Path) -> None:
    train_key = "validation_0"
    valid_key = "validation_1"
    train_metrics = evals_result.get(train_key, {})
    valid_metrics = evals_result.get(valid_key, {})

    metrics_to_plot = [m for m in ["logloss", "auc", "error"] if m in train_metrics and m in valid_metrics]
    if not metrics_to_plot:
        return

    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 4 * len(metrics_to_plot)))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_to_plot):
        train_vals = train_metrics.get(metric, [])
        valid_vals = valid_metrics.get(metric, [])
        xs = np.arange(1, len(train_vals) + 1)

        ax.plot(xs, train_vals, label=f"train_{metric}")
        ax.plot(xs, valid_vals, label=f"valid_{metric}")
        if metric == "error":
            ax.set_title("XGBoost error by boosting round (accuracy = 1 - error)")
        else:
            ax.set_title(f"XGBoost {metric} by boosting round")
        ax.set_xlabel("Boosting round")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_tabm_training_curves(history_df: pd.DataFrame, save_path: Path) -> None:
    if history_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    plot_map = [
        ("train_loss", "Train loss"),
        ("valid_logloss", "Validation logloss"),
        ("valid_accuracy", "Validation accuracy"),
        ("valid_roc_auc", "Validation ROC-AUC"),
    ]

    for ax, (col, title) in zip(axes, plot_map):
        if col in history_df.columns:
            ax.plot(history_df["epoch"], history_df[col])
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_optuna_trials(study: optuna.Study, title: str, save_path: Path) -> None:
    df = study.trials_dataframe()
    if df.empty or "value" not in df.columns:
        return

    df = df.dropna(subset=["value"]).copy()
    if df.empty:
        return

    df["best_so_far"] = df["value"].cummax()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["number"], df["value"], marker="o", linestyle="-", label="trial value")
    ax.plot(df["number"], df["best_so_far"], linestyle="--", label="best so far")
    ax.set_title(title)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Validation ROC-AUC")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def plot_resource_curves(resource_samples_df: pd.DataFrame, save_path: Path) -> None:
    if resource_samples_df.empty:
        return

    metrics = [
        ("process_cpu_percent", "Process CPU %"),
        ("process_rss_gb", "Process RSS GB"),
        ("gpu_util_percent", "GPU util %"),
        ("gpu_used_vram_gb", "GPU used VRAM GB"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, metrics):
        plotted_any = False

        for label, grp in resource_samples_df.groupby("label"):
            grp = grp.sort_values("seconds_from_start").copy()

            if metric not in grp.columns or not grp[metric].notna().any():
                continue

            grp_valid = grp.dropna(subset=[metric]).copy()
            if grp_valid.empty:
                continue

            plotted_any = True

            if len(grp_valid) == 1:
                ax.scatter(
                    grp_valid["seconds_from_start"],
                    grp_valid[metric],
                    label=label,
                    s=50,
                )
            else:
                ax.plot(
                    grp_valid["seconds_from_start"],
                    grp_valid[metric],
                    label=label,
                    marker="o",
                    markersize=4,
                )

        ax.set_title(title)
        ax.set_xlabel("Seconds from start")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

        if plotted_any:
            ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_final_metric_comparison(comparison_df: pd.DataFrame, save_path: Path) -> None:
    if comparison_df.empty:
        return

    plot_cols = ["accuracy", "f1", "roc_auc", "pr_auc"]
    plot_df = comparison_df[["Model"] + plot_cols].copy().melt(id_vars="Model", var_name="Metric", value_name="Value")

    fig, ax = plt.subplots(figsize=(10, 5))
    models = plot_df["Model"].unique().tolist()
    metrics = plot_df["Metric"].unique().tolist()
    x = np.arange(len(metrics))
    width = 0.35 if len(models) <= 2 else 0.8 / len(models)

    for idx, model_name in enumerate(models):
        vals = plot_df[plot_df["Model"] == model_name].set_index("Metric").loc[metrics, "Value"].to_numpy()
        ax.bar(x + idx * width - ((len(models) - 1) * width / 2), vals, width=width, label=model_name)

    ax.set_title("Final metric comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall_vs_threshold(sweep_df: pd.DataFrame, title: str, save_path: Path, selected_threshold: float | None = None) -> None:
    if sweep_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sweep_df["threshold"], sweep_df["precision"], label="precision")
    ax.plot(sweep_df["threshold"], sweep_df["recall"], label="recall")
    ax.plot(sweep_df["threshold"], sweep_df["f1"], label="f1", linestyle="--")

    if selected_threshold is not None:
        ax.axvline(selected_threshold, linestyle=":", label=f"selected={selected_threshold:.2f}")

    ax.set_title(title)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_threshold(sweep_df: pd.DataFrame, title: str, save_path: Path, selected_threshold: float | None = None) -> None:
    if sweep_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sweep_df["threshold"], sweep_df["accuracy"], label="accuracy")
    ax.plot(sweep_df["threshold"], sweep_df["balanced_accuracy"], label="balanced_accuracy")
    ax.plot(sweep_df["threshold"], sweep_df["predicted_positive_rate"], label="predicted_positive_rate", linestyle="--")

    if selected_threshold is not None:
        ax.axvline(selected_threshold, linestyle=":", label=f"selected={selected_threshold:.2f}")

    ax.set_title(title)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

# =========================================================
# 7. DEFINE XGBOOST RUN
# =========================================================
def build_xgb_model(params: dict) -> XGBClassifier:
    params = params.copy()
    params.update({
                    "random_state": SEED,
                    "n_jobs": -1,
                    "tree_method": "hist",
                    "device": XGB_DEVICE,
                    "objective": "binary:logistic",
                    "eval_metric": ["logloss", "auc", "error"],
                    })

    model = XGBClassifier(**params)
    return model

def train_xgboost(
                X_train: pd.DataFrame,
                y_train: pd.Series,
                X_valid: pd.DataFrame,
                y_valid: pd.Series,
                params: dict,
) -> tuple[XGBClassifier, dict]:
    model = build_xgb_model(params)
    model.fit(X_train,y_train,eval_set=[(X_train, y_train), (X_valid, y_valid)],verbose=False)
    return model, model.evals_result()


def get_default_xgb_params() -> dict:
    return {
            "n_estimators": 600,
            "max_depth": 10,
            "learning_rate": 0.001,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 2,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            }


def tune_xgboost_optuna(
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_valid: pd.DataFrame,
                        y_valid: pd.Series,
                        n_trials: int = XGB_OPTUNA_TRIALS,
) -> tuple[dict, optuna.Study]:
    def objective(trial: optuna.Trial) -> float:
        params = {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 1600),
                    "max_depth": trial.suggest_int("max_depth", 5, 15),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                }

        model, _ = train_xgboost(X_train, y_train, X_valid, y_valid, params)
        prob = model.predict_proba(X_valid)[:, 1]
        score = roc_auc_score(y_valid, prob)
        return float(score)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study

# =========================================================
# 8. DEFINE TABM RUN
# =========================================================
@dataclass
class TabMConfig:
                n_blocks: int = 3
                d_block: int = 256
                dropout: float = 0.10
                k: int = 32
                arch_type: str = "tabm"
                learning_rate: float = 0.001
                weight_decay: float = 0.0003
                batch_size: int = 256


class TabMClassifier(nn.Module):
    def __init__(self, n_num_features: int, cat_cardinalities: list[int], config: TabMConfig):
        super().__init__()
        num_embeddings = None
        if n_num_features > 0:
            num_embeddings = LinearReLUEmbeddings(n_num_features)

        self.model = TabM.make(
                                n_num_features=n_num_features,
                                cat_cardinalities=cat_cardinalities if cat_cardinalities else None,
                                num_embeddings=num_embeddings,
                                d_out=1,
                                arch_type=config.arch_type,
                                k=config.k,
                                n_blocks=config.n_blocks,
                                d_block=config.d_block,
                                dropout=config.dropout,
                                )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        if x_cat.shape[1] > 0:
            return self.model(x_num, x_cat)
        return self.model(x_num)


@torch.no_grad()
def predict_tabm_proba(model: TabMClassifier, x_num: np.ndarray, x_cat: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    probs_all: list[np.ndarray] = []

    for start in range(0, len(x_num), batch_size):
        end = start + batch_size
        xb_num = torch.tensor(x_num[start:end], dtype=torch.float32, device=TORCH_DEVICE)
        xb_cat = torch.tensor(x_cat[start:end], dtype=torch.long, device=TORCH_DEVICE)

        logits = model(xb_num, xb_cat).squeeze(-1)
        probs = torch.sigmoid(logits).mean(dim=1)
        probs_all.append(probs.detach().cpu().numpy())

    return np.concatenate(probs_all)


def get_default_tabm_config() -> TabMConfig:
    return TabMConfig()


def make_tabm_config_from_trial(trial: optuna.Trial) -> TabMConfig:
    wd_choice = trial.suggest_categorical("weight_decay_mode", ["zero", "nonzero"])
    weight_decay = 0.0 if wd_choice == "zero" else trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)

    return TabMConfig(
                    n_blocks=trial.suggest_int("n_blocks", 2, 4),
                    d_block=trial.suggest_int("d_block", 128, 1024, step=64),
                    dropout=trial.suggest_float("dropout", 0.0, 0.30),
                    k=32,
                    arch_type="tabm",
                    learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
                    weight_decay=weight_decay,
                    batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
                    )


def train_tabm(
    x_num_train: np.ndarray,
    x_cat_train: np.ndarray,
    y_train: np.ndarray,
    x_num_valid: np.ndarray,
    x_cat_valid: np.ndarray,
    y_valid: np.ndarray,
    cat_cardinalities: list[int],
    config: TabMConfig,
    max_epochs: int,
    patience: int,
    verbose: bool = True,
) -> tuple[TabMClassifier, pd.DataFrame]:
    train_ds = TensorDataset(
        torch.tensor(x_num_train, dtype=torch.float32),
        torch.tensor(x_cat_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds,batch_size=config.batch_size,shuffle=True,num_workers=0,pin_memory=(TORCH_DEVICE.type == "cuda"))
    model = TabMClassifier(n_num_features=x_num_train.shape[1],cat_cardinalities=cat_cardinalities,config=config).to(TORCH_DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)

    best_state = None
    best_score = -np.inf
    bad_epochs = 0
    history_rows = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses = []

        for xb_num, xb_cat, yb in train_loader:
            xb_num = xb_num.to(TORCH_DEVICE)
            xb_cat = xb_cat.to(TORCH_DEVICE)
            yb = yb.to(TORCH_DEVICE)

            optimizer.zero_grad()
            logits = model(xb_num, xb_cat).squeeze(-1)
            y_rep = yb.unsqueeze(1).expand_as(logits)
            loss = F.binary_cross_entropy_with_logits(logits, y_rep)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        valid_probs = predict_tabm_proba(model, x_num_valid, x_cat_valid, batch_size=max(1024, config.batch_size))
        valid_metrics = evaluate_binary_model(y_valid, valid_probs, threshold=DEFAULT_FIXED_THRESHOLD)

        row = {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "valid_accuracy": valid_metrics["accuracy"],
                "valid_f1": valid_metrics["f1"],
                "valid_precision": valid_metrics["precision"],
                "valid_recall": valid_metrics["recall"],
                "valid_roc_auc": valid_metrics["roc_auc"],
                "valid_pr_auc": valid_metrics["pr_auc"],
                "valid_logloss": valid_metrics["logloss"],
                }
        history_rows.append(row)

        if verbose:
            print(
                    f"[TabM] epoch={epoch:03d} "
                    f"train_loss={row['train_loss']:.6f} "
                    f"valid_acc={row['valid_accuracy']:.6f} "
                    f"valid_roc_auc={row['valid_roc_auc']:.6f} "
                    f"valid_pr_auc={row['valid_pr_auc']:.6f}"
                    )

        score = row["valid_roc_auc"]
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"[TabM] Early stopping po {epoch} epochach.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(history_rows)

def tune_tabm_optuna(
    x_num_train: np.ndarray,
    x_cat_train: np.ndarray,
    y_train: np.ndarray,
    x_num_valid: np.ndarray,
    x_cat_valid: np.ndarray,
    y_valid: np.ndarray,
    cat_cardinalities: list[int],
    n_trials: int = TABM_OPTUNA_TRIALS,
) -> tuple[TabMConfig, optuna.Study]:
    def objective(trial: optuna.Trial) -> float:
        config = make_tabm_config_from_trial(trial)
        model, history_df = train_tabm(
                                        x_num_train=x_num_train,
                                        x_cat_train=x_cat_train,
                                        y_train=y_train,
                                        x_num_valid=x_num_valid,
                                        x_cat_valid=x_cat_valid,
                                        y_valid=y_valid,
                                        cat_cardinalities=cat_cardinalities,
                                        config=config,
                                        max_epochs=TABM_TUNE_EPOCHS,
                                        patience=TABM_TUNE_PATIENCE,
                                        verbose=True,
                                        )
        if history_df.empty:
            return 0.0
        return float(history_df["valid_roc_auc"].max())

    study = optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params.copy()
    best_config = TabMConfig(
                                n_blocks=int(best_params["n_blocks"]),
                                d_block=int(best_params["d_block"]),
                                dropout=float(best_params["dropout"]),
                                k=32,
                                arch_type="tabm",
                                learning_rate=float(best_params["learning_rate"]),
                                weight_decay=0.0 if best_params["weight_decay_mode"] == "zero" else float(best_params["weight_decay"]),
                                batch_size=int(best_params["batch_size"]),
                            )
    return best_config, study


# =========================================================
# 9. MAIN FUNCTION
# =========================================================
def build_prediction_output(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    model_name: str,
) -> pd.DataFrame:
    out = meta_df.copy()
    out["Actual_Attrition"] = y_true
    out["Predicted_Prob"] = np.round(y_prob * 100, 2)
    out["Predicted_Prob"] = out["Predicted_Prob"].astype(str) + "%"
    out["Applied_Threshold"] = float(threshold)
    out["Predicted_Label"] = np.where(y_prob >= threshold, "Yes", "No")
    out["Model_Name"] = model_name
    return out


def main() -> None:
    print("START: Tuned XGBoost vs Tuned TabM benchmark")

    df = load_hr_dataframe()
    X_all, y_all, meta_all = prepare_base_dataframe(df)

    X_train_full, X_test, y_train_full, y_test, meta_train_full, meta_test = train_test_split(
                                                                                                X_all,
                                                                                                y_all,
                                                                                                meta_all,
                                                                                                test_size=TEST_SIZE,
                                                                                                random_state=SEED,
                                                                                                stratify=y_all,
                                                                                            )

    X_train, X_valid, y_train, y_valid, _, _ = train_test_split(
                                                                X_train_full,
                                                                y_train_full,
                                                                meta_train_full,
                                                                test_size=VALID_SIZE,
                                                                random_state=SEED,
                                                                stratify=y_train_full,
                                                                )

    print(
        f"Split -> train={len(X_train)}, valid={len(X_valid)}, test={len(X_test)} | "
        f"target mean train={y_train.mean():.4f}, valid={y_valid.mean():.4f}, test={y_test.mean():.4f}"
        )

    preprocessor = TabularPreprocessor.fit(X_train)
    joblib.dump(preprocessor, MODEL_DIR / "comparison_preprocessor.pkl")

    resource_rows = []
    resource_samples = []

    # -----------------------------------------------------
    # a. Shared transformed data
    # -----------------------------------------------------
    X_train_xgb = preprocessor.transform_for_xgboost(X_train)
    X_valid_xgb = preprocessor.transform_for_xgboost(X_valid)
    X_test_xgb = preprocessor.transform_for_xgboost(X_test)

    x_num_train, x_cat_train = preprocessor.transform_for_tabm(X_train)
    x_num_valid, x_cat_valid = preprocessor.transform_for_tabm(X_valid)
    x_num_test, x_cat_test = preprocessor.transform_for_tabm(X_test)

    # -----------------------------------------------------
    # b. XGBoost optuna tuning 
    # -----------------------------------------------------
    if RUN_XGB_OPTUNA:
        print("Optuna tuning: XGBoost")
        best_xgb_params, xgb_study = tune_xgboost_optuna(
                                                            X_train=X_train_xgb,
                                                            y_train=y_train,
                                                            X_valid=X_valid_xgb,
                                                            y_valid=y_valid,
                                                            n_trials=XGB_OPTUNA_TRIALS,
                                                        )
    else:
        best_xgb_params = get_default_xgb_params()
        xgb_study = None

    save_json(best_xgb_params, OPTUNA_DIR / "xgb_best_params.json")
    if xgb_study is not None:
        xgb_trials_df = xgb_study.trials_dataframe()
        xgb_trials_df.to_csv(OPTUNA_DIR / "xgb_optuna_trials.csv", index=False, encoding="utf-8-sig")
        plot_optuna_trials(xgb_study, "XGBoost Optuna trials", PLOTS_DIR / "xgb_optuna_trials.png")

    print("XGBoost best params:")
    print(best_xgb_params)

    # -----------------------------------------------------
    # c. Final XGBoost training + monitoring
    # -----------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    xgb_monitor = ResourceMonitor(label="XGBoost")
    xgb_monitor.start()

    if RUN_XGB_OPTUNA:
        print("Optuna tuning: XGBoost")
        best_xgb_params, xgb_study = tune_xgboost_optuna(
                                                        X_train=X_train_xgb,
                                                        y_train=y_train,
                                                        X_valid=X_valid_xgb,
                                                        y_valid=y_valid,
                                                        n_trials=XGB_OPTUNA_TRIALS,
                                                        )
    else:
        best_xgb_params = get_default_xgb_params()
        xgb_study = None

    save_json(best_xgb_params, OPTUNA_DIR / "xgb_best_params.json")
    if xgb_study is not None:
        xgb_trials_df = xgb_study.trials_dataframe()
        xgb_trials_df.to_csv(
            OPTUNA_DIR / "xgb_optuna_trials.csv",
            index=False,
            encoding="utf-8-sig",
        )
        plot_optuna_trials(
            xgb_study,
            "XGBoost Optuna trials",
            PLOTS_DIR / "xgb_optuna_trials.png",
        )

    print("XGBoost best params:")
    print(best_xgb_params)

    xgb_model, xgb_evals_result = train_xgboost(
                                                X_train=X_train_xgb,
                                                y_train=y_train,
                                                X_valid=X_valid_xgb,
                                                y_valid=y_valid,
                                                params=best_xgb_params,
                                                )

    xgb_valid_prob = xgb_model.predict_proba(X_valid_xgb)[:, 1]
    xgb_test_prob = xgb_model.predict_proba(X_test_xgb)[:, 1]

    xgb_monitor.stop()

    xgb_fixed_metrics = evaluate_binary_model(
        y_test.to_numpy(),
        xgb_test_prob,
        threshold=DEFAULT_FIXED_THRESHOLD,
    )
    xgb_valid_sweep_df = build_threshold_sweep(
        y_valid.to_numpy(),
        xgb_valid_prob,
        "XGBoost",
        "validation",
    )
    xgb_test_sweep_df = build_threshold_sweep(
        y_test.to_numpy(),
        xgb_test_prob,
        "XGBoost",
        "test",
    )
    xgb_best_threshold_row = select_best_threshold(
        xgb_valid_sweep_df,
        metric=THRESHOLD_SELECTION_METRIC,
    )
    xgb_selected_threshold = float(xgb_best_threshold_row["threshold"])
    xgb_metrics = evaluate_binary_model(
        y_test.to_numpy(),
        xgb_test_prob,
        threshold=xgb_selected_threshold,
    )

    print_metric_block("XGBoost (threshold = 0.50)", xgb_fixed_metrics)
    print_metric_block(
        f"XGBoost (selected threshold = {xgb_selected_threshold:.2f})",
        xgb_metrics,
    )

    xgb_resource_summary = xgb_monitor.summary()
    resource_rows.append(xgb_resource_summary)
    resource_samples.append(xgb_monitor.samples_df())

    print("=== XGBoost RESOURCES ===")
    print(pd.DataFrame([xgb_resource_summary]).to_string(index=False))

    xgb_importance = pd.DataFrame(
        {
            "Feature": X_train_xgb.columns,
            "Importance": xgb_model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    xgb_predictions_df = build_prediction_output(meta_df=meta_test,y_true=y_test.to_numpy(),y_prob=xgb_test_prob,threshold=xgb_selected_threshold,model_name="XGBoost")

    joblib.dump(xgb_model, MODEL_DIR / "xgb_comparison_model.pkl")
    plot_xgb_training_curves(xgb_evals_result,PLOTS_DIR / "xgb_training_curves.png")
    xgb_valid_sweep_df.to_csv(THRESHOLD_DIR / "xgb_validation_threshold_sweep.csv", index=False, encoding="utf-8-sig")
    xgb_test_sweep_df.to_csv(THRESHOLD_DIR / "xgb_test_threshold_sweep.csv",index=False,encoding="utf-8-sig")
    plot_precision_recall_vs_threshold(xgb_valid_sweep_df,"XGBoost validation: precision / recall / F1 vs threshold",PLOTS_DIR / "xgb_precision_recall_vs_threshold.png",selected_threshold=xgb_selected_threshold)
    plot_accuracy_vs_threshold(xgb_valid_sweep_df,"XGBoost validation: accuracy vs threshold",PLOTS_DIR / "xgb_accuracy_vs_threshold.png",selected_threshold=xgb_selected_threshold)

    # -----------------------------------------------------
    # d. TabM tuning
    # -----------------------------------------------------
    if RUN_TABM_OPTUNA:
        print("Optuna tuning: TabM")
        best_tabm_config, tabm_study = tune_tabm_optuna(
                                                            x_num_train=x_num_train,
                                                            x_cat_train=x_cat_train,
                                                            y_train=y_train.to_numpy(),
                                                            x_num_valid=x_num_valid,
                                                            x_cat_valid=x_cat_valid,
                                                            y_valid=y_valid.to_numpy(),
                                                            cat_cardinalities=preprocessor.categorical_cardinalities,
                                                            n_trials=TABM_OPTUNA_TRIALS,
                                                        )
    else:
        best_tabm_config = get_default_tabm_config()
        tabm_study = None

    save_json(asdict(best_tabm_config), OPTUNA_DIR / "tabm_best_params.json")
    if tabm_study is not None:
        tabm_trials_df = tabm_study.trials_dataframe()
        tabm_trials_df.to_csv(OPTUNA_DIR / "tabm_optuna_trials.csv", index=False, encoding="utf-8-sig")
        plot_optuna_trials(tabm_study, "TabM Optuna trials", PLOTS_DIR / "tabm_optuna_trials.png")

    print("TabM best params:")
    print(asdict(best_tabm_config))

    # -----------------------------------------------------
    # e. Final TabM training + monitoring
    # -----------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tabm_monitor = ResourceMonitor(label="TabM")
    tabm_monitor.start()

    if RUN_TABM_OPTUNA:
        print("=== Optuna tuning: TabM ===")
        best_tabm_config, tabm_study = tune_tabm_optuna(
                                                        x_num_train=x_num_train,
                                                        x_cat_train=x_cat_train,
                                                        y_train=y_train.to_numpy(),
                                                        x_num_valid=x_num_valid,
                                                        x_cat_valid=x_cat_valid,
                                                        y_valid=y_valid.to_numpy(),
                                                        cat_cardinalities=preprocessor.categorical_cardinalities,
                                                        n_trials=TABM_OPTUNA_TRIALS
                                                        )
    else:
        best_tabm_config = get_default_tabm_config()
        tabm_study = None

    save_json(asdict(best_tabm_config), OPTUNA_DIR / "tabm_best_params.json")
    if tabm_study is not None:
        tabm_trials_df = tabm_study.trials_dataframe()
        tabm_trials_df.to_csv(
            OPTUNA_DIR / "tabm_optuna_trials.csv",
            index=False,
            encoding="utf-8-sig",
        )
        plot_optuna_trials(
            tabm_study,
            "TabM Optuna trials",
            PLOTS_DIR / "tabm_optuna_trials.png",
        )

    print("TabM best params:")
    print(asdict(best_tabm_config))

    tabm_model, tabm_history_df = train_tabm(
                                            x_num_train=x_num_train,
                                            x_cat_train=x_cat_train,
                                            y_train=y_train.to_numpy(),
                                            x_num_valid=x_num_valid,
                                            x_cat_valid=x_cat_valid,
                                            y_valid=y_valid.to_numpy(),
                                            cat_cardinalities=preprocessor.categorical_cardinalities,
                                            config=best_tabm_config,
                                            max_epochs=TABM_FINAL_EPOCHS,
                                            patience=TABM_FINAL_PATIENCE,
                                            verbose=True,
                                            )

    tabm_valid_prob = predict_tabm_proba(tabm_model,x_num_valid,x_cat_valid,batch_size=max(1024, best_tabm_config.batch_size))
    tabm_test_prob = predict_tabm_proba(tabm_model,x_num_test,x_cat_test,batch_size=max(1024, best_tabm_config.batch_size))

    tabm_monitor.stop()

    tabm_fixed_metrics = evaluate_binary_model(y_test.to_numpy(), tabm_test_prob, threshold=DEFAULT_FIXED_THRESHOLD)
    tabm_valid_sweep_df = build_threshold_sweep(y_valid.to_numpy(),tabm_valid_prob,"TabM","validation")
    tabm_test_sweep_df = build_threshold_sweep(y_test.to_numpy(),tabm_test_prob,"TabM","test")
    tabm_best_threshold_row = select_best_threshold(tabm_valid_sweep_df,metric=THRESHOLD_SELECTION_METRIC)
    tabm_selected_threshold = float(tabm_best_threshold_row["threshold"])
    tabm_metrics = evaluate_binary_model(y_test.to_numpy(),tabm_test_prob,threshold=tabm_selected_threshold)

    print_metric_block("TabM (threshold = 0.50)", tabm_fixed_metrics)
    print_metric_block(f"TabM (selected threshold = {tabm_selected_threshold:.2f})",tabm_metrics)

    tabm_resource_summary = tabm_monitor.summary()
    resource_rows.append(tabm_resource_summary)
    resource_samples.append(tabm_monitor.samples_df())

    print("TabM RESOURCES")
    print(pd.DataFrame([tabm_resource_summary]).to_string(index=False))

    torch.save(tabm_model.state_dict(), MODEL_DIR / "tabm_comparison_model.pt")
    joblib.dump({"n_num_features": x_num_train.shape[1],"cat_cardinalities": preprocessor.categorical_cardinalities,"config": asdict(best_tabm_config),},MODEL_DIR / "tabm_comparison_metadata.pkl")

    tabm_predictions_df = build_prediction_output(
                                                    meta_df=meta_test,
                                                    y_true=y_test.to_numpy(),
                                                    y_prob=tabm_test_prob,
                                                    threshold=tabm_selected_threshold,
                                                    model_name="TabM"
                                                    )

    tabm_history_df.to_csv( OUTPUT_DIR / "tabm_training_history.csv", index=False, encoding="utf-8-sig")
    plot_tabm_training_curves(
        tabm_history_df,
        PLOTS_DIR / "tabm_training_curves.png",
    )
    tabm_valid_sweep_df.to_csv(THRESHOLD_DIR / "tabm_validation_threshold_sweep.csv",index=False,encoding="utf-8-sig")
    tabm_test_sweep_df.to_csv(THRESHOLD_DIR / "tabm_test_threshold_sweep.csv",index=False,encoding="utf-8-sig")
    plot_precision_recall_vs_threshold(
        tabm_valid_sweep_df,
        "TabM validation: precision / recall / F1 vs threshold",
        PLOTS_DIR / "tabm_precision_recall_vs_threshold.png",
        selected_threshold=tabm_selected_threshold,
    )
    plot_accuracy_vs_threshold(tabm_valid_sweep_df,"TabM validation: accuracy vs threshold",PLOTS_DIR / "tabm_accuracy_vs_threshold.png",selected_threshold=tabm_selected_threshold)
    
    # -----------------------------------------------------
    # f. Final comparison / export
    # -----------------------------------------------------
    comparison_df = pd.DataFrame(
        [
            {
                "Model": "XGBoost",
                "selected_threshold": xgb_selected_threshold,
                "threshold_metric": THRESHOLD_SELECTION_METRIC,
                "fixed_accuracy_0_50": xgb_fixed_metrics["accuracy"],
                "fixed_f1_0_50": xgb_fixed_metrics["f1"],
                **xgb_metrics,
            },
            {
                "Model": "TabM",
                "selected_threshold": tabm_selected_threshold,
                "threshold_metric": THRESHOLD_SELECTION_METRIC,
                "fixed_accuracy_0_50": tabm_fixed_metrics["accuracy"],
                "fixed_f1_0_50": tabm_fixed_metrics["f1"],
                **tabm_metrics,
            },
        ]
    ).sort_values(by="roc_auc", ascending=False)

    selected_thresholds_df = pd.DataFrame([xgb_best_threshold_row, tabm_best_threshold_row])
    resource_summary_df = pd.DataFrame(resource_rows)
    resource_samples_df = pd.concat(resource_samples, ignore_index=True) if resource_samples else pd.DataFrame()

    print("FINAL COMPARISON")
    print(comparison_df.to_string(index=False))

    comparison_df.to_csv(COMPARISON_CSV_PATH, index=False, encoding="utf-8-sig")
    selected_thresholds_df.to_csv(THRESHOLD_DIR / "selected_thresholds_summary.csv", index=False, encoding="utf-8-sig")
    resource_summary_df.to_csv(RESOURCE_SUMMARY_CSV_PATH, index=False, encoding="utf-8-sig")
    resource_samples_df.to_csv(RESOURCE_SAMPLES_CSV_PATH, index=False, encoding="utf-8-sig")
    xgb_predictions_df.to_csv(XGB_PREDICTIONS_CSV_PATH, index=False, encoding="utf-8-sig")
    tabm_predictions_df.to_csv(TABM_PREDICTIONS_CSV_PATH, index=False, encoding="utf-8-sig")
    xgb_importance.to_csv(XGB_IMPORTANCE_CSV_PATH, index=False, encoding="utf-8-sig")

    plot_resource_curves(resource_samples_df, PLOTS_DIR / "resource_curves.png")
    plot_final_metric_comparison(comparison_df, PLOTS_DIR / "final_metric_comparison.png")

    if USE_SQL_EXPORT:
        save_to_sql(comparison_df, "Model_Comparison_TabM_vs_XGBoost")
        save_to_sql(xgb_predictions_df, "Predictions_Attrition_XGBoost")
        save_to_sql(tabm_predictions_df, "Predictions_Attrition_TabM")
        save_to_sql(xgb_importance, "Model_Feature_Importance_XGBoost")
        save_to_sql(resource_summary_df, "Benchmark_Resource_Summary")
        save_to_sql(resource_samples_df, "Benchmark_Resource_Samples")
        save_to_sql(tabm_history_df, "Benchmark_TabM_Training_History")
        save_to_sql(selected_thresholds_df, "Benchmark_Selected_Thresholds")
        save_to_sql(xgb_valid_sweep_df, "Benchmark_XGBoost_Validation_Threshold_Sweep")
        save_to_sql(tabm_valid_sweep_df, "Benchmark_TabM_Validation_Threshold_Sweep")

    print(f"Ulozene: {COMPARISON_CSV_PATH}")
    print(f"Ulozene: {RESOURCE_SUMMARY_CSV_PATH}")
    print(f"Ulozene: {RESOURCE_SAMPLES_CSV_PATH}")
    print(f"Ulozene: {XGB_PREDICTIONS_CSV_PATH}")
    print(f"Ulozene: {TABM_PREDICTIONS_CSV_PATH}")
    print(f"Ulozene: {XGB_IMPORTANCE_CSV_PATH}")
    print(f"Ulozene: {OUTPUT_DIR / 'tabm_training_history.csv'}")
    print(f"Threshold CSV ulozene do: {THRESHOLD_DIR}")
    print(f"Grafy ulozene do: {PLOTS_DIR}")
    print(f"Optuna exporty ulozene do: {OPTUNA_DIR}")
    print("HOTOVO")


if __name__ == "__main__":
    main()