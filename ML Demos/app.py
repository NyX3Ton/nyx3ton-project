from __future__ import annotations

import ast, json, os, re, time, optuna, torch
from pathlib import Path
from typing import Any
import gradio as gr
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

try:
    from generate_inputs import generate_movie_genre, generate_movie_recommendation
except ImportError:
    from generate_inputs import generate_movie_genre, generate_movie_recommendation
# ============================================================
# Global config
# ============================================================

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

optuna.logging.set_verbosity(optuna.logging.WARNING)

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")

BASE_DIR = Path(__file__).resolve().parent
INPUTS_DIR = BASE_DIR / "Inputs"
INPUTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = BASE_DIR / "Cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OPTUNA_DB = CACHE_DIR / "optuna_studies.sqlite3"
OPTUNA_STORAGE = f"sqlite:///{OPTUNA_DB.as_posix()}"

GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Fantasy", "Romance", "Documentary"]
MOODS = ["Relax", "Funny", "Emotional", "Adrenaline", "Mind-bending"]

IMAGE_PRODUCT_CATEGORIES = [
    "Tech gadget",
    "Gaming accessory",
    "Fitness product",
    "Travel gear",
    "Food product",
    "Fashion item",
    "Home appliance",
    "Book",
    "Musical instrument",
    "Sports equipment",
    "Office equipment",
    "Beauty product",
]

OPTUNA_TRIALS_XGB = int(os.getenv("OPTUNA_TRIALS_XGB", "12"))
OPTUNA_TRIALS_MLP = int(os.getenv("OPTUNA_TRIALS_MLP", "8"))

MAX_MOVIE_ROWS = int(os.getenv("MAX_MOVIE_ROWS", "15000"))
MAX_GENRE_ROWS = int(os.getenv("MAX_GENRE_ROWS", "20000"))

CLIP_MODEL_ID = os.getenv("CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = None
CLIP_PROCESSOR = None

CUSTOM_CSS = """
.gradio-container {
    max-width: 1550px !important;
}

.hero-card {
    padding: 24px;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(25, 80, 170, 0.16), rgba(120, 60, 180, 0.14));
    border: 1px solid rgba(120, 120, 120, 0.20);
    margin-bottom: 16px;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(180px, 1fr));
    gap: 12px;
}

.metric-card {
    padding: 15px 18px;
    border-radius: 16px;
    background: rgba(120, 120, 120, 0.09);
    border: 1px solid rgba(120, 120, 120, 0.16);
    min-height: 100px;
}

.metric-card h3 {
    margin-top: 0;
    margin-bottom: 6px;
    font-size: 1.45rem;
}

.small-note {
    opacity: 0.82;
    font-size: 0.92rem;
}

.prediction-box textarea {
    font-size: 1.04rem !important;
    line-height: 1.45 !important;
}

.dataframe-table {
    font-size: 0.92rem;
}
"""


# ============================================================
# Small helpers
# ============================================================

def safe_name(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "study"


def to_dense_float32(matrix: object) -> np.ndarray:
    toarray = getattr(matrix, "toarray", None)
    if callable(toarray):
        return np.asarray(toarray(), dtype=np.float32)

    to_numpy = getattr(matrix, "to_numpy", None)
    if callable(to_numpy):
        return np.asarray(to_numpy(), dtype=np.float32)

    return np.asarray(matrix, dtype=np.float32)


def find_existing_input(candidates: list[str]) -> Path | None:
    for name in candidates:
        path = INPUTS_DIR / name
        if path.exists():
            return path
    return None


def read_table_file(path: Path, **kwargs) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)

    if suffix in [".xlsx", ".xls"]:
        kwargs.pop("low_memory", None)
        return pd.read_excel(path, **kwargs)

    if suffix == ".txt":
        return pd.read_csv(path, **kwargs)

    raise ValueError(f"Unsupported input file type: {path}")


def read_input(stems: list[str]) -> tuple[pd.DataFrame | None, str]:
    for stem in stems:
        for ext in [".csv", ".xlsx", ".xls"]:
            path = INPUTS_DIR / f"{stem}{ext}"
            if path.exists():
                return read_table_file(path), path.name

    return None, "generated fallback"


def split_data(X, y):
    try:
        return train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=SEED,
        )
    except ValueError:
        return train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=SEED,
        )


def softmax_numpy(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values)


# ============================================================
# Persistent Optuna helpers
# ============================================================

def get_or_create_study(
    study_name: str,
    direction: str = "maximize",
) -> optuna.Study:
    return optuna.create_study(
        study_name=study_name,
        storage=OPTUNA_STORAGE,
        direction=direction,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )


def completed_trial_count(study: optuna.Study) -> int:
    return sum(
        1
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    )


def ensure_study_trials(
    study: optuna.Study,
    objective,
    required_trials: int,
    timeout_sec: int | None = None,
) -> None:
    if required_trials <= 0:
        print(f"Optuna study '{study.study_name}' skipped because required_trials <= 0.")
        return

    existing = completed_trial_count(study)

    if existing >= required_trials:
        print(
            f"Optuna study '{study.study_name}' already has "
            f"{existing}/{required_trials} completed trials. Skipping tuning."
        )
        return

    missing = required_trials - existing

    print(
        f"Optuna study '{study.study_name}' has "
        f"{existing}/{required_trials} completed trials. "
        f"Running {missing} new trials."
    )

    study.optimize(
        objective,
        n_trials=missing,
        timeout=timeout_sec,
        show_progress_bar=False,
    )


# ============================================================
# Torch models
# ============================================================

class BinaryMLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_1: int = 64,
        hidden_2: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MultiMLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_1: int = 128,
        hidden_2: int = 64,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Movie and genre data loaders
# ============================================================

def parse_movie_genres(raw_value: object) -> str:
    if pd.isna(raw_value):
        return "Unknown"

    text = str(raw_value).strip()

    if not text or text == "[]":
        return "Unknown"

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list) and parsed:
            for item in parsed:
                if isinstance(item, dict) and item.get("name"):
                    return str(item["name"])
    except Exception:
        pass

    return "Unknown"


def canonical_movie_genre(raw_genre: object) -> str:
    text = str(raw_genre).strip()

    mapping = {
        "Action": "Action",
        "Adventure": "Action",
        "War": "Action",
        "Crime": "Action",
        "Thriller": "Action",
        "Comedy": "Comedy",
        "Drama": "Drama",
        "Science Fiction": "Sci-Fi",
        "Sci-Fi": "Sci-Fi",
        "Fantasy": "Fantasy",
        "Animation": "Fantasy",
        "Family": "Fantasy",
        "Romance": "Romance",
        "Documentary": "Documentary",
    }

    return mapping.get(text, "Drama")


def load_movie_data() -> tuple[pd.DataFrame, np.ndarray, str]:
    metadata_path = find_existing_input(["movies_metadata.csv", "movies_metadata.xlsx"])
    ratings_path = find_existing_input(["ratings_small.csv", "ratings_small.xlsx", "ratings.csv", "ratings.xlsx"])
    links_path = find_existing_input(["links_small.csv", "links_small.xlsx", "links.csv", "links.xlsx"])

    if metadata_path is not None and ratings_path is not None:
        movies = read_table_file(metadata_path, low_memory=False)
        ratings = read_table_file(ratings_path)

        movies = movies.copy()
        ratings = ratings.copy()

        movies["id_numeric"] = pd.to_numeric(movies.get("id"), errors="coerce")
        movies = movies.dropna(subset=["id_numeric"])
        movies["id_numeric"] = movies["id_numeric"].astype(int)

        ratings["movieId"] = pd.to_numeric(ratings.get("movieId"), errors="coerce")
        ratings["rating"] = pd.to_numeric(ratings.get("rating"), errors="coerce")
        ratings = ratings.dropna(subset=["movieId", "rating"])
        ratings["movieId"] = ratings["movieId"].astype(int)

        if links_path is not None:
            links = read_table_file(links_path)
            links = links.copy()
            links["movieId"] = pd.to_numeric(links.get("movieId"), errors="coerce")
            links["tmdbId"] = pd.to_numeric(links.get("tmdbId"), errors="coerce")
            links = links.dropna(subset=["movieId", "tmdbId"])
            links["movieId"] = links["movieId"].astype(int)
            links["tmdbId"] = links["tmdbId"].astype(int)

            merged = ratings.merge(links[["movieId", "tmdbId"]], on="movieId", how="inner")
            merged = merged.merge(movies, left_on="tmdbId", right_on="id_numeric", how="inner")
            source = f"{metadata_path.name} + {ratings_path.name} + {links_path.name}"
        else:
            merged = ratings.merge(movies, left_on="movieId", right_on="id_numeric", how="inner")
            source = f"{metadata_path.name} + {ratings_path.name}"

        if len(merged) >= 100:
            merged = merged.sample(n=min(len(merged), MAX_MOVIE_ROWS), random_state=SEED).copy()
            merged["raw_main_genre"] = merged.get("genres", "Unknown").apply(parse_movie_genres)
            merged["canonical_genre"] = merged["raw_main_genre"].apply(canonical_movie_genre)
            merged["movie_genre_id"] = merged["canonical_genre"].map({value: idx for idx, value in enumerate(GENRES)}).fillna(0)

            if "release_date" in merged.columns:
                release_year = pd.to_datetime(merged["release_date"], errors="coerce").dt.year
                current_year = pd.Timestamp.now().year
                merged["movie_age_years"] = current_year - release_year
            else:
                merged["movie_age_years"] = 10

            if "vote_average" in merged.columns:
                merged["movie_rating"] = pd.to_numeric(merged["vote_average"], errors="coerce")
            else:
                merged["movie_rating"] = 7.0

            if "runtime" in merged.columns:
                merged["movie_length_min"] = pd.to_numeric(merged["runtime"], errors="coerce")
            else:
                merged["movie_length_min"] = 110

            user_genre_mean = merged.groupby(["userId", "canonical_genre"])["rating"].mean().reset_index()
            pivot = user_genre_mean.pivot_table(index="userId", columns="canonical_genre", values="rating", fill_value=0)

            for col in GENRES:
                if col not in pivot.columns:
                    pivot[col] = 0.0

            pivot = pivot.reset_index()
            merged = merged.merge(pivot[["userId", *GENRES]], on="userId", how="left")

            merged["user_action"] = merged["Action"] / 5.0
            merged["user_comedy"] = merged["Comedy"] / 5.0
            merged["user_drama"] = merged["Drama"] / 5.0
            merged["user_scifi"] = merged["Sci-Fi"] / 5.0
            merged["user_romance"] = merged["Romance"] / 5.0
            merged["user_documentary"] = merged["Documentary"] / 5.0
            merged["mood_id"] = 0

            features = [
                "user_action",
                "user_comedy",
                "user_drama",
                "user_scifi",
                "user_romance",
                "user_documentary",
                "movie_genre_id",
                "movie_rating",
                "movie_age_years",
                "movie_length_min",
                "mood_id",
            ]

            X = merged[features].apply(pd.to_numeric, errors="coerce").fillna(0)
            y = (pd.to_numeric(merged["rating"], errors="coerce").fillna(0) >= 4.0).astype(int).to_numpy()
            return X, y, source

        print("Kaggle movie files were found, but merge produced too few rows. Falling back to local full/sample data.")

    df, source = read_input(["movie_recommendation_full", "movie_recommendation_sample"])

    if df is None:
        df = generate_movie_recommendation(rows=5000)
        source = "generated fallback"

    df = df.copy()
    df["movie_genre_id"] = df["movie_genre"].map({value: idx for idx, value in enumerate(GENRES)}).fillna(0)
    df["mood_id"] = df["mood"].map({value: idx for idx, value in enumerate(MOODS)}).fillna(0)

    features = [
        "user_action",
        "user_comedy",
        "user_drama",
        "user_scifi",
        "user_romance",
        "user_documentary",
        "movie_genre_id",
        "movie_rating",
        "movie_age_years",
        "movie_length_min",
        "mood_id",
    ]

    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = pd.to_numeric(df["recommended"], errors="coerce").fillna(0).astype(int).to_numpy()
    return X, y, source


def load_genre_dataframe() -> tuple[pd.DataFrame, str]:
    genre_path = find_existing_input([
        "movie_genre_full.csv",
        "movie_genre_full.xlsx",
        "movie_genre_sample.csv",
        "movie_genre_sample.xlsx",
        "train_data.txt",
        "train_data_solution.txt",
    ])

    if genre_path is None:
        df = generate_movie_genre(rows_per_genre=600)
        df = df.copy()
        df["title"] = [f"Demo Movie {i + 1}" for i in range(len(df))]
        return df[["title", "description", "genre"]], "generated fallback"

    if genre_path.suffix.lower() == ".txt":
        df = pd.read_csv(genre_path, sep=r"\s*:::\s*", engine="python", names=["id", "title", "genre", "description"])
        df = df[["title", "description", "genre"]].dropna().copy()
        if len(df) > MAX_GENRE_ROWS:
            df = df.sample(n=MAX_GENRE_ROWS, random_state=SEED)
        return df, genre_path.name

    df = read_table_file(genre_path)

    possible_title_cols = ["title", "Title", "movie_title", "Movie Title", "Movie name", "Movie Name", "name", "Name"]
    possible_description_cols = ["description", "Description", "plot", "Plot", "overview", "Overview", "summary", "Summary"]
    possible_genre_cols = ["genre", "Genre", "genres", "Genres"]

    title_col = next((col for col in possible_title_cols if col in df.columns), None)
    description_col = next((col for col in possible_description_cols if col in df.columns), None)
    genre_col = next((col for col in possible_genre_cols if col in df.columns), None)

    if description_col is None or genre_col is None:
        raise ValueError(f"Genre file {genre_path.name} must contain description and genre columns.")

    result = df[[description_col, genre_col]].dropna().copy()
    result.columns = ["description", "genre"]

    if title_col is not None:
        result["title"] = df.loc[result.index, title_col].astype(str)
    else:
        result["title"] = [f"Movie Example {i + 1}" for i in range(len(result))]

    result = result[["title", "description", "genre"]]

    if len(result) > MAX_GENRE_ROWS:
        result = result.sample(n=MAX_GENRE_ROWS, random_state=SEED)

    return result, genre_path.name


# ============================================================
# Training functions with Optuna
# ============================================================

def train_binary_mlp(
    X: np.ndarray,
    y: np.ndarray,
    optuna_trials: int = OPTUNA_TRIALS_MLP,
    timeout_sec: int | None = None,
    study_name: str = "binary_mlp",
) -> tuple[nn.Module, float, dict]:
    start_total = time.perf_counter()
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    X_train, X_val, y_train, y_val = split_data(X, y)

    def run_training(
        model: nn.Module,
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        lr: float,
        weight_decay: float,
        batch_size: int,
        epochs: int,
        trial: optuna.Trial | None = None,
        X_eval: np.ndarray | None = None,
        y_eval: np.ndarray | None = None,
    ) -> nn.Module:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        loader = DataLoader(
            TensorDataset(torch.tensor(X_fit, dtype=torch.float32), torch.tensor(y_fit, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=True,
        )
        model.train()
        for epoch in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
            if trial is not None and X_eval is not None and y_eval is not None:
                model.eval()
                with torch.no_grad():
                    val_prob = torch.sigmoid(model(torch.tensor(X_eval, dtype=torch.float32))).numpy()
                val_pred = (val_prob >= 0.5).astype(int)
                try:
                    score = roc_auc_score(y_eval, val_prob)
                except ValueError:
                    score = f1_score(y_eval, val_pred)
                trial.report(float(score), step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                model.train()
        model.eval()
        return model

    if optuna_trials <= 0:
        best = {
            "hidden_1": 64,
            "hidden_2": 32,
            "dropout": 0.05,
            "lr": 0.01,
            "weight_decay": 1e-4,
            "batch_size": 64,
            "epochs": 35,
        }
    else:
        def objective(trial: optuna.Trial) -> float:
            torch.manual_seed(SEED)
            hidden_1 = trial.suggest_int("hidden_1", 16, 128, step=16)
            hidden_2 = trial.suggest_int("hidden_2", 8, 64, step=8)
            dropout = trial.suggest_float("dropout", 0.0, 0.35)
            lr = trial.suggest_float("lr", 1e-4, 3e-2, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            epochs = trial.suggest_int("epochs", 12, 55)
            model = BinaryMLP(X.shape[1], hidden_1, hidden_2, dropout)
            model = run_training(model, X_train, y_train, lr, weight_decay, int(batch_size), int(epochs), trial, X_val, y_val)
            with torch.no_grad():
                val_prob = torch.sigmoid(model(torch.tensor(X_val, dtype=torch.float32))).numpy()
            val_pred = (val_prob >= 0.5).astype(int)
            try:
                return float(roc_auc_score(y_val, val_prob))
            except ValueError:
                return float(f1_score(y_val, val_pred))

        study = get_or_create_study(study_name=study_name, direction="maximize")
        ensure_study_trials(study, objective, optuna_trials, timeout_sec)
        best = dict(study.best_params)

    torch.manual_seed(SEED)
    final_model = BinaryMLP(X.shape[1], int(best["hidden_1"]), int(best["hidden_2"]), float(best["dropout"]))
    final_model = run_training(final_model, X, y, float(best["lr"]), float(best["weight_decay"]), int(best["batch_size"]), int(best["epochs"]))
    total_time = time.perf_counter() - start_total
    return final_model, total_time, best


def train_multi_mlp(
    X: np.ndarray,
    y: np.ndarray,
    optuna_trials: int = OPTUNA_TRIALS_MLP,
    timeout_sec: int | None = None,
    study_name: str = "multi_mlp",
) -> tuple[nn.Module, float, dict]:
    start_total = time.perf_counter()
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    n_classes = int(np.max(y)) + 1
    X_train, X_val, y_train, y_val = split_data(X, y)

    def run_training(
        model: nn.Module,
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        lr: float,
        weight_decay: float,
        batch_size: int,
        epochs: int,
        trial: optuna.Trial | None = None,
        X_eval: np.ndarray | None = None,
        y_eval: np.ndarray | None = None,
    ) -> nn.Module:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        loader = DataLoader(
            TensorDataset(torch.tensor(X_fit, dtype=torch.float32), torch.tensor(y_fit, dtype=torch.long)),
            batch_size=batch_size,
            shuffle=True,
        )
        model.train()
        for epoch in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
            if trial is not None and X_eval is not None and y_eval is not None:
                model.eval()
                with torch.no_grad():
                    logits = model(torch.tensor(X_eval, dtype=torch.float32))
                    pred = torch.argmax(logits, dim=1).numpy()
                score = f1_score(y_eval, pred, average="macro")
                trial.report(float(score), step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                model.train()
        model.eval()
        return model

    if optuna_trials <= 0:
        best = {
            "hidden_1": 128,
            "hidden_2": 64,
            "dropout": 0.10,
            "lr": 0.006,
            "weight_decay": 1e-4,
            "batch_size": 64,
            "epochs": 45,
        }
    else:
        def objective(trial: optuna.Trial) -> float:
            torch.manual_seed(SEED)
            hidden_1 = trial.suggest_int("hidden_1", 32, 192, step=32)
            hidden_2 = trial.suggest_int("hidden_2", 16, 96, step=16)
            dropout = trial.suggest_float("dropout", 0.0, 0.40)
            lr = trial.suggest_float("lr", 1e-4, 3e-2, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            epochs = trial.suggest_int("epochs", 12, 65)
            model = MultiMLP(X.shape[1], n_classes, hidden_1, hidden_2, dropout)
            model = run_training(model, X_train, y_train, lr, weight_decay, int(batch_size), int(epochs), trial, X_val, y_val)
            with torch.no_grad():
                logits = model(torch.tensor(X_val, dtype=torch.float32))
                pred = torch.argmax(logits, dim=1).numpy()
            return float(f1_score(y_val, pred, average="macro"))

        study = get_or_create_study(study_name=study_name, direction="maximize")
        ensure_study_trials(study, objective, optuna_trials, timeout_sec)
        best = dict(study.best_params)

    torch.manual_seed(SEED)
    final_model = MultiMLP(X.shape[1], n_classes, int(best["hidden_1"]), int(best["hidden_2"]), float(best["dropout"]))
    final_model = run_training(final_model, X, y, float(best["lr"]), float(best["weight_decay"]), int(best["batch_size"]), int(best["epochs"]))
    total_time = time.perf_counter() - start_total
    return final_model, total_time, best


def binary_metrics(
    task: str,
    source: str,
    xgb,
    mlp,
    X_test,
    X_test_scaled,
    y_test,
    xgb_time,
    mlp_time,
    xgb_params: dict | None = None,
    mlp_params: dict | None = None,
) -> pd.DataFrame:
    xgb_prob = xgb.predict_proba(X_test)[:, 1]
    with torch.no_grad():
        mlp_prob = torch.sigmoid(mlp(torch.tensor(X_test_scaled, dtype=torch.float32))).numpy()
    rows = []
    for name, prob, sec, params in [
        ("XGBoost", xgb_prob, xgb_time, xgb_params or {}),
        ("Torch MLP", mlp_prob, mlp_time, mlp_params or {}),
    ]:
        pred = (prob >= 0.5).astype(int)
        try:
            auc = roc_auc_score(y_test, prob)
        except ValueError:
            auc = np.nan
        rows.append(
            {
                "Task": task,
                "Model": name,
                "Accuracy": round(float(accuracy_score(y_test, pred)), 4),
                "F1": round(float(f1_score(y_test, pred)), 4),
                "ROC-AUC": round(float(auc), 4) if not np.isnan(auc) else np.nan,
                "Train sec incl. Optuna": round(float(sec), 3),
                "Source": source,
                "Best params": json.dumps(params, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def train_binary_bundle(
    task: str,
    loader,
    max_depth: int,
    optuna_trials_xgb: int = OPTUNA_TRIALS_XGB,
    optuna_trials_mlp: int = OPTUNA_TRIALS_MLP,
):
    X, y, source = loader()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_xgb_train, X_xgb_val, y_xgb_train, y_xgb_val = split_data(X_train, y_train)
    study_prefix = safe_name(f"{task}_{source}_{X.shape[1]}features")

    if optuna_trials_xgb <= 0:
        best_xgb_params = {
            "n_estimators": 130,
            "max_depth": max_depth,
            "learning_rate": 0.06,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "reg_alpha": 1e-8,
            "reg_lambda": 1.0,
        }
        xgb_time = 0.0
    else:
        def xgb_objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 80, 260, step=20),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
                "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
                "gamma": trial.suggest_float("gamma", 0.0, 4.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "eval_metric": "logloss",
                "random_state": SEED,
                "n_jobs": 8,
                "tree_method": "hist",
            }
            model = XGBClassifier(**params)
            model.fit(X_xgb_train, y_xgb_train)
            prob = model.predict_proba(X_xgb_val)[:, 1]
            pred = (prob >= 0.5).astype(int)
            try:
                return float(roc_auc_score(y_xgb_val, prob))
            except ValueError:
                return float(f1_score(y_xgb_val, pred))

        start_xgb = time.perf_counter()
        xgb_study = get_or_create_study(study_name=f"{study_prefix}_xgboost", direction="maximize")
        ensure_study_trials(xgb_study, xgb_objective, optuna_trials_xgb)
        best_xgb_params = dict(xgb_study.best_params)
        xgb_time = time.perf_counter() - start_xgb

    final_xgb_params = {
        **best_xgb_params,
        "eval_metric": "logloss",
        "random_state": SEED,
        "n_jobs": 8,
        "tree_method": "hist",
    }
    start_final_xgb = time.perf_counter()
    xgb = XGBClassifier(**final_xgb_params)
    xgb.fit(X_train, y_train)
    xgb_time += time.perf_counter() - start_final_xgb

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp, mlp_time, best_mlp_params = train_binary_mlp(
        X_train_scaled,
        y_train,
        optuna_trials=optuna_trials_mlp,
        study_name=f"{study_prefix}_torch_mlp",
    )

    metric_table = binary_metrics(
        task=task,
        source=source,
        xgb=xgb,
        mlp=mlp,
        X_test=X_test,
        X_test_scaled=X_test_scaled,
        y_test=y_test,
        xgb_time=xgb_time,
        mlp_time=mlp_time,
        xgb_params=best_xgb_params,
        mlp_params=best_mlp_params,
    )

    return {
        "xgb": xgb,
        "mlp": mlp,
        "scaler": scaler,
        "metrics": metric_table,
        "source": source,
        "rows": len(X),
        "xgb_params": best_xgb_params,
        "mlp_params": best_mlp_params,
    }


def train_genre_bundle(
    optuna_trials_xgb: int = OPTUNA_TRIALS_XGB,
    optuna_trials_mlp: int = OPTUNA_TRIALS_MLP,
):
    df, source = load_genre_dataframe()
    df = df[["title", "description", "genre"]].dropna().copy()
    labels = sorted(df["genre"].unique())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    y = df["genre"].map(label_to_id).to_numpy()

    X_train_text, X_test_text, y_train, y_test = split_data(df["description"].astype(str).tolist(), y)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=700)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    X_xgb_train, X_xgb_val, y_xgb_train, y_xgb_val = split_data(X_train, y_train)
    study_prefix = safe_name(f"genre_classifier_{source}_{X_train.shape[1]}features")

    if optuna_trials_xgb <= 0:
        best_xgb_params = {
            "n_estimators": 160,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "reg_alpha": 1e-8,
            "reg_lambda": 1.0,
        }
        xgb_time = 0.0
    else:
        def xgb_objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 80, 260, step=20),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
                "gamma": trial.suggest_float("gamma", 0.0, 4.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "eval_metric": "mlogloss",
                "random_state": SEED,
                "n_jobs": 8,
                "tree_method": "hist",
            }
            model = XGBClassifier(**params)
            model.fit(X_xgb_train, y_xgb_train)
            pred = model.predict(X_xgb_val)
            return float(f1_score(y_xgb_val, pred, average="macro"))

        start_xgb = time.perf_counter()
        xgb_study = get_or_create_study(study_name=f"{study_prefix}_xgboost", direction="maximize")
        ensure_study_trials(xgb_study, xgb_objective, optuna_trials_xgb)
        best_xgb_params = dict(xgb_study.best_params)
        xgb_time = time.perf_counter() - start_xgb

    final_xgb_params = {
        **best_xgb_params,
        "eval_metric": "mlogloss",
        "random_state": SEED,
        "n_jobs": 8,
        "tree_method": "hist",
    }
    start_final_xgb = time.perf_counter()
    xgb = XGBClassifier(**final_xgb_params)
    xgb.fit(X_train, y_train)
    xgb_time += time.perf_counter() - start_final_xgb

    X_train_dense = to_dense_float32(X_train)
    X_test_dense = to_dense_float32(X_test)
    mlp, mlp_time, best_mlp_params = train_multi_mlp(
        X_train_dense,
        y_train,
        optuna_trials=optuna_trials_mlp,
        study_name=f"{study_prefix}_torch_mlp",
    )
    xgb_pred = xgb.predict(X_test)
    with torch.no_grad():
        mlp_logits = mlp(torch.tensor(X_test_dense, dtype=torch.float32))
        mlp_probs = torch.softmax(mlp_logits, dim=1).numpy()
        mlp_pred = np.argmax(mlp_probs, axis=1)

    metrics = pd.DataFrame(
        [
            {
                "Task": "Genre classifier",
                "Model": "XGBoost",
                "Accuracy": round(float(accuracy_score(y_test, xgb_pred)), 4),
                "Macro F1": round(float(f1_score(y_test, xgb_pred, average="macro")), 4),
                "Train sec incl. Optuna": round(float(xgb_time), 3),
                "Source": source,
                "Best params": json.dumps(best_xgb_params, ensure_ascii=False),
            },
            {
                "Task": "Genre classifier",
                "Model": "Torch MLP",
                "Accuracy": round(float(accuracy_score(y_test, mlp_pred)), 4),
                "Macro F1": round(float(f1_score(y_test, mlp_pred, average="macro")), 4),
                "Train sec incl. Optuna": round(float(mlp_time), 3),
                "Source": source,
                "Best params": json.dumps(best_mlp_params, ensure_ascii=False),
            },
        ]
    )

    return {
        "xgb": xgb,
        "mlp": mlp,
        "vectorizer": vectorizer,
        "labels": labels,
        "metrics": metrics,
        "source": source,
        "rows": len(df),
        "xgb_params": best_xgb_params,
        "mlp_params": best_mlp_params,
    }


# ============================================================
# CLIP image relevance
# ============================================================

def get_clip_model():
    global CLIP_MODEL, CLIP_PROCESSOR

    if CLIP_MODEL is None or CLIP_PROCESSOR is None:
        from transformers import CLIPModel, CLIPProcessor

        print(f"Loading CLIP model: {CLIP_MODEL_ID} on {CLIP_DEVICE}")
        CLIP_PROCESSOR = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        CLIP_MODEL = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        CLIP_MODEL.to(CLIP_DEVICE)
        CLIP_MODEL.eval()

    return CLIP_MODEL, CLIP_PROCESSOR


def clip_image_text_scores(image: Any, texts: list[str]) -> np.ndarray:
    model, processor = get_clip_model()
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    inputs = {key: value.to(CLIP_DEVICE) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0].detach().cpu().numpy()
    return softmax_numpy(logits)


def evaluate_product_image(image, selected_category: str, customer_preference: str):
    if image is None:
        return "Please upload a JPG, JPEG or PNG product image first.", pd.DataFrame()

    if not customer_preference or not customer_preference.strip():
        customer_preference = "a product that matches the customer preference"

    try:
        from PIL import Image

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("RGB")

        category_prompts = [f"a product photo of a {category.lower()}" for category in IMAGE_PRODUCT_CATEGORIES]
        category_scores = clip_image_text_scores(image, category_prompts)
        category_table = pd.DataFrame({"Check": IMAGE_PRODUCT_CATEGORIES, "Score": np.round(category_scores, 4)}).sort_values("Score", ascending=False)
        category_table["Score %"] = (category_table["Score"] * 100).round(1).astype(str) + " %"
        category_table["Type"] = "Visual category check"

        selected_category_index = IMAGE_PRODUCT_CATEGORIES.index(selected_category)
        selected_category_score = float(category_scores[selected_category_index])

        preference_prompts = [
            f"a product photo that matches this customer preference: {customer_preference}",
            f"a product photo that partially matches this customer preference: {customer_preference}",
            f"a product photo that does not match this customer preference: {customer_preference}",
            f"a product photo from the selected category: {selected_category}",
            f"a product photo from a different category than: {selected_category}",
        ]
        preference_scores = clip_image_text_scores(image, preference_prompts)

        match_score = float(preference_scores[0])
        partial_score = float(preference_scores[1])
        mismatch_score = float(preference_scores[2])
        selected_category_prompt_score = float(preference_scores[3])
        different_category_prompt_score = float(preference_scores[4])

        combined_score = 0.45 * selected_category_score + 0.45 * match_score + 0.10 * selected_category_prompt_score

        if combined_score >= 0.65:
            decision = "Good fit"
        elif combined_score >= 0.45:
            decision = "Partial fit"
        else:
            decision = "Weak fit"

        summary = (
            f"Selected category: {selected_category}\n"
            f"Customer preference: {customer_preference}\n\n"
            f"Selected category match: {selected_category_score * 100:.1f} %\n"
            f"Customer preference match: {match_score * 100:.1f} %\n"
            f"Partial match signal: {partial_score * 100:.1f} %\n"
            f"Mismatch signal: {mismatch_score * 100:.1f} %\n\n"
            f"Combined relevance score: {combined_score * 100:.1f} %\n"
            f"Decision: {decision}\n\n"
            "Audience interpretation:\n"
            "The model compares the uploaded image with text descriptions. "
            "It does not know the product name directly. It estimates whether the visual content "
            "looks similar to the selected category and customer preference."
        )

        prompt_table = pd.DataFrame(
            {
                "Check": [
                    "Matches customer preference",
                    "Partially matches customer preference",
                    "Does not match customer preference",
                    "Looks like selected category",
                    "Looks like different category",
                ],
                "Score": [
                    round(match_score, 4),
                    round(partial_score, 4),
                    round(mismatch_score, 4),
                    round(selected_category_prompt_score, 4),
                    round(different_category_prompt_score, 4),
                ],
                "Type": [
                    "Preference check",
                    "Preference check",
                    "Preference check",
                    "Category check",
                    "Category check",
                ],
            }
        )
        prompt_table["Score %"] = (prompt_table["Score"] * 100).round(1).astype(str) + " %"
        result_table = pd.concat(
            [prompt_table[["Type", "Check", "Score", "Score %"]], category_table[["Type", "Check", "Score", "Score %"]]],
            ignore_index=True,
        )
        return summary, result_table

    except Exception as exc:
        message = (
            "The image relevance model could not be loaded or evaluated.\n\n"
            f"Error: {exc}\n\n"
            "Most common fix:\n"
            "Install dependencies and make sure the CLIP model is available:\n"
            "pip install transformers pillow\n\n"
            "If you are offline, run it once with internet access so the model is cached locally."
        )
        return message, pd.DataFrame()


# ============================================================
# Train models at startup
# ============================================================

print("Training/loading Movie model")
MOVIE = train_binary_bundle(
    "Movie recommender",
    load_movie_data,
    max_depth=5,
    optuna_trials_xgb=OPTUNA_TRIALS_XGB,
    optuna_trials_mlp=OPTUNA_TRIALS_MLP,
)

print("Product image relevance uses CLIP zero-shot model. No startup training required.")

print("Training/loading Genre model")
GENRE_MODEL = train_genre_bundle(
    optuna_trials_xgb=OPTUNA_TRIALS_XGB,
    optuna_trials_mlp=OPTUNA_TRIALS_MLP,
)


# ============================================================
# Presentation outputs
# ============================================================

def build_dashboard_markdown() -> str:
    model_count = 5
    total_rows = MOVIE["rows"] + GENRE_MODEL["rows"]

    return f"""
<div class="hero-card">

# Machine Learning demo dashboard

Application to show ML behavior on real-life examples:
**XGBoost**, **Torch MLP neural network**, and **CLIP image-text model**.

</div>

<div class="metric-grid">

<div class="metric-card">
<h3>{model_count}</h3>
<div>Models used</div>
<div class="small-note">XGBoost + Torch MLP + CLIP</div>
</div>

<div class="metric-card">
<h3>{total_rows:,}</h3>
<div>Rows in trained datasets</div>
<div class="small-note">Movie and genre CSV/XLSX/Kaggle files</div>
</div>

<div class="metric-card">
<h3>Image upload</h3>
<div>Product relevance</div>
<div class="small-note">JPG / PNG / JPEG via CLIP</div>
</div>

<div class="metric-card">
<h3>Optuna</h3>
<div>Auto tuning</div>
<div class="small-note">Cache: Cache/optuna_studies.sqlite3</div>
</div>

</div>
"""


def get_all_metrics() -> pd.DataFrame:
    clip_row = pd.DataFrame(
        [
            {
                "Task": "Product image relevance",
                "Model": f"CLIP zero-shot ({CLIP_MODEL_ID})",
                "Accuracy": np.nan,
                "F1": np.nan,
                "ROC-AUC": np.nan,
                "Train sec incl. Optuna": 0.0,
                "Source": "Uploaded JPG/PNG/JPEG image + customer preference text",
                "Best params": "No training. Zero-shot image-text similarity.",
            }
        ]
    )
    return pd.concat([MOVIE["metrics"], clip_row, GENRE_MODEL["metrics"]], ignore_index=True)


def get_training_sources() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Dataset": "Movie recommender",
                "Source": MOVIE["source"],
                "Rows": MOVIE["rows"],
                "Purpose": "Movie recommendations",
            },
            {
                "Dataset": "Product image relevance",
                "Source": f"Uploaded image + {CLIP_MODEL_ID}",
                "Rows": "N/A",
                "Purpose": "Image-category and customer preference matching",
            },
            {
                "Dataset": "Genre classifier",
                "Source": GENRE_MODEL["source"],
                "Rows": GENRE_MODEL["rows"],
                "Purpose": "Genre prediction from description",
            },
        ]
    )


# ============================================================
# Prediction functions
# ============================================================

def predict_movie(action, comedy, drama, scifi, romance, documentary, genre, rating, age, length, mood):
    row = pd.DataFrame(
        [[action, comedy, drama, scifi, romance, documentary, GENRES.index(genre), rating, age, length, MOODS.index(mood)]],
        columns=[
            "user_action",
            "user_comedy",
            "user_drama",
            "user_scifi",
            "user_romance",
            "user_documentary",
            "movie_genre_id",
            "movie_rating",
            "movie_age_years",
            "movie_length_min",
            "mood_id",
        ],
    )
    xgb_prob = float(MOVIE["xgb"].predict_proba(row)[0, 1])
    with torch.no_grad():
        mlp_prob = float(torch.sigmoid(MOVIE["mlp"](torch.tensor(MOVIE["scaler"].transform(row), dtype=torch.float32))).numpy()[0])

    verdict = "Recommended" if (xgb_prob + mlp_prob) / 2 >= 0.5 else "Probably not ideal"
    summary = (
        f"XGBoost: {xgb_prob * 100:.1f} %\n"
        f"Torch MLP: {mlp_prob * 100:.1f} %\n\n"
        f"Verdict: {verdict}\n"
        f"Source: {MOVIE['source']} ({MOVIE['rows']} rows)\n\n"
        "Audience interpretation:\n"
        "Both models solve the same recommendation task, but each model learns patterns differently."
    )
    return summary, MOVIE["metrics"]


def predict_genre(description):
    X = GENRE_MODEL["vectorizer"].transform([description or ""])
    X_dense = to_dense_float32(X)
    xgb_probs = GENRE_MODEL["xgb"].predict_proba(X)[0]
    with torch.no_grad():
        mlp_probs = torch.softmax(GENRE_MODEL["mlp"](torch.tensor(X_dense, dtype=torch.float32)), dim=1).numpy()[0]
    labels = GENRE_MODEL["labels"]
    table = pd.DataFrame(
        {
            "Genre": labels,
            "XGBoost probability": np.round(xgb_probs, 4),
            "Torch MLP probability": np.round(mlp_probs, 4),
            "Difference": np.round(np.abs(xgb_probs - mlp_probs), 4),
        }
    ).sort_values("XGBoost probability", ascending=False)
    xgb_label = labels[int(np.argmax(xgb_probs))]
    mlp_label = labels[int(np.argmax(mlp_probs))]
    summary = (
        f"XGBoost predicts: {xgb_label} ({np.max(xgb_probs) * 100:.1f} %)\n"
        f"Torch MLP predicts: {mlp_label} ({np.max(mlp_probs) * 100:.1f} %)\n\n"
    )
    if xgb_label == mlp_label:
        summary += "Models agree. This is an example of a consistent signal."
    else:
        summary += "Models disagree. This shows that different algorithms can read different signals in the same text."
    return summary, table, GENRE_MODEL["metrics"]


# ============================================================
# Gradio UI
# ============================================================

with gr.Blocks(title="ML Demo - XGBoost vs Torch MLP + CLIP") as demo:
    gr.Markdown(build_dashboard_markdown())

    with gr.Tabs():
        with gr.Tab("Dashboard"):
            gr.Markdown(
                """
                ## Overview

                This dashboard shows loaded datasets, model metrics and training sources.
                """
            )
            all_metrics_table = gr.Dataframe(value=get_all_metrics(), label="Model metrics", interactive=False)
            sources_table = gr.Dataframe(value=get_training_sources(), label="Datasets", interactive=False)

        with gr.Tab("1. Movie recommender"):
            gr.Markdown(
                """
                ## Movie recommendation

                The model receives user preferences and movie attributes, then predicts
                whether the movie is likely to be a good recommendation.
                """
            )
            with gr.Row():
                with gr.Column():
                    action = gr.Slider(0, 1, value=0.7, step=0.1, label="Likes action")
                    comedy = gr.Slider(0, 1, value=0.4, step=0.1, label="Likes comedy")
                    drama = gr.Slider(0, 1, value=0.3, step=0.1, label="Likes drama")
                    scifi = gr.Slider(0, 1, value=0.8, step=0.1, label="Likes sci-fi")
                    romance = gr.Slider(0, 1, value=0.2, step=0.1, label="Likes romance")
                    documentary = gr.Slider(0, 1, value=0.25, step=0.1, label="Likes documentaries")
                with gr.Column():
                    genre = gr.Dropdown(GENRES, value="Sci-Fi", label="Movie genre")
                    rating = gr.Slider(3.5, 9.8, value=8.1, step=0.1, label="Movie rating")
                    age = gr.Slider(0, 80, value=6, step=1, label="Movie age in years")
                    length = gr.Slider(60, 240, value=125, step=1, label="Movie length in minutes")
                    mood = gr.Dropdown(MOODS, value="Mind-bending", label="Current mood")
                    movie_btn = gr.Button("Predict", variant="primary")
            movie_out = gr.Textbox(label="Prediction", lines=8, elem_classes=["prediction-box"])
            movie_metrics = gr.Dataframe(label="Movie recommender metrics", interactive=False)
            movie_btn.click(
                predict_movie,
                [action, comedy, drama, scifi, romance, documentary, genre, rating, age, length, mood],
                [movie_out, movie_metrics],
            )

        with gr.Tab("2. Product image relevance"):
            gr.Markdown(
                """
                ## Product image relevance

                Upload a product image, choose the expected category and describe the customer preference.

                This tab uses a CLIP image-text model. It is easier to present to non-technical users
                because the audience can directly see the product image and the resulting match score.
                """
            )
            with gr.Row():
                with gr.Column(scale=1):
                    product_image = gr.Image(label="Upload product image", type="pil", sources=["upload"], image_mode="RGB")
                    selected_product_category = gr.Dropdown(IMAGE_PRODUCT_CATEGORIES, value="Tech gadget", label="Expected product category")
                    customer_preference = gr.Textbox(
                        label="Customer preference",
                        lines=4,
                        value="black wireless headphones suitable for office calls and travel",
                        placeholder="Example: lightweight running shoes for gym training",
                    )
                    product_image_btn = gr.Button("Evaluate image relevance", variant="primary")
                with gr.Column(scale=1):
                    product_image_out = gr.Textbox(label="Image relevance result", lines=14, elem_classes=["prediction-box"])
                    product_image_scores = gr.Dataframe(label="Model scores", interactive=False, elem_classes=["dataframe-table"])
            product_image_btn.click(
                evaluate_product_image,
                [product_image, selected_product_category, customer_preference],
                [product_image_out, product_image_scores],
            )

        with gr.Tab("3. Genre classifier"):
            gr.Markdown(
                """
                ## Genre classification

                The model receives a movie description and predicts the most likely genre.
                """
            )
            description = gr.Textbox(value="A starship crew discovers an alien signal on a distant planet.", lines=4, label="Movie description")
            genre_btn = gr.Button("Classify", variant="primary")
            genre_out = gr.Textbox(label="Prediction", lines=6, elem_classes=["prediction-box"])
            genre_probs = gr.Dataframe(label="Class probabilities", interactive=False)
            genre_metrics = gr.Dataframe(label="Genre classifier metrics", interactive=False)
            genre_btn.click(predict_genre, [description], [genre_out, genre_probs, genre_metrics])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", inbrowser=True, share=False, css=CUSTOM_CSS)
