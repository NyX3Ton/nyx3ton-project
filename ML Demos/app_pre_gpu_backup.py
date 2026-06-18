from __future__ import annotations

import json, os, re, time, optuna, torch, datetime
from pathlib import Path
from typing import Any
import gradio as gr
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

try:
    from generate_inputs import generate_movie_genre
except ImportError:
    from generate_inputs import generate_movie_genre

# ============================================================
# Global config
# ============================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
try:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
except RuntimeError:
    # Thread pools may already be initialized in some environments.
    pass

optuna.logging.set_verbosity(optuna.logging.WARNING)

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")

BASE_DIR = Path(__file__).resolve().parent
INPUTS_DIR = BASE_DIR / "Inputs"
INPUTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = BASE_DIR / "Cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OPTUNA_STORAGE = f"sqlite:///{(CACHE_DIR / 'optuna_studies.sqlite3').as_posix()}"

GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Fantasy", "Romance", "Documentary"]
MOODS = ["Relax", "Funny", "Emotional", "Adrenaline", "Mind-bending"]

IMAGE_PRODUCT_CATEGORIES = ["Tech gadget", "Gaming accessory", "Fitness product", "Travel gear","Food product", "Fashion item", "Home appliance", "Book","Musical instrument", "Sports equipment", "Office equipment", "Beauty product"]

OPTUNA_TRIALS_XGB = int(os.getenv("OPTUNA_TRIALS_XGB", "15"))
OPTUNA_TRIALS_MLP = int(os.getenv("OPTUNA_TRIALS_MLP", "15"))
#MAX_GENRE_ROWS = int(os.getenv("MAX_GENRE_ROWS", "20000"))

# Playlist creator (Spotify) settings
SPOTIFY_CSV_NAMES = ["spotify_data.csv", "spotify_1million_tracks.csv", "spotify_tracks.csv"]
#SPOTIFY_AUDIO_FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness","instrumentalness", "liveness", "valence", "tempo"]
SPOTIFY_AUDIO_FEATURES = ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
TOP_GENRES_N = int(os.getenv("TOP_GENRES_N", "14"))
MAX_SPOTIFY_ROWS = int(os.getenv("MAX_SPOTIFY_ROWS", "24000"))
PLAYLIST_CANDIDATES = int(os.getenv("PLAYLIST_CANDIDATES", "6000"))

CLIP_MODEL_ID = os.getenv("CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
OPENVINO_DEVICE = os.getenv("OPENVINO_DEVICE", "CPU")

CLIP_DEVICE = None
CLIP_BACKEND = None
CLIP_MODEL = None
CLIP_PROCESSOR = None
CLIP_TEXT_INPUT_CACHE: dict[tuple[str, ...], dict[str, torch.Tensor]] = {}

MOVIE_CSV_NAMES = ["IMDb_Genres_real_enriched.csv", "IMDb_Genres_real_enriched.xlsx"]

PREF_COL = {
            "action": "preference_action_score",
            "comedy": "preference_comedy_score",
            "drama": "preference_drama_score",
            "scifi": "preference_sci_fi_score",
            "romance": "preference_romance_score",
            "documentary": "preference_documentary_score",
            }
MOOD_COL = {
            "Relax": "mood_relax",
            "Funny": "mood_funny",
            "Emotional": "mood_emotional",
            "Adrenaline": "mood_adrenaline",
            "Mind-bending": "mood_mind_bending",
            }

SCORE_WEIGHTS = {"pref": 0.40, "mood": 0.20, "genre": 0.15, "rating": 0.15, "length": 0.10}

CUSTOM_CSS = """
.gradio-container { max-width: 1550px !important; }
.hero-card {
            padding: 24px; border-radius: 20px;
            background: linear-gradient(135deg, rgba(25, 80, 170, 0.16), rgba(120, 60, 180, 0.14));
            border: 1px solid rgba(120, 120, 120, 0.20); margin-bottom: 16px;
            }
.metric-grid { display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 12px; }
.metric-card {
                padding: 15px 18px; border-radius: 16px; background: rgba(120, 120, 120, 0.09);
                border: 1px solid rgba(120, 120, 120, 0.16); min-height: 100px;
                }
.metric-card h3 { margin-top: 0; margin-bottom: 6px; font-size: 1.45rem; }
.small-note { opacity: 0.82; font-size: 0.92rem; }
.prediction-box textarea { font-size: 1.04rem !important; line-height: 1.45 !important; }
.dataframe-table { font-size: 0.92rem; }
"""
# ============================================================
# Small helpers
# ============================================================
def safe_name(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.lower().strip())
    return value.strip("_") or "study"

def to_dense_float32(matrix: object) -> np.ndarray:
    for attr in ("toarray", "to_numpy"):
        fn = getattr(matrix, attr, None)
        if callable(fn):
            return np.asarray(fn(), dtype=np.float32)
    return np.asarray(matrix, dtype=np.float32)

def find_existing_input(candidates: list[str]) -> Path | None:
    for name in candidates:
        for folder in (INPUTS_DIR, BASE_DIR):
            path = folder / name
            if path.exists():
                return path
    return None

def read_table_file(path: Path, **kwargs) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt"):
        kwargs.setdefault("encoding", "utf-8-sig")
        return pd.read_csv(path, **kwargs)
    if suffix in (".xlsx", ".xls"):
        kwargs.pop("low_memory", None)
        kwargs.pop("encoding", None)
        return pd.read_excel(path, **kwargs)
    raise ValueError(f"Unsupported input file type: {path}")

def split_data(X, y):
    try:
        return train_test_split(X, y, test_size= 0.2, stratify=y, random_state=SEED)
    except ValueError:
        return train_test_split(X, y, test_size= 0.2, random_state=SEED)

def softmax_numpy(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values)
# ============================================================
# Persistent Optuna helpers (used by the genre classifier)
# ============================================================
def get_or_create_study(study_name: str, direction: str = "maximize") -> optuna.Study:
    return optuna.create_study(
                                study_name=study_name,
                                storage=OPTUNA_STORAGE,
                                direction=direction,
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
                            )

def ensure_study_trials(study: optuna.Study, objective, required_trials: int, timeout_sec: int | None = None) -> None:
    if required_trials <= 0:
        return
    existing = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    if existing >= required_trials:
        print(f"Optuna study '{study.study_name}' already has {existing}/{required_trials} trials. Skipping.")
        return
    missing = required_trials - existing
    print(f"Optuna study '{study.study_name}' running {missing} new trials ({existing}/{required_trials}).")
    study.optimize(objective, n_trials=missing, timeout=timeout_sec, show_progress_bar=False)
# ============================================================
# Movie recommender (content-based, no training)
# ============================================================
def load_movie_catalog() -> dict:
    path = find_existing_input(MOVIE_CSV_NAMES)
    if path is None:
        raise FileNotFoundError(
                                "Movie catalog not found. Place 'IMDb_Genres_real_enriched.csv' next to app.py "
                                f"or in the Inputs folder ({INPUTS_DIR})."
                                )

    df = read_table_file(path)

    numeric_cols = [
                    "imdb_rating_10", "runtime_minutes", "release_year", "popularity_score",
                    *PREF_COL.values(), *MOOD_COL.values(),
                    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in ("movie_title", "all_genres", "director_clean", "lead_actor",
                "overview_clean", "recommendation_explanation_base"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    df["_genres_lc"] = df["all_genres"].str.lower()
    return {"df": df.reset_index(drop=True), "source": path.name, "rows": len(df)}

def recommend_movies(action, comedy, drama, scifi, romance, documentary, genre, min_rating, max_age, length, mood, top_n: int = 8):
    df = MOVIE["df"]
    weights = {
                "action": action, 
                "comedy": comedy, 
                "drama": drama,
                "scifi": scifi, 
                "romance": romance, 
                "documentary": documentary,
                }
    wsum = sum(weights.values()) or 1.0

    pref = sum(w * df[PREF_COL[k]] for k, w in weights.items()) / wsum
    mood_part = df[MOOD_COL.get(mood, "mood_relax")]
    genre_boost = df["_genres_lc"].str.contains(re.escape(genre.lower()), regex=True).astype(float)
    rating_part = (df["imdb_rating_10"] / 10.0).clip(0, 1)
    length_part = (1 - (df["runtime_minutes"] - float(length)).abs() / 120.0).clip(0, 1)

    score = (SCORE_WEIGHTS["pref"] * pref + SCORE_WEIGHTS["mood"] * mood_part + SCORE_WEIGHTS["genre"] * genre_boost + SCORE_WEIGHTS["rating"] * rating_part + SCORE_WEIGHTS["length"] * length_part)
    scored = df.assign(_match=score)

    current_year = datetime.date.today().year
    min_year = current_year - float(max_age)
    notes = []
    for stage in range(3):
        keep = scored["_match"] > -1
        if stage < 2:
            keep &= scored["imdb_rating_10"] >= float(min_rating)
        if stage < 1:
            keep &= scored["release_year"] >= min_year
        result = scored[keep]
        if not result.empty:
            if stage == 1:
                notes.append(f"No title within the last {int(max_age)} years matched, so the age filter was relaxed.")
            if stage == 2:
                notes.append(f"No title with rating >= {min_rating} matched, so the rating filter was relaxed.")
            break
    else:
        result = scored

    result = result.sort_values(["_match", "imdb_rating_10", "popularity_score"], ascending=False).head(top_n)

    table = pd.DataFrame({
                            "Title": result["movie_title"],
                            "Year": result["release_year"].astype(int),
                            "Genres": result["all_genres"],
                            "IMDb": result["imdb_rating_10"].round(1),
                            "Runtime (min)": result["runtime_minutes"].astype(int),
                            "Director": result["director_clean"],
                            "Match %": (result["_match"] * 100).round(1),
                            }).reset_index(drop=True)

    top = result.iloc[0]
    explanation = top["recommendation_explanation_base"] or "good overall match for your preferences"
    summary = (
                f"Top pick: {top['movie_title']} ({int(top['release_year'])})\n"
                f"Match score: {top['_match'] * 100:.1f} %\n\n"
                f"Genres: {top['all_genres']}\n"
                f"IMDb rating: {top['imdb_rating_10']:.1f}/10   |   Runtime: {int(top['runtime_minutes'])} min\n"
                f"Director: {top['director_clean']}\n"
                f"Cast: {top['lead_actor']}\n\n"
                f"{top['overview_clean']}\n\n"
                f"Why recommended: {explanation}.\n"
                f"Source: {MOVIE['source']} ({MOVIE['rows']} movies)"
                )
    if notes:
        summary += "\n\nNote: " + " ".join(notes)

    return summary, table
# ============================================================
# Genre classifier (XGBoost vs Torch MLP)
# ============================================================
class MultiMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_1: int = 256, hidden_2: int = 128, dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
                                nn.Linear(n_features, hidden_1), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(hidden_1, hidden_2), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(hidden_2, n_classes),
                                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def load_spotify_dataframe() -> tuple[pd.DataFrame, str]:
    path = find_existing_input(SPOTIFY_CSV_NAMES)
    if path is None:
        raise FileNotFoundError(
                                "Spotify dataset not found. Place 'spotify_data.csv' next to app.py "
                                f"or in the Inputs folder ({INPUTS_DIR})."
                                )
    df = read_table_file(path, low_memory=False)
    needed = ["artist_name", "track_name", "genre", *SPOTIFY_AUDIO_FEATURES]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Spotify file {path.name} is missing columns: {missing}")

    for col in SPOTIFY_AUDIO_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("popularity", "year"):
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = df.dropna(subset=["genre", *SPOTIFY_AUDIO_FEATURES]).copy()
    df["genre"] = df["genre"].astype(str)
    df["artist_name"] = df["artist_name"].fillna("").astype(str)
    df["track_name"] = df["track_name"].fillna("").astype(str)

    top_genres = df["genre"].value_counts().head(TOP_GENRES_N).index.tolist()
    df = df[df["genre"].isin(top_genres)].copy()

    if len(df) > MAX_SPOTIFY_ROWS:
        per_genre = max(1, MAX_SPOTIFY_ROWS // len(top_genres))
        df = df.groupby("genre", group_keys=False).apply(
                        lambda g: g.sample(n=min(len(g), per_genre), random_state=SEED)
                        )
    return df.reset_index(drop=True), path.name

def train_multi_mlp(X, y, optuna_trials: int = OPTUNA_TRIALS_MLP, study_name: str = "multi_mlp") -> tuple[nn.Module, float, dict]:
    start_total = time.perf_counter()
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    n_classes = int(np.max(y)) + 1
    X_train, X_val, y_train, y_val = split_data(X, y)

    def run_training(model, X_fit, y_fit, lr, weight_decay, batch_size, epochs, trial=None, X_eval=None, y_eval=None):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(torch.tensor(X_fit, dtype=torch.float32), torch.tensor(y_fit, dtype=torch.long)),batch_size=batch_size, shuffle=True)
        model.train()
        for epoch in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss_fn(model(xb), yb).backward()
                optimizer.step()
            if trial is not None and X_eval is not None:
                model.eval()
                with torch.no_grad():
                    pred = torch.argmax(model(torch.tensor(X_eval, dtype=torch.float32)), dim=1).numpy()
                trial.report(float(f1_score(y_eval, pred, average="macro")), step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                model.train()
        model.eval()
        return model

    if optuna_trials <= 0:
        best = {"hidden_1": 256, "hidden_2": 128, "dropout": 0.2, "lr": 0.002,
                "weight_decay": 1.1, "batch_size": 128, "epochs": 75}
    else:
        def objective(trial: optuna.Trial) -> float:
            torch.manual_seed(SEED)
            model = MultiMLP(
                                X.shape[1], n_classes,
                                trial.suggest_int("hidden_1", 64, 256, step=64),
                                trial.suggest_int("hidden_2", 32, 128, step=32),
                                trial.suggest_float("dropout", 0.0, 0.2),
                            )
            model = run_training(
                                model, X_train, y_train,
                                trial.suggest_float("lr", 0.005, 0.1, log=True),
                                trial.suggest_float("weight_decay", 0.01, 1, log=True),
                                int(trial.suggest_categorical("batch_size", [32, 64, 128, 256])),
                                trial.suggest_int("epochs", 50, 300),
                                trial, X_val, y_val,
                                )
            with torch.no_grad():
                pred = torch.argmax(model(torch.tensor(X_val, dtype=torch.float32)), dim=1).numpy()
            return float(f1_score(y_val, pred, average="macro"))

        study = get_or_create_study(study_name=study_name)
        ensure_study_trials(study, objective, optuna_trials)
        best = dict(study.best_params)

    torch.manual_seed(SEED)
    final_model = MultiMLP(X.shape[1], n_classes, int(best["hidden_1"]), int(best["hidden_2"]), float(best["dropout"]))
    final_model = run_training(final_model, X, y, float(best["lr"]), float(best["weight_decay"]),int(best["batch_size"]), int(best["epochs"]))
    return final_model, time.perf_counter() - start_total, best

def train_playlist_bundle(optuna_trials_xgb: int = OPTUNA_TRIALS_XGB, optuna_trials_mlp: int = OPTUNA_TRIALS_MLP) -> dict:
    df, source = load_spotify_dataframe()
    labels = sorted(df["genre"].unique())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    y = df["genre"].map(label_to_id).to_numpy()

    scaler = StandardScaler()
    X = scaler.fit_transform(df[SPOTIFY_AUDIO_FEATURES].to_numpy(dtype=np.float32)).astype(np.float32)

    X_train, X_test, y_train, y_test = split_data(X, y)
    X_xgb_train, X_xgb_val, y_xgb_train, y_xgb_val = split_data(X_train, y_train)
    study_prefix = safe_name(f"playlist_{source}_{X.shape[1]}feat_{len(labels)}genres")

    if optuna_trials_xgb <= 0:
        best_xgb_params = {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.015,
                            "subsample": 0.7, "colsample_bytree": 0.75, "min_child_weight": 4.3,
                            "gamma": 2.0, "reg_alpha": 0.02, "reg_lambda": 0.001}
        xgb_time = 0.0
    else:
        def xgb_objective(trial: optuna.Trial) -> float:
            params = {
                        "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
                        "max_depth": trial.suggest_int("max_depth", 5, 8),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
                        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
                        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
                        "reg_lambda": trial.suggest_float("reg_lambda", 0.00005, 1, log=True),
                        "eval_metric": "mlogloss", 
                        "random_state": SEED, 
                        "n_jobs": 8, 
                        "device": "cuda",
                        }
            model = XGBClassifier(**params)
            model.fit(X_xgb_train, y_xgb_train)
            return float(f1_score(y_xgb_val, model.predict(X_xgb_val), average="macro"))

        start_xgb = time.perf_counter()
        xgb_study = get_or_create_study(study_name=f"{study_prefix}_xgboost")
        ensure_study_trials(xgb_study, xgb_objective, optuna_trials_xgb)
        best_xgb_params = dict(xgb_study.best_params)
        xgb_time = time.perf_counter() - start_xgb

    final_xgb_params = {**best_xgb_params, "eval_metric": "mlogloss", "random_state": SEED, "n_jobs": 8, "device": "cuda"}
    start_final_xgb = time.perf_counter()
    xgb = XGBClassifier(**final_xgb_params)
    xgb.fit(X_train, y_train)
    xgb_time += time.perf_counter() - start_final_xgb

    mlp, mlp_time, best_mlp_params = train_multi_mlp(X_train, y_train, optuna_trials=optuna_trials_mlp, study_name=f"{study_prefix}_torch_mlp")

    xgb_pred = xgb.predict(X_test)
    with torch.no_grad():
        mlp_pred = np.argmax(torch.softmax(mlp(torch.tensor(np.asarray(X_test, dtype=np.float32))), dim=1).numpy(), axis=1)

    metrics = pd.DataFrame([
                            {"Task": "Playlist creator", "Model": "XGBoost",
                                "Accuracy": round(float(accuracy_score(y_test, xgb_pred)), 4),
                                "Macro F1": round(float(f1_score(y_test, xgb_pred, average="macro")), 4),
                                "Train sec incl. Optuna": round(float(xgb_time), 3), "Source": source,
                                "Best params": json.dumps(best_xgb_params, ensure_ascii=False)},
                            {"Task": "Playlist creator", "Model": "Torch MLP",
                                "Accuracy": round(float(accuracy_score(y_test, mlp_pred)), 4),
                                "Macro F1": round(float(f1_score(y_test, mlp_pred, average="macro")), 4),
                                "Train sec incl. Optuna": round(float(mlp_time), 3), "Source": source,
                                "Best params": json.dumps(best_mlp_params, ensure_ascii=False)},
                            ])

    catalog = df.sample(n=min(len(df), PLAYLIST_CANDIDATES), random_state=SEED).reset_index(drop=True)
    catalog_scaled = scaler.transform(catalog[SPOTIFY_AUDIO_FEATURES].to_numpy(dtype=np.float32)).astype(np.float32)

    return {"xgb": xgb, "mlp": mlp, "scaler": scaler, "labels": labels, "label_to_id": label_to_id,
            "catalog": catalog, "catalog_scaled": catalog_scaled, "metrics": metrics,
            "source": source, "rows": len(df)}

def create_playlist(target_genre, energy, danceability, valence, acousticness, tempo, min_popularity, n_tracks):
    n_tracks = int(n_tracks)
    catalog = PLAYLIST_MODEL["catalog"]
    scaled = PLAYLIST_MODEL["catalog_scaled"]
    target_id = PLAYLIST_MODEL["label_to_id"].get(target_genre, 0)

    notes = []
    mask = (catalog["popularity"] >= float(min_popularity)).to_numpy()
    if mask.sum() < n_tracks:
        notes.append(f"Few tracks above popularity {int(min_popularity)}, so that filter was relaxed.")
        mask = np.ones(len(catalog), dtype=bool)
    idx = np.where(mask)[0]

    feats = scaled[idx]
    xgb_p = PLAYLIST_MODEL["xgb"].predict_proba(feats)[:, target_id]
    with torch.no_grad():
        mlp_all = torch.softmax(PLAYLIST_MODEL["mlp"](torch.tensor(feats, dtype=torch.float32)), dim=1).numpy()
    mlp_p = mlp_all[:, target_id]

    sub = catalog.iloc[idx]
    desired = np.array([energy, danceability, valence, acousticness, float(tempo) / 250.0], dtype=np.float32)
    track_taste = np.column_stack([
                        sub["energy"].to_numpy(),
                        sub["danceability"].to_numpy(),
                        sub["valence"].to_numpy(),
                        sub["acousticness"].to_numpy(),
                        sub["tempo"].to_numpy() / 250.0,
                        ]).astype(np.float32)
    dist = np.sqrt(((track_taste - desired) ** 2).mean(axis=1))
    taste = np.clip(1.0 - dist, 0.0, 1.0)

    score_xgb = 0.6 * xgb_p + 0.4 * taste
    score_mlp = 0.6 * mlp_p + 0.4 * taste

    def build(scores, model_p):
        scores = np.asarray(scores, dtype=np.float64)
        model_p = np.asarray(model_p, dtype=np.float64)
        order = np.argsort(scores)[::-1][:n_tracks]
        chosen = sub.iloc[order]
        table = pd.DataFrame({
                            "Track": chosen["track_name"].to_numpy(),
                            "Artist": chosen["artist_name"].to_numpy(),
                            "Actual genre": chosen["genre"].to_numpy(),
                            "Year": chosen["year"].astype(int).to_numpy(),
                            f"P({target_genre})": np.round(model_p[order], 3),
                            "Energy": np.round(chosen["energy"].to_numpy(), 2),
                            "Dance": np.round(chosen["danceability"].to_numpy(), 2),
                            "Valence": np.round(chosen["valence"].to_numpy(), 2),
                            "Tempo": np.round(chosen["tempo"].to_numpy(), 0),
                            "Match %": np.round(scores[order] * 100, 1),
                            }).reset_index(drop=True)
        picks = set(zip(chosen["track_name"].tolist(), chosen["artist_name"].tolist()))
        conf = float(np.mean(model_p[order])) if len(order) else 0.0
        return table, picks, conf

    xgb_table, xgb_set, xgb_conf = build(score_xgb, xgb_p)
    mlp_table, mlp_set, mlp_conf = build(score_mlp, mlp_p)
    overlap = len(xgb_set & mlp_set)

    summary = (
                f"Target vibe: {target_genre}\n"
                f"Playlist length: {n_tracks} tracks per model\n\n"
                f"Shared tracks (both models): {overlap}/{n_tracks}\n"
                f"XGBoost mean confidence in '{target_genre}': {xgb_conf * 100:.1f} %\n"
                f"Torch MLP mean confidence in '{target_genre}': {mlp_conf * 100:.1f} %\n\n"
                f"Candidate pool: {len(sub):,} tracks from {PLAYLIST_MODEL['source']}.\n"
                )
    summary += ("Models largely agree on this vibe." if overlap >= n_tracks / 2
                else "Models pick noticeably different tracks. Different algorithms read the same audio features differently.")
    if notes:
        summary += "\n\nNote: " + " ".join(notes)
    return summary, xgb_table, mlp_table, PLAYLIST_MODEL["metrics"]
# ============================================================
# CLIP image relevance
# ============================================================
def load_clip_processor():
    from transformers import CLIPProcessor

    try:
        return CLIPProcessor.from_pretrained(CLIP_MODEL_ID,use_fast=False)
    except TypeError:
        return CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

def get_clip_model():
    global CLIP_MODEL, CLIP_PROCESSOR, CLIP_DEVICE, CLIP_BACKEND

    if CLIP_MODEL is not None and CLIP_PROCESSOR is not None:
        return CLIP_MODEL, CLIP_PROCESSOR, CLIP_DEVICE, CLIP_BACKEND

    CLIP_PROCESSOR = load_clip_processor()

    if torch.cuda.is_available():
        from transformers import CLIPModel

        CLIP_DEVICE = "cuda"
        CLIP_BACKEND = "Torch CUDA"

        print(f"Loading CLIP model: {CLIP_MODEL_ID} on CUDA")
        CLIP_MODEL = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        CLIP_MODEL.to(CLIP_DEVICE)
        CLIP_MODEL.eval()

        return CLIP_MODEL, CLIP_PROCESSOR, CLIP_DEVICE, CLIP_BACKEND

    try:
        from optimum.intel.openvino import OVModelForZeroShotImageClassification

        CLIP_DEVICE = "openvino"
        CLIP_BACKEND = f"OpenVINO {OPENVINO_DEVICE}"

        print(f"Loading CLIP model: {CLIP_MODEL_ID} with {CLIP_BACKEND}")
        CLIP_MODEL = OVModelForZeroShotImageClassification.from_pretrained(CLIP_MODEL_ID,export=True)

        return CLIP_MODEL, CLIP_PROCESSOR, CLIP_DEVICE, CLIP_BACKEND

    except Exception as exc:
        print(f"OpenVINO CLIP fallback failed. Using Torch CPU instead. Reason: {exc}")

    from transformers import CLIPModel

    CLIP_DEVICE = "cpu"
    CLIP_BACKEND = "Torch CPU"

    print(f"Loading CLIP model: {CLIP_MODEL_ID} on CPU")
    CLIP_MODEL = CLIPModel.from_pretrained(CLIP_MODEL_ID)
    CLIP_MODEL.to(CLIP_DEVICE)
    CLIP_MODEL.eval()

    return CLIP_MODEL, CLIP_PROCESSOR, CLIP_DEVICE, CLIP_BACKEND

def get_cached_clip_text_inputs(texts: list[str], device: str) -> dict[str, torch.Tensor]:
    _, processor, _, _ = get_clip_model()
    key = tuple(texts)

    if key not in CLIP_TEXT_INPUT_CACHE:
        text_inputs = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)

        CLIP_TEXT_INPUT_CACHE[key] = {
                                    name: value.detach().cpu()
                                    for name, value in text_inputs.items()
                                    }

    cached = CLIP_TEXT_INPUT_CACHE[key]

    if device in {"cuda", "cpu"}:
        return {
                name: value.to(device)
                for name, value in cached.items()
                }

    return cached

def get_clip_image_inputs(image: Any, device: str) -> dict[str, torch.Tensor]:
    _, processor, _, _ = get_clip_model()

    image_inputs = processor(
                            images=image,
                            return_tensors="pt",
                            )

    if device in {"cuda", "cpu"}:
        return {
                name: value.to(device)
                for name, value in image_inputs.items()
                }

    return image_inputs

def clip_image_text_scores(image: Any, texts: list[str]) -> np.ndarray:
    model, _, device, _ = get_clip_model()

    text_inputs = get_cached_clip_text_inputs(texts, device)
    image_inputs = get_clip_image_inputs(image, device)
    inputs = {**text_inputs, **image_inputs}

    if device in {"cuda", "cpu"}:
        with torch.inference_mode():
            logits = model(**inputs).logits_per_image[0].detach().cpu().numpy()
    else:
        output = model(**inputs)
        logits = output.logits_per_image[0]

        if hasattr(logits, "detach"):
            logits = logits.detach().cpu().numpy()
        else:
            logits = np.asarray(logits)

    return softmax_numpy(logits)

def evaluate_product_image(image, selected_category: str, customer_preference: str):
    _, _, _, clip_backend = get_clip_model()
    if image is None:
        return "Please upload a JPG, JPEG or PNG product image first.", pd.DataFrame()
    if not customer_preference or not customer_preference.strip():
        customer_preference = "a product that matches the customer preference"

    try:
        from PIL import Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("RGB")

        category_prompts = [f"a product photo of a {c.lower()}" for c in IMAGE_PRODUCT_CATEGORIES]
        category_scores = clip_image_text_scores(image, category_prompts)
        category_table = pd.DataFrame({"Check": IMAGE_PRODUCT_CATEGORIES, "Score": np.round(category_scores, 4)}).sort_values("Score", ascending=False)
        category_table["Score %"] = (category_table["Score"] * 100).round(1).astype(str) + " %"
        category_table["Type"] = "Visual category check"

        selected_category_score = float(category_scores[IMAGE_PRODUCT_CATEGORIES.index(selected_category)])
        preference_prompts = [
                                f"a product photo that matches this customer preference: {customer_preference}",
                                f"a product photo that partially matches this customer preference: {customer_preference}",
                                f"a product photo that does not match this customer preference: {customer_preference}",
                                f"a product photo from the selected category: {selected_category}",
                                f"a product photo from a different category than: {selected_category}",
                            ]
        preference_scores = clip_image_text_scores(image, preference_prompts)
        match_score, partial_score, mismatch_score, sel_cat_prompt, diff_cat_prompt = (float(s) for s in preference_scores)

        combined_score = (0.75 * match_score + 0.15 * selected_category_score + 0.10 * sel_cat_prompt - 0.20 * mismatch_score)
        combined_score = float(np.clip(combined_score, 0.0, 1.0))
        
        decision = "Good fit" if combined_score >= 0.51 else "Partial fit" if combined_score >= 0.33 else "Weak fit"

        summary = (
                    f"CLIP backend: {clip_backend}\n\n"
                    f"Selected category: {selected_category}\n"
                    f"Customer preference: {customer_preference}\n\n"
                    f"Selected category match: {selected_category_score * 100:.1f} %\n"
                    f"Customer preference match: {match_score * 100:.1f} %\n"
                    f"Partial match signal: {partial_score * 100:.1f} %\n"
                    f"Mismatch signal: {mismatch_score * 100:.1f} %\n\n"
                    f"Combined relevance score: {combined_score * 100:.1f} %\n"
                    f"Decision: {decision}\n\n"
                    "Audience interpretation:\n"
                    "The model compares the uploaded image with text descriptions. It does not know the product name "
                    "directly. It estimates whether the visual content looks similar to the selected category and preference."
                    )

        prompt_table = pd.DataFrame({
                                    "Check": ["Matches customer preference", "Partially matches customer preference",
                                                "Does not match customer preference", "Looks like selected category",
                                                "Looks like different category"],
                                    "Score": [round(match_score, 4), round(partial_score, 4), round(mismatch_score, 4),round(sel_cat_prompt, 4), round(diff_cat_prompt, 4)],
                                    "Type": ["Preference check", "Preference check", "Preference check","Category check", "Category check"],
                                    })
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
                    "Most common fix: pip install transformers pillow\n"
                    "If you are offline, run it once with internet access so the model is cached locally."
                    )
        return message, pd.DataFrame()

# ============================================================
# Startup: load catalog + train genre model
# ============================================================
print("Loading movie catalog (content-based recommender, no training)")
MOVIE = load_movie_catalog()

print("Product image relevance uses CLIP zero-shot model. No startup training required.")

print("Training/loading Playlist models (XGBoost vs Torch MLP)")
PLAYLIST_MODEL = train_playlist_bundle(optuna_trials_xgb=OPTUNA_TRIALS_XGB, optuna_trials_mlp=OPTUNA_TRIALS_MLP)

# ============================================================
# Dashboard
# ============================================================
def build_dashboard_markdown() -> str:
    total_rows = MOVIE["rows"] + PLAYLIST_MODEL["rows"]
    return f"""
<div class="hero-card">

# Machine Learning demo dashboard

Application to show ML behavior on real-life examples:
**content-based recommender**, **XGBoost**, **Torch MLP neural network**, and **CLIP image-text model**.

</div>

<div class="metric-grid">

<div class="metric-card">
<h3>3</h3>
<div>Model families</div>
<div class="small-note">Recommender + XGBoost/Torch MLP + CLIP</div>
</div>

<div class="metric-card">
<h3>{total_rows:,}</h3>
<div>Rows in datasets</div>
<div class="small-note">IMDb catalog + Spotify tracks</div>
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
    clip_row = pd.DataFrame([{
                            "Task": "Product image relevance", "Model": f"CLIP zero-shot ({CLIP_MODEL_ID})",
                            "Accuracy": np.nan, "F1": np.nan, "Macro F1": np.nan, "Train sec incl. Optuna": 0.0,
                            "Source": "Uploaded image + customer preference text",
                            "Best params": "No training. Zero-shot image-text similarity.",
                            }])
    return pd.concat([clip_row, PLAYLIST_MODEL["metrics"]], ignore_index=True)

def get_training_sources() -> pd.DataFrame:
    return pd.DataFrame([
                        {"Dataset": "Movie recommender", "Source": MOVIE["source"], "Rows": MOVIE["rows"],
                        "Purpose": "Content-based movie recommendations from the IMDb catalog"},
                        {"Dataset": "Product image relevance", "Source": f"Uploaded image + {CLIP_MODEL_ID}",
                        "Rows": "N/A", "Purpose": "Image-category and customer preference matching"},
                        {"Dataset": "Playlist creator", "Source": PLAYLIST_MODEL["source"], "Rows": PLAYLIST_MODEL["rows"],
                        "Purpose": "Build a playlist from audio features (XGBoost vs Torch MLP)"},
                        ])
# ============================================================
# Gradio UI
# ============================================================
with gr.Blocks(title="ML Demo - Recommender + XGBoost vs Torch MLP + CLIP") as demo:
    gr.Markdown(build_dashboard_markdown())

    with gr.Tabs():
        with gr.Tab("Dashboard"):
            gr.Markdown("## Overview\n\nThis dashboard shows loaded datasets, model metrics and training sources.")
            gr.Dataframe(value=get_all_metrics(), label="Model metrics", interactive=False)
            gr.Dataframe(value=get_training_sources(), label="Datasets", interactive=False)

        with gr.Tab("1. Movie recommender"):
            gr.Markdown(
                        "## Movie recommendation\n\n"
                        "Set your taste, mood and constraints. The app scores every movie in "
                        f"`{MOVIE['source']}` ({MOVIE['rows']} titles) and returns the best matches."
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
                    genre = gr.Dropdown(GENRES, value="Sci-Fi", label="Preferred genre")
                    rating = gr.Slider(1, 9.3, value=7.5, step=0.1, label="Minimum IMDb rating")
                    age = gr.Slider(1, 105, value=30, step=1, label="Maximum movie age (years)")
                    length = gr.Slider(45, 240, value=125, step=1, label="Preferred length (minutes)")
                    mood = gr.Dropdown(MOODS, value="Mind-bending", label="Current mood")
                    top_n = gr.Slider(1, 20, value=8, step=1, label="How many recommendations")
                    movie_btn = gr.Button("Recommend", variant="primary")
            movie_out = gr.Textbox(label="Top recommendation", lines=12, elem_classes=["prediction-box"])
            movie_table = gr.Dataframe(label="Recommended movies", interactive=False, elem_classes=["dataframe-table"])
            movie_btn.click(
                            recommend_movies,
                            [action, comedy, drama, scifi, romance, documentary, genre, rating, age, length, mood, top_n],
                            [movie_out, movie_table],
                            )

        with gr.Tab("2. Product image relevance"):
            gr.Markdown(
                        "## Product image relevance\n\n"
                        "Upload a product image, choose the expected category and describe the customer preference.\n\n"
                        "This tab uses a CLIP image-text model, easy to present to non-technical users because the "
                        "audience can directly see the product image and the resulting match score."
                        )
            with gr.Row():
                with gr.Column(scale=1):
                    product_image = gr.Image(label="Upload product image", type="pil", sources=["upload"], image_mode="RGB")
                    selected_product_category = gr.Dropdown(IMAGE_PRODUCT_CATEGORIES, value="Tech gadget", label="Expected product category")
                    customer_preference = gr.Textbox(
                        label="Customer preference", lines=4,value="black wireless headphones suitable for office calls and travel",placeholder="Example: lightweight running shoes for gym training",)
                    product_image_btn = gr.Button("Evaluate image relevance", variant="primary")
                with gr.Column(scale=1):
                    product_image_out = gr.Textbox(label="Image relevance result", lines=14, elem_classes=["prediction-box"])
                    product_image_scores = gr.Dataframe(label="Model scores", interactive=False, elem_classes=["dataframe-table"])
            product_image_btn.click(evaluate_product_image,[product_image, selected_product_category, customer_preference],[product_image_out, product_image_scores])

        with gr.Tab("3. Playlist creator"):
            gr.Markdown(
                        "## Playlist creator (XGBoost vs Torch MLP)\n\n"
                        "Pick a target vibe and shape the audio profile. Both models score every track in "
                        f"`{PLAYLIST_MODEL['source']}` ({PLAYLIST_MODEL['rows']:,} tracks) and each builds its own playlist, "
                        "so you can compare how the two algorithms read the same taste."
                        )
            _pl_genres = PLAYLIST_MODEL["labels"]
            _pl_default = "pop" if "pop" in _pl_genres else _pl_genres[0]
            with gr.Row():
                with gr.Column():
                    pl_genre = gr.Dropdown(_pl_genres, value=_pl_default, label="Target vibe (genre)")
                    pl_energy = gr.Slider(0, 1, value=0.7, step=0.1, label="Energy")
                    pl_dance = gr.Slider(0, 1, value=0.6, step=0.1, label="Danceability")
                    pl_valence = gr.Slider(0, 1, value=0.5, step=0.1, label="Positivity (valence)")
                with gr.Column():
                    pl_acoustic = gr.Slider(0, 1, value=0.2, step=0.1, label="Acousticness")
                    pl_tempo = gr.Slider(60, 200, value=120, step=5, label="Preferred tempo (BPM)")
                    pl_pop = gr.Slider(0, 100, value=30, step=1, label="Minimum popularity")
                    pl_n = gr.Slider(10, 20, value=15, step=1, label="Playlist length (tracks)")
                    pl_btn = gr.Button("Create playlist", variant="primary")
            pl_out = gr.Textbox(label="Playlist summary", lines=9, elem_classes=["prediction-box"])
            with gr.Row():
                pl_xgb_table = gr.Dataframe(label="XGBoost playlist", interactive=False, elem_classes=["dataframe-table"])
                pl_mlp_table = gr.Dataframe(label="Torch MLP playlist", interactive=False, elem_classes=["dataframe-table"])
            pl_metrics = gr.Dataframe(label="Playlist model metrics", interactive=False)
            pl_btn.click(
                        create_playlist,
                        [pl_genre, pl_energy, pl_dance, pl_valence, pl_acoustic, pl_tempo, pl_pop, pl_n],
                        [pl_out, pl_xgb_table, pl_mlp_table, pl_metrics],
                        )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", inbrowser=True, share=False, css=CUSTOM_CSS)    