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

# ============================================================
# 1. Global config
# ============================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
try:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

# Auto-detect GPU; fall back to CPU if CUDA is unavailable.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
XGB_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE} (XGBoost device: {XGB_DEVICE})")

optuna.logging.set_verbosity(optuna.logging.WARNING)

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")

BASE_DIR = Path(__file__).resolve().parent
INPUTS_DIR = BASE_DIR / "Inputs"
INPUTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = BASE_DIR / "Cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OPTUNA_STORAGE = f"sqlite:///{(CACHE_DIR / 'optuna_studies.sqlite3').as_posix()}"
OPTUNA_DASHBOARD = os.getenv("OPTUNA_DASHBOARD", "1") == "1"
OPTUNA_DASHBOARD_PORT = int(os.getenv("OPTUNA_DASHBOARD_PORT", "8080"))

GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Fantasy", "Romance", "Documentary"]
MOODS = ["Relax", "Funny", "Emotional", "Adrenaline", "Mind-bending"]

IMAGE_PRODUCT_CATEGORIES = ["Tech gadget", "Gaming accessory", "Fitness product", "Travel gear","Food product", "Fashion item", "Home appliance", "Book","Musical instrument", "Sports equipment", "Office equipment", "Beauty product"]

OPTUNA_TRIALS_XGB = int(os.getenv("OPTUNA_TRIALS_XGB", "0"))
OPTUNA_TRIALS_MLP = int(os.getenv("OPTUNA_TRIALS_MLP", "0"))
OPTUNA_TRIALS_TABM = int(os.getenv("OPTUNA_TRIALS_TABM", "0"))

# Enable/disable each model family at startup (disabled models are not trained and not shown).
ENABLE_XGBOOST = os.getenv("ENABLE_XGBOOST", "1") == "1"
ENABLE_MLP = os.getenv("ENABLE_MLP", "1") == "1"
ENABLE_TABM = os.getenv("ENABLE_TABM", "1") == "1"
TABM_K = int(os.getenv("TABM_K", "16"))  # number of TabM ensemble submodels

# Best hyperparameters found via Optuna, reused as the OPTUNA_TRIALS_* == 0 defaults (per task, per model).
PLAYLIST_XGB_DEFAULTS = {"n_estimators": 800, "max_depth": 11, "learning_rate": 0.04,"subsample": 0.7, "colsample_bytree": 1,"min_child_weight": 5.3, "gamma": 0.55, "reg_alpha": 0.2, "reg_lambda": 0.054}
MOVIE_XGB_DEFAULTS = {"n_estimators": 900, "max_depth": 6, "learning_rate": 0.0244,"subsample": 0.6, "colsample_bytree": 0.81,"min_child_weight": 6.2, "gamma": 1.1,"reg_alpha": 0.06, "reg_lambda": 0.00155}
PLAYLIST_MLP_DEFAULTS = {"hidden_1": 320, "hidden_2": 256, "dropout": 0.0265,"lr": 0.002, "weight_decay": 0.000015,"batch_size": 256, "epochs": 216}
MOVIE_MLP_DEFAULTS = {"hidden_1": 512, "hidden_2": 96, "dropout": 0.18,"lr": 0.001, "weight_decay": 0.01,"batch_size": 128, "epochs": 125}
PLAYLIST_TABM_DEFAULTS = {"n_blocks": 3, "d_block": 512, "dropout": 0.0172,"lr": 0.0035, "weight_decay": 1.97e-05,"batch_size": 128, "epochs": 87}
MOVIE_TABM_DEFAULTS = {"n_blocks": 4, "d_block": 256, "dropout": 0.146,"lr": 0.00042, "weight_decay": 0.0002,"batch_size": 128, "epochs": 59}

try:
    from tabm import TabM
    _TABM_AVAILABLE = True
except Exception:
    TabM = None
    _TABM_AVAILABLE = False
    if ENABLE_TABM:
        print("tabm package not installed; TabM disabled. Install with: pip install tabm")
    ENABLE_TABM = False
#MAX_GENRE_ROWS = int(os.getenv("MAX_GENRE_ROWS", "20000"))

# Playlist creator (Spotify) settings
SPOTIFY_CSV_NAMES = ["spotify_data.csv", "spotify_1million_tracks.csv", "spotify_tracks.csv"]
#SPOTIFY_AUDIO_FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness","instrumentalness", "liveness", "valence", "tempo"]
SPOTIFY_AUDIO_FEATURES = ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]

# Extra (non-audio) signal the models can use; sliders/taste-match still use SPOTIFY_AUDIO_FEATURES only.
SPOTIFY_EXTRA_FEATURES = ["popularity", "year", "duration_ms", "time_signature"]
SPOTIFY_MODEL_FEATURES = SPOTIFY_AUDIO_FEATURES + SPOTIFY_EXTRA_FEATURES

TOP_GENRES_N = int(os.getenv("TOP_GENRES_N", "14"))
MAX_SPOTIFY_ROWS = int(os.getenv("MAX_SPOTIFY_ROWS", "1048000"))
PLAYLIST_CANDIDATES = int(os.getenv("PLAYLIST_CANDIDATES", "10000"))

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

MOVIE_TRAIN_CSV = ["movie_recommendation_full.csv", "movie_recommendation_sample.csv"]
MOVIE_NUM_FEATURES = ["user_action", "user_comedy", "user_drama", "user_scifi", "user_romance","user_documentary", "movie_rating", "movie_age_years", "movie_length_min"]

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
# 2. Small helpers
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

def launch_optuna_dashboard():
    if not OPTUNA_DASHBOARD:
        return
    try:
        from optuna_dashboard import run_server
    except ImportError:
        print("optuna-dashboard not installed; skipping. Install: pip install optuna-dashboard")
        return
    import threading
    def _run():
        try:
            run_server(OPTUNA_STORAGE, host="127.0.0.1", port=OPTUNA_DASHBOARD_PORT)
        except Exception as exc:
            print(f"Optuna dashboard failed to start: {exc}")
    threading.Thread(target=_run, daemon=True).start()
    print(f"Optuna dashboard at http://127.0.0.1:{OPTUNA_DASHBOARD_PORT}")

# ============================================================
# 3. Persistent Optuna helpers (used by the genre classifier)
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
# 4. Movie recommender (content-based, no training)
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
    top_n = int(top_n)
    cand = MOVIE_MODEL["candidates"]
    proba_fns = MOVIE_MODEL["proba_fns"]

    notes = []
    work = cand
    if genre:
        gsel = work[work["movie_genre"] == genre]
        if len(gsel) >= top_n:
            work = gsel
        else:
            notes.append(f"Few '{genre}' titles in the catalog, so the genre filter was relaxed.")
    filtered = work[(work["movie_rating"] >= float(min_rating)) & (work["movie_age_years"] <= float(max_age))]
    if len(filtered) < top_n:
        notes.append("Rating/age filters relaxed to find enough titles.")
        filtered = work
    cand_f = filtered.reset_index(drop=True)

    inf = pd.DataFrame({
                        "user_action": float(action), "user_comedy": float(comedy), "user_drama": float(drama),
                        "user_scifi": float(scifi), "user_romance": float(romance), "user_documentary": float(documentary),
                        "movie_rating": cand_f["movie_rating"].to_numpy(),
                        "movie_age_years": cand_f["movie_age_years"].to_numpy(),
                        "movie_length_min": cand_f["movie_length_min"].to_numpy(),
                        "movie_genre": cand_f["movie_genre"].to_numpy(),
                        "mood": str(mood),
                        })
    X = movie_feature_matrix(inf, MOVIE_MODEL["scaler"])
    length_part = np.clip(1.0 - np.abs(cand_f["movie_length_min"].to_numpy() - float(length)) / 120.0, 0.0, 1.0)

    def build(p):
        p = np.asarray(p, dtype=np.float64)
        score = 0.85 * p + 0.15 * length_part
        order = np.argsort(score)[::-1][:top_n]
        c = cand_f.iloc[order]
        table = pd.DataFrame({
                            "Title": c["movie_title"].to_numpy(),
                            "Year": c["release_year"].astype(int).to_numpy(),
                            "Genre": c["movie_genre"].to_numpy(),
                            "IMDb": np.round(c["imdb_rating_10"].to_numpy(), 1),
                            "Runtime (min)": c["runtime_minutes"].astype(int).to_numpy(),
                            "Director": c["director_clean"].to_numpy(),
                            "P(recommend) %": np.round(p[order] * 100, 1),
                            "Match %": np.round(score[order] * 100, 1),
                            }).reset_index(drop=True)
        conf = float(np.mean(p[order])) if len(order) else 0.0
        return table, conf

    model_order = ["XGBoost", "Torch MLP", "TabM"]
    empty = pd.DataFrame()
    results = {n: (build(proba_fns[n](X)[:, 1]) if n in proba_fns else (empty, 0.0)) for n in model_order}

    enabled = [n for n in model_order if n in proba_fns]
    lines = [f"Preferred genre: {genre}   |   Mood: {mood}", f"Recommendations per model: {top_n}", ""]
    for n in enabled:
        lines.append(f"{n} mean P(recommend) for its picks: {results[n][1] * 100:.1f} %")
    if not enabled:
        lines.append("No models are enabled. Set ENABLE_XGBOOST / ENABLE_MLP / ENABLE_TABM.")
    lines += ["", f"Scored {len(cand_f):,} candidate titles from {MOVIE['source']}."]
    summary = "\n".join(lines)
    if notes:
        summary += "\n\nNote: " + " ".join(notes)
    return summary, results["XGBoost"][0], results["Torch MLP"][0], results["TabM"][0], MOVIE_MODEL["metrics"]

# ============================================================
# 5. Genre classifier (XGBoost vs Torch MLP)
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
    needed = ["artist_name", "track_name", "genre", *SPOTIFY_MODEL_FEATURES]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Spotify file {path.name} is missing columns: {missing}")

    for col in SPOTIFY_MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["genre", *SPOTIFY_MODEL_FEATURES]).copy()
    df["genre"] = df["genre"].astype(str)
    df["artist_name"] = df["artist_name"].fillna("").astype(str)
    df["track_name"] = df["track_name"].fillna("").astype(str)

    dedupe_key = "track_id" if "track_id" in df.columns else ["artist_name", "track_name"]
    before = len(df)
    df = df.drop_duplicates(subset=dedupe_key, keep="first").copy()
    if len(df) < before:
        print(f"Removed {before - len(df)} duplicate track(s); kept first occurrence (its genre).")

    top_genres = df["genre"].value_counts().head(TOP_GENRES_N).index.tolist()
    df = df[df["genre"].isin(top_genres)].copy()

    if len(df) > MAX_SPOTIFY_ROWS:
        per_genre = max(1, MAX_SPOTIFY_ROWS // len(top_genres))
        df = df.groupby("genre", group_keys=False).apply(
                        lambda g: g.sample(n=min(len(g), per_genre), random_state=SEED)
                        )
    return df.reset_index(drop=True), path.name

def train_multi_mlp(X, y, optuna_trials: int = OPTUNA_TRIALS_MLP, study_name: str = "multi_mlp", default_params: dict | None = None) -> tuple[nn.Module, float, dict]:
    start_total = time.perf_counter()
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    n_classes = int(np.max(y)) + 1
    X_train, X_val, y_train, y_val = split_data(X, y)

    def run_training(model, X_fit, y_fit, lr, weight_decay, batch_size, epochs, trial=None, X_eval=None, y_eval=None, patience=15):
        model = model.to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(torch.tensor(X_fit, dtype=torch.float32), torch.tensor(y_fit, dtype=torch.long)),batch_size=batch_size, shuffle=True)
        best_f1, best_state, no_improve = -1.0, None, 0
        for epoch in range(epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss_fn(model(xb), yb).backward()
                optimizer.step()
            if X_eval is not None and y_eval is not None:
                model.eval()
                with torch.no_grad():
                    pred = torch.argmax(model(torch.tensor(X_eval, dtype=torch.float32).to(DEVICE)), dim=1).cpu().numpy()
                val_f1 = float(f1_score(y_eval, pred, average="macro"))
                if val_f1 > best_f1:
                    best_f1, no_improve = val_f1, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    no_improve += 1
                if trial is not None:
                    trial.report(val_f1, step=epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                if no_improve >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model

    if optuna_trials <= 0:
        best = dict(default_params) if default_params else {"hidden_1": 256, "hidden_2": 128, "dropout": 0.1, "lr": 0.005, "weight_decay": 1e-4, "batch_size": 256, "epochs": 200}
    else:
        def objective(trial: optuna.Trial) -> float:
            torch.manual_seed(SEED)
            model = MultiMLP(
                                X.shape[1], n_classes,
                                trial.suggest_int("hidden_1", 128, 2024, step=128),
                                trial.suggest_int("hidden_2", 64, 256, step=64),
                                trial.suggest_float("dropout", 0.02, 0.2),
                            )
            model = run_training(
                                model, X_train, y_train,
                                trial.suggest_float("lr", 1e-3, 3e-2, log=True),
                                trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                                int(trial.suggest_categorical("batch_size", [128, 256])),
                                trial.suggest_int("epochs", 50, 250),
                                trial, X_val, y_val,
                                )
            with torch.no_grad():
                pred = torch.argmax(model(torch.tensor(X_val, dtype=torch.float32).to(DEVICE)), dim=1).cpu().numpy()
            return float(f1_score(y_val, pred, average="macro"))

        study = get_or_create_study(study_name=study_name)
        ensure_study_trials(study, objective, optuna_trials)
        best = dict(study.best_params)

    torch.manual_seed(SEED)
    final_model = MultiMLP(X.shape[1], n_classes, int(best["hidden_1"]), int(best["hidden_2"]), float(best["dropout"]))
    final_model = run_training(final_model, X_train, y_train, float(best["lr"]), float(best["weight_decay"]), int(best["batch_size"]), int(best["epochs"]), X_eval=X_val, y_eval=y_val)
    return final_model, time.perf_counter() - start_total, best

def train_tabm(X, y, optuna_trials: int = OPTUNA_TRIALS_TABM, study_name: str = "tabm", default_params: dict | None = None) -> tuple[object, float, dict]:
    start_total = time.perf_counter()
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    n_classes = int(np.max(y)) + 1
    n_features = X.shape[1]
    X_train, X_val, y_train, y_val = split_data(X, y)

    def make_model(n_blocks, d_block, dropout):
        return TabM.make(n_num_features=n_features, d_out=n_classes,n_blocks=int(n_blocks), d_block=int(d_block),dropout=float(dropout), k=TABM_K).to(DEVICE)

    def run_training(model, X_fit, y_fit, lr, weight_decay, batch_size, epochs, trial=None, X_eval=None, y_eval=None, patience=15):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(torch.tensor(X_fit, dtype=torch.float32), torch.tensor(y_fit, dtype=torch.long)), batch_size=batch_size, shuffle=True)
        best_f1, best_state, no_improve = -1.0, None, 0
        for epoch in range(epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                out = model(xb)                                  # (B, k, C); k independent submodels
                k = out.shape[1]
                loss = loss_fn(out.reshape(-1, out.shape[-1]), yb.repeat_interleave(k))
                loss.backward()
                optimizer.step()
            if X_eval is not None and y_eval is not None:
                model.eval()
                with torch.no_grad():
                    out = model(torch.tensor(X_eval, dtype=torch.float32).to(DEVICE))
                    proba = torch.softmax(out, dim=-1).mean(dim=1).cpu().numpy()
                val_f1 = float(f1_score(y_eval, np.argmax(proba, axis=1), average="macro"))
                if val_f1 > best_f1:
                    best_f1, no_improve = val_f1, 0
                    best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
                else:
                    no_improve += 1
                if trial is not None:
                    trial.report(val_f1, step=epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                if no_improve >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model

    if optuna_trials <= 0:
        best = dict(default_params) if default_params else {"n_blocks": 2, "d_block": 256, "dropout": 0.1, "lr": 0.002, "weight_decay": 3e-4, "batch_size": 256, "epochs": 150}
    else:
        def objective(trial: optuna.Trial) -> float:
            torch.manual_seed(SEED)
            model = make_model(trial.suggest_int("n_blocks", 3, 5),
                                trial.suggest_int("d_block", 128, 512, step=128),
                                trial.suggest_float("dropout", 0.001, 0.20))
            model = run_training(model, X_train, y_train,
                                    trial.suggest_float("lr", 1e-4, 5e-3, log=True),
                                    trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
                                    int(trial.suggest_categorical("batch_size", [128, 256])),
                                    trial.suggest_int("epochs", 50, 250),
                                    trial, X_val, y_val)
            with torch.no_grad():
                out = model(torch.tensor(X_val, dtype=torch.float32).to(DEVICE))
                proba = torch.softmax(out, dim=-1).mean(dim=1).cpu().numpy()
            return float(f1_score(y_val, np.argmax(proba, axis=1), average="macro"))

        study = get_or_create_study(study_name=study_name)
        ensure_study_trials(study, objective, optuna_trials)
        best = dict(study.best_params)

    torch.manual_seed(SEED)
    final_model = make_model(best["n_blocks"], best["d_block"], best["dropout"])
    final_model = run_training(final_model, X_train, y_train, float(best["lr"]), float(best["weight_decay"]),
                                int(best["batch_size"]), int(best["epochs"]), X_eval=X_val, y_eval=y_val)
    return final_model, time.perf_counter() - start_total, best

def tabm_predict_proba(model, X) -> np.ndarray:
    with torch.no_grad():
        out = model(torch.tensor(np.asarray(X, dtype=np.float32)).to(DEVICE))
        return torch.softmax(out, dim=-1).mean(dim=1).cpu().numpy()

def train_playlist_bundle(optuna_trials_xgb: int = OPTUNA_TRIALS_XGB, optuna_trials_mlp: int = OPTUNA_TRIALS_MLP,
                            optuna_trials_tabm: int = OPTUNA_TRIALS_TABM) -> dict:
    df, source = load_spotify_dataframe()
    labels = sorted(df["genre"].unique())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    y = df["genre"].map(label_to_id).to_numpy()

    X_raw = df[SPOTIFY_MODEL_FEATURES].to_numpy(dtype=np.float32)
    X_raw_train, X_raw_test, y_train, y_test = split_data(X_raw, y)
    scaler = StandardScaler().fit(X_raw_train)
    X_train = scaler.transform(X_raw_train).astype(np.float32)
    X_test = scaler.transform(X_raw_test).astype(np.float32)
    X_xgb_train, X_xgb_val, y_xgb_train, y_xgb_val = split_data(X_train, y_train)
    study_prefix = safe_name(f"playlist_{source}_{X_train.shape[1]}feat_{len(labels)}genres")

    proba_fns: dict = {}      # name -> callable(feats_np) -> (N, C) probabilities (base models only)
    test_proba: dict = {}     # name -> probabilities on X_test
    metric_rows: list = []

    def _row(model_name, pred, secs, params):
        return {"Task": "Playlist creator", "Model": model_name,
                "Accuracy": round(float(accuracy_score(y_test, pred)), 4),
                "Macro F1": round(float(f1_score(y_test, pred, average="macro")), 4),
                "Train sec incl. Optuna": round(float(secs), 3), "Source": source,
                "Best params": params if isinstance(params, str) else json.dumps(params, ensure_ascii=False)}

    if ENABLE_XGBOOST:
        if optuna_trials_xgb <= 0:
            best_xgb_params = dict(PLAYLIST_XGB_DEFAULTS)
            xgb_time = 0.0
        else:
            def xgb_objective(trial: optuna.Trial) -> float:
                params = {
                            "n_estimators": trial.suggest_int("n_estimators", 600, 1000, step=100),
                            "max_depth": trial.suggest_int("max_depth", 6, 11),
                            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
                            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                            "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 10.0),
                            "gamma": trial.suggest_float("gamma", 0.5, 2.0),
                            "reg_alpha": trial.suggest_float("reg_alpha", 0.05, 1.0, log=True),
                            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 0.1, log=True),
                            "eval_metric": "mlogloss", "early_stopping_rounds": 50,
                            "random_state": SEED, "n_jobs": 8, "device": XGB_DEVICE,
                            }
                model = XGBClassifier(**params)
                model.fit(X_xgb_train, y_xgb_train, eval_set=[(X_xgb_val, y_xgb_val)], verbose=False)
                return float(f1_score(y_xgb_val, model.predict(X_xgb_val), average="macro"))

            start_xgb = time.perf_counter()
            xgb_study = get_or_create_study(study_name=f"{study_prefix}_xgboost")
            ensure_study_trials(xgb_study, xgb_objective, optuna_trials_xgb)
            best_xgb_params = dict(xgb_study.best_params)
            xgb_time = time.perf_counter() - start_xgb

        final_xgb_params = {**best_xgb_params, "eval_metric": "mlogloss", "early_stopping_rounds": 50,
                            "random_state": SEED, "n_jobs": 8, "device": XGB_DEVICE}
        start_final_xgb = time.perf_counter()
        xgb = XGBClassifier(**final_xgb_params)
        xgb.fit(X_xgb_train, y_xgb_train, eval_set=[(X_xgb_val, y_xgb_val)], verbose=False)
        xgb_time += time.perf_counter() - start_final_xgb
        proba_fns["XGBoost"] = lambda f, _m=xgb: _m.predict_proba(f)
        test_proba["XGBoost"] = xgb.predict_proba(X_test)
        metric_rows.append(_row("XGBoost", np.argmax(test_proba["XGBoost"], axis=1), xgb_time, best_xgb_params))

    if ENABLE_MLP:
        mlp, mlp_time, best_mlp_params = train_multi_mlp(X_train, y_train, optuna_trials=optuna_trials_mlp,
                                                            study_name=f"{study_prefix}_torch_mlp", default_params=PLAYLIST_MLP_DEFAULTS)
        def _mlp_proba(f, _m=mlp):
            with torch.no_grad():
                return torch.softmax(_m(torch.tensor(np.asarray(f, dtype=np.float32)).to(DEVICE)), dim=1).cpu().numpy()
        proba_fns["Torch MLP"] = _mlp_proba
        test_proba["Torch MLP"] = _mlp_proba(X_test)
        metric_rows.append(_row("Torch MLP", np.argmax(test_proba["Torch MLP"], axis=1), mlp_time, best_mlp_params))

    if ENABLE_TABM:
        tabm_model, tabm_time, best_tabm_params = train_tabm(X_train, y_train, optuna_trials=optuna_trials_tabm,
                                                                study_name=f"{study_prefix}_tabm", default_params=PLAYLIST_TABM_DEFAULTS)
        proba_fns["TabM"] = lambda f, _m=tabm_model: tabm_predict_proba(_m, f)
        test_proba["TabM"] = tabm_predict_proba(tabm_model, X_test)
        metric_rows.append(_row("TabM", np.argmax(test_proba["TabM"], axis=1), tabm_time, best_tabm_params))

    base_fns = dict(proba_fns)
    if len(test_proba) >= 2:
        ens_test = np.mean(list(test_proba.values()), axis=0)
        metric_rows.append(_row("Ensemble (soft vote)", np.argmax(ens_test, axis=1),
                                0.0, f"Average of {', '.join(test_proba.keys())} probabilities"))

    metrics = pd.DataFrame(metric_rows) if metric_rows else pd.DataFrame(
        [{"Task": "Playlist creator", "Model": "None enabled", "Accuracy": np.nan, "Macro F1": np.nan,
            "Train sec incl. Optuna": 0.0, "Source": source, "Best params": "All models disabled."}])

    catalog = df.sample(n=min(len(df), PLAYLIST_CANDIDATES), random_state=SEED).reset_index(drop=True)
    catalog_scaled = scaler.transform(catalog[SPOTIFY_MODEL_FEATURES].to_numpy(dtype=np.float32)).astype(np.float32)

    return {"proba_fns": base_fns, "scaler": scaler, "labels": labels, "label_to_id": label_to_id,
            "catalog": catalog, "catalog_scaled": catalog_scaled, "metrics": metrics,
            "source": source, "rows": len(df)}

def create_playlist(target_genre, energy, danceability, valence, acousticness, tempo, min_popularity, n_tracks):
    n_tracks = int(n_tracks)
    catalog = PLAYLIST_MODEL["catalog"]
    scaled = PLAYLIST_MODEL["catalog_scaled"]
    proba_fns = PLAYLIST_MODEL["proba_fns"]
    target_id = PLAYLIST_MODEL["label_to_id"].get(target_genre, 0)

    notes = []
    mask = (catalog["popularity"] >= float(min_popularity)).to_numpy()
    if mask.sum() < n_tracks:
        notes.append(f"Few tracks above popularity {int(min_popularity)}, so that filter was relaxed.")
        mask = np.ones(len(catalog), dtype=bool)
    idx = np.where(mask)[0]
    feats = scaled[idx]
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
    taste = np.clip(1.0 - dist, 0.0, 1.0).astype(np.float64)

    def build(model_p):
        model_p = np.asarray(model_p, dtype=np.float64)
        scores = 0.6 * model_p + 0.4 * taste
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

    model_order = ["XGBoost", "Torch MLP", "TabM"]
    empty = pd.DataFrame()
    results = {}
    for name in model_order:
        if name in proba_fns:
            results[name] = build(proba_fns[name](feats)[:, target_id])
        else:
            results[name] = (empty, set(), 0.0)

    enabled = [n for n in model_order if n in proba_fns]
    lines = [f"Target vibe: {target_genre}", f"Playlist length: {n_tracks} tracks per model", ""]
    for n in enabled:
        lines.append(f"{n} mean confidence in '{target_genre}': {results[n][2] * 100:.1f} %")
    if len(enabled) >= 2:
        common = set.intersection(*[results[n][1] for n in enabled])
        lines += ["", f"Tracks shared by all enabled models: {len(common)}/{n_tracks}",
                    ("Models largely agree on this vibe." if len(common) >= n_tracks / 2
                    else "Models pick noticeably different tracks for this vibe.")]
    elif not enabled:
        lines.append("No models are enabled. Set ENABLE_XGBOOST / ENABLE_MLP / ENABLE_TABM.")
    lines += ["", f"Candidate pool: {len(sub):,} tracks from {PLAYLIST_MODEL['source']}."]
    summary = "\n".join(lines)
    if notes:
        summary += "\n\nNote: " + " ".join(notes)
    return summary, results["XGBoost"][0], results["Torch MLP"][0], results["TabM"][0], PLAYLIST_MODEL["metrics"]

# ============================================================
# 6. CLIP image relevance
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
        CLIP_MODEL.to(CLIP_DEVICE)  # type: ignore[arg-type]
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
    CLIP_MODEL.to(CLIP_DEVICE)  # type: ignore[arg-type]
    CLIP_MODEL.eval()

    return CLIP_MODEL, CLIP_PROCESSOR, CLIP_DEVICE, CLIP_BACKEND

def get_cached_clip_text_inputs(texts: list[str], device: str | None) -> dict[str, torch.Tensor]:
    _, processor, _, _ = get_clip_model()
    key = tuple(texts)

    if key not in CLIP_TEXT_INPUT_CACHE:
        text_inputs = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)  # type: ignore[call-arg]

        CLIP_TEXT_INPUT_CACHE[key] = {
                                    name: value.detach().cpu()
                                    for name, value in text_inputs.items()
                                    }

    cached = CLIP_TEXT_INPUT_CACHE[key]

    if device in {"cuda", "cpu"}:
        return {name: value.to(device) for name, value in cached.items()}

    return cached

def get_clip_image_inputs(image: Any, device: str | None) -> dict[str, torch.Tensor]:
    _, processor, _, _ = get_clip_model()

    image_inputs = processor(images=image, return_tensors="pt")  # type: ignore[call-arg]

    if device in {"cuda", "cpu"}:
        return {name: value.to(device) for name, value in image_inputs.items()}

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
# 6b. Movie recommender (trained: XGBoost / Torch MLP / TabM)
# ============================================================
def generate_movie_training_data(rows: int = 12000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n_g, n_m = len(GENRES), len(MOODS)
    data = []
    for _ in range(rows):
        prefs = rng.uniform(0, 1, 6)
        gid = int(rng.integers(0, n_g))
        rating = float(np.clip(rng.normal(7.0, 1.1), 3.5, 9.8))
        age = int(rng.integers(0, 45))
        length = float(np.clip(rng.normal(112, 22), 70, 190))
        mid = int(rng.integers(0, n_m))
        genre_match = [prefs[0], prefs[1], prefs[2], prefs[3],
                       0.65 * prefs[3] + 0.35 * prefs[0], prefs[4], prefs[5]][gid]
        mood_bonus = 0.12 if ((mid == 4 and gid in (3, 4)) or (mid == 1 and gid == 1) or (mid == 3 and gid == 0)) else 0.0
        score = (1.85 * genre_match + 0.23 * (rating - 6.5) - 0.006 * age
                 - 0.002 * max(length - 130, 0) + mood_bonus + rng.normal(0, 0.25))
        data.append({
            "user_action": round(float(prefs[0]), 3), "user_comedy": round(float(prefs[1]), 3),
            "user_drama": round(float(prefs[2]), 3), "user_scifi": round(float(prefs[3]), 3),
            "user_romance": round(float(prefs[4]), 3), "user_documentary": round(float(prefs[5]), 3),
            "movie_genre": GENRES[gid], "movie_rating": round(rating, 2), "movie_age_years": age,
            "movie_length_min": int(length), "mood": MOODS[mid], "recommended": int(score > 0.95)})
    return pd.DataFrame(data)

def movie_feature_matrix(frame, scaler) -> np.ndarray:
    num = scaler.transform(frame[MOVIE_NUM_FEATURES].to_numpy(dtype=np.float32))
    genre_oh = np.stack([(frame["movie_genre"].astype(str).to_numpy() == g).astype(np.float32) for g in GENRES], axis=1)
    mood_oh = np.stack([(frame["mood"].astype(str).to_numpy() == m).astype(np.float32) for m in MOODS], axis=1)
    return np.concatenate([num, genre_oh, mood_oh], axis=1).astype(np.float32)

def map_catalog_to_movie_features(movie_df) -> pd.DataFrame:
    current_year = datetime.date.today().year
    genres_lc = movie_df["all_genres"].astype(str).str.lower().tolist()
    def first_genre(g):
        for cand in GENRES:
            if cand.lower() in g:
                return cand
        return "Drama"
    return pd.DataFrame({
        "movie_title": movie_df["movie_title"].to_numpy(),
        "release_year": movie_df["release_year"].astype(int).to_numpy(),
        "director_clean": movie_df["director_clean"].to_numpy(),
        "imdb_rating_10": movie_df["imdb_rating_10"].to_numpy(),
        "runtime_minutes": movie_df["runtime_minutes"].astype(int).to_numpy(),
        "movie_genre": [first_genre(g) for g in genres_lc],
        "movie_rating": movie_df["imdb_rating_10"].to_numpy(),
        "movie_age_years": (current_year - movie_df["release_year"]).clip(lower=0).to_numpy(),
        "movie_length_min": movie_df["runtime_minutes"].to_numpy(),
    })

def train_movie_bundle(optuna_trials_xgb: int = OPTUNA_TRIALS_XGB, optuna_trials_mlp: int = OPTUNA_TRIALS_MLP,
                        optuna_trials_tabm: int = OPTUNA_TRIALS_TABM) -> dict:
    path = find_existing_input(MOVIE_TRAIN_CSV)
    if path is not None:
        df = read_table_file(path)
        source = path.name
    else:
        df = generate_movie_training_data(rows=12000)
        source = "synthetic (built-in generator)"

    for c in MOVIE_NUM_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=MOVIE_NUM_FEATURES + ["movie_genre", "mood", "recommended"]).copy()
    df["movie_genre"] = df["movie_genre"].astype(str)
    df["mood"] = df["mood"].astype(str)
    y = df["recommended"].astype(int).to_numpy()

    idx = np.arange(len(df))
    idx_train, idx_test, y_train, y_test = split_data(idx, y)
    train_df, test_df = df.iloc[idx_train], df.iloc[idx_test]
    scaler = StandardScaler().fit(train_df[MOVIE_NUM_FEATURES].to_numpy(dtype=np.float32))
    X_train = movie_feature_matrix(train_df, scaler)
    X_test = movie_feature_matrix(test_df, scaler)
    X_xgb_train, X_xgb_val, y_xgb_train, y_xgb_val = split_data(X_train, y_train)
    study_prefix = safe_name(f"movie_{source}_{X_train.shape[1]}feat")

    proba_fns: dict = {}
    test_proba: dict = {}
    metric_rows: list = []
    def _row(name, pred, secs, params):
        return {"Task": "Movie recommender", "Model": name,
                "Accuracy": round(float(accuracy_score(y_test, pred)), 4),
                "Macro F1": round(float(f1_score(y_test, pred, average="macro")), 4),
                "Train sec incl. Optuna": round(float(secs), 3), "Source": source,
                "Best params": params if isinstance(params, str) else json.dumps(params, ensure_ascii=False)}

    if ENABLE_XGBOOST:
        if optuna_trials_xgb <= 0:
            best_xgb = dict(MOVIE_XGB_DEFAULTS)
            xgb_time = 0.0
        else:
            def xgb_obj(trial: optuna.Trial) -> float:
                params = {"n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
                            "max_depth": trial.suggest_int("max_depth", 3, 8),
                            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
                            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
                            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 1.0, log=True),
                            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 5.0, log=True),
                            "eval_metric": "logloss", "early_stopping_rounds": 40,
                            "random_state": SEED, "n_jobs": 8, "device": XGB_DEVICE}
                m = XGBClassifier(**params)
                m.fit(X_xgb_train, y_xgb_train, eval_set=[(X_xgb_val, y_xgb_val)], verbose=False)
                return float(f1_score(y_xgb_val, m.predict(X_xgb_val), average="macro"))
            t0 = time.perf_counter()
            st = get_or_create_study(study_name=f"{study_prefix}_xgboost")
            ensure_study_trials(st, xgb_obj, optuna_trials_xgb)
            best_xgb = dict(st.best_params); xgb_time = time.perf_counter() - t0
        fp = {**best_xgb, "eval_metric": "logloss", "early_stopping_rounds": 40,"random_state": SEED, "n_jobs": 8, "device": XGB_DEVICE}
        t0 = time.perf_counter()
        xgbm = XGBClassifier(**fp)
        xgbm.fit(X_xgb_train, y_xgb_train, eval_set=[(X_xgb_val, y_xgb_val)], verbose=False)
        xgb_time += time.perf_counter() - t0
        proba_fns["XGBoost"] = lambda f, _m=xgbm: _m.predict_proba(f)
        test_proba["XGBoost"] = xgbm.predict_proba(X_test)
        metric_rows.append(_row("XGBoost", np.argmax(test_proba["XGBoost"], axis=1), xgb_time, best_xgb))

    if ENABLE_MLP:
        mlpm, mlp_time, best_mlp = train_multi_mlp(X_train, y_train, optuna_trials=optuna_trials_mlp, study_name=f"{study_prefix}_torch_mlp", default_params=MOVIE_MLP_DEFAULTS)
        def _mlp_proba(f, _m=mlpm):
            with torch.no_grad():
                return torch.softmax(_m(torch.tensor(np.asarray(f, dtype=np.float32)).to(DEVICE)), dim=1).cpu().numpy()
        proba_fns["Torch MLP"] = _mlp_proba
        test_proba["Torch MLP"] = _mlp_proba(X_test)
        metric_rows.append(_row("Torch MLP", np.argmax(test_proba["Torch MLP"], axis=1), mlp_time, best_mlp))

    if ENABLE_TABM:
        tabmm, tabm_time, best_tabm = train_tabm(X_train, y_train, optuna_trials=optuna_trials_tabm, study_name=f"{study_prefix}_tabm", default_params=MOVIE_TABM_DEFAULTS)
        proba_fns["TabM"] = lambda f, _m=tabmm: tabm_predict_proba(_m, f)
        test_proba["TabM"] = tabm_predict_proba(tabmm, X_test)
        metric_rows.append(_row("TabM", np.argmax(test_proba["TabM"], axis=1), tabm_time, best_tabm))

    if len(test_proba) >= 2:
        ens = np.mean(list(test_proba.values()), axis=0)
        metric_rows.append(_row("Ensemble (soft vote)", np.argmax(ens, axis=1), 0.0,
                                f"Average of {', '.join(test_proba.keys())} probabilities"))

    metrics = pd.DataFrame(metric_rows) if metric_rows else pd.DataFrame(
        [{"Task": "Movie recommender", "Model": "None enabled", "Accuracy": np.nan, "Macro F1": np.nan,
            "Train sec incl. Optuna": 0.0, "Source": source, "Best params": "All models disabled."}])

    candidates = map_catalog_to_movie_features(MOVIE["df"])
    return {"proba_fns": proba_fns, "scaler": scaler, "candidates": candidates,
            "metrics": metrics, "source": source, "rows": len(df)}

# ============================================================
# 7. Startup: load catalog + train genre model
# ============================================================
print("Loading movie catalog")
MOVIE = load_movie_catalog()

print("Training/loading Movie recommender models (XGBoost / Torch MLP / TabM, per ENABLE_* flags)")
MOVIE_MODEL = train_movie_bundle(optuna_trials_xgb=OPTUNA_TRIALS_XGB, optuna_trials_mlp=OPTUNA_TRIALS_MLP)

print("Product image relevance uses CLIP zero-shot model. No startup training required.")

print("Training/loading Playlist models (XGBoost / Torch MLP / TabM, per ENABLE_* flags)")
PLAYLIST_MODEL = train_playlist_bundle(optuna_trials_xgb=OPTUNA_TRIALS_XGB, optuna_trials_mlp=OPTUNA_TRIALS_MLP)

# ============================================================
# 8. Dashboard
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
    return pd.concat([clip_row, MOVIE_MODEL["metrics"], PLAYLIST_MODEL["metrics"]], ignore_index=True)

def get_training_sources() -> pd.DataFrame:
    return pd.DataFrame([
                        {"Dataset": "Movie recommender", "Source": MOVIE_MODEL["source"], "Rows": MOVIE_MODEL["rows"],
                        "Purpose": "Predict 'recommended' from user taste + movie attributes (XGBoost / MLP / TabM)"},
                        {"Dataset": "Product image relevance", "Source": f"Uploaded image + {CLIP_MODEL_ID}",
                        "Rows": "N/A", "Purpose": "Image-category and customer preference matching"},
                        {"Dataset": "Playlist creator", "Source": PLAYLIST_MODEL["source"], "Rows": PLAYLIST_MODEL["rows"],
                        "Purpose": "Build a playlist from audio features (XGBoost vs Torch MLP)"},
                        ])
# ============================================================
# 9. Gradio UI
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
                        "## Movie recommendation (XGBoost vs Torch MLP vs TabM)\n\n"
                        f"Enabled models: **{', '.join(MOVIE_MODEL['proba_fns'].keys()) or 'none'}**. "
                        "Set your taste, mood and constraints. Each enabled model predicts how likely you are to "
                        f"like every title in `{MOVIE['source']}` ({MOVIE['rows']} movies) and returns its own top picks."
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
            movie_out = gr.Textbox(label="Recommendation summary", lines=8, elem_classes=["prediction-box"])
            with gr.Row():
                movie_xgb_table = gr.Dataframe(label="XGBoost picks", interactive=False, elem_classes=["dataframe-table"])
                movie_mlp_table = gr.Dataframe(label="Torch MLP picks", interactive=False, elem_classes=["dataframe-table"])
                movie_tabm_table = gr.Dataframe(label="TabM picks", interactive=False, elem_classes=["dataframe-table"])
            movie_metrics = gr.Dataframe(label="Movie model metrics", interactive=False)
            movie_btn.click(
                            recommend_movies,
                            [action, comedy, drama, scifi, romance, documentary, genre, rating, age, length, mood, top_n],
                            [movie_out, movie_xgb_table, movie_mlp_table, movie_tabm_table, movie_metrics],
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
                        "## Playlist creator (XGBoost vs Torch MLP vs TabM)\n\n"
                        f"Enabled models: **{', '.join(PLAYLIST_MODEL['proba_fns'].keys()) or 'none'}**. "
                        "Pick a target vibe and shape the audio profile. Each enabled model scores every track in "
                        f"`{PLAYLIST_MODEL['source']}` ({PLAYLIST_MODEL['rows']:,} tracks) and builds its own playlist, "
                        "so you can compare how the algorithms read the same taste."
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
                pl_tabm_table = gr.Dataframe(label="TabM playlist", interactive=False, elem_classes=["dataframe-table"])
            pl_metrics = gr.Dataframe(label="Playlist model metrics", interactive=False)
            pl_btn.click(
                        create_playlist,
                        [pl_genre, pl_energy, pl_dance, pl_valence, pl_acoustic, pl_tempo, pl_pop, pl_n],
                        [pl_out, pl_xgb_table, pl_mlp_table, pl_tabm_table, pl_metrics],
                        )

if __name__ == "__main__":
    launch_optuna_dashboard()
    demo.launch(server_name="127.0.0.1", inbrowser=True, share=False, css=CUSTOM_CSS)
