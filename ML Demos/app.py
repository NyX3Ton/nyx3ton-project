from __future__ import annotations

import ast, json, os, re, time, optuna, torch, datetime
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
OPTUNA_STORAGE = f"sqlite:///{(CACHE_DIR / 'optuna_studies.sqlite3').as_posix()}"

GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Fantasy", "Romance", "Documentary"]
MOODS = ["Relax", "Funny", "Emotional", "Adrenaline", "Mind-bending"]

IMAGE_PRODUCT_CATEGORIES = ["Tech gadget", "Gaming accessory", "Fitness product", "Travel gear","Food product", "Fashion item", "Home appliance", "Book","Musical instrument", "Sports equipment", "Office equipment", "Beauty product"]

OPTUNA_TRIALS_XGB = int(os.getenv("OPTUNA_TRIALS_XGB", "12"))
OPTUNA_TRIALS_MLP = int(os.getenv("OPTUNA_TRIALS_MLP", "8"))
MAX_GENRE_ROWS = int(os.getenv("MAX_GENRE_ROWS", "20000"))

CLIP_MODEL_ID = os.getenv("CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = None
CLIP_PROCESSOR = None

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
        return train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    except ValueError:
        return train_test_split(X, y, test_size=0.20, random_state=SEED)

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
    """Load the enriched IMDb CSV and pre-compute the numeric columns used for scoring."""
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
        keep = scored["_match"] > -1  # all
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
    def __init__(self, n_features: int, n_classes: int, hidden_1: int = 256, hidden_2: int = 128, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
                                nn.Linear(n_features, hidden_1), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(hidden_1, hidden_2), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(hidden_2, n_classes),
                                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def load_genre_dataframe() -> tuple[pd.DataFrame, str]:
    genre_path = find_existing_input([
                                        "movie_genre_full.csv", "movie_genre_full.xlsx",
                                        "movie_genre_sample.csv", "movie_genre_sample.xlsx",
                                        "train_data.txt", "train_data_solution.txt",
                                        ])

    if genre_path is None:
        if generate_movie_genre is None:
            raise FileNotFoundError("No genre dataset found and generate_inputs.generate_movie_genre is unavailable.")
        df = generate_movie_genre(rows_per_genre=600).copy()
        df["title"] = [f"Demo Movie {i + 1}" for i in range(len(df))]
        return df[["title", "description", "genre"]], "generated fallback"

    if genre_path.suffix.lower() == ".txt":
        df = pd.read_csv(genre_path, sep=r"\s*:::\s*", engine="python", names=["id", "title", "genre", "description"])
        df = df[["title", "description", "genre"]].dropna().copy()
        if len(df) > MAX_GENRE_ROWS:
            df = df.sample(n=MAX_GENRE_ROWS, random_state=SEED)
        return df, genre_path.name

    df = read_table_file(genre_path)
    title_cols = ["title", "Title", "movie_title", "Movie Title", "Movie name", "Movie Name", "name", "Name"]
    desc_cols = ["description", "Description", "plot", "Plot", "overview", "Overview", "summary", "Summary"]
    genre_cols = ["genre", "Genre", "genres", "Genres"]

    title_col = next((c for c in title_cols if c in df.columns), None)
    desc_col = next((c for c in desc_cols if c in df.columns), None)
    genre_col = next((c for c in genre_cols if c in df.columns), None)
    if desc_col is None or genre_col is None:
        raise ValueError(f"Genre file {genre_path.name} must contain description and genre columns.")

    result = df[[desc_col, genre_col]].dropna().copy()
    result.columns = ["description", "genre"]
    result["title"] = (df.loc[result.index, title_col].astype(str)
                        if title_col else [f"Movie Example {i + 1}" for i in range(len(result))])
    result = result[["title", "description", "genre"]]
    if len(result) > MAX_GENRE_ROWS:
        result = result.sample(n=MAX_GENRE_ROWS, random_state=SEED)
    return result, genre_path.name


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
        best = {"hidden_1": 256, "hidden_2": 128, "dropout": 0.10, "lr": 0.005,
                "weight_decay": 1e-4, "batch_size": 64, "epochs": 45}
    else:
        def objective(trial: optuna.Trial) -> float:
            torch.manual_seed(SEED)
            model = MultiMLP(
                                X.shape[1], n_classes,
                                trial.suggest_int("hidden_1", 32, 192, step=32),
                                trial.suggest_int("hidden_2", 16, 96, step=16),
                                trial.suggest_float("dropout", 0.0, 0.40),
                            )
            model = run_training(
                                model, X_train, y_train,
                                trial.suggest_float("lr", 1e-4, 3e-2, log=True),
                                trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
                                int(trial.suggest_categorical("batch_size", [32, 64, 128])),
                                trial.suggest_int("epochs", 15, 100),
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


def train_genre_bundle(optuna_trials_xgb: int = OPTUNA_TRIALS_XGB, optuna_trials_mlp: int = OPTUNA_TRIALS_MLP) -> dict:
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
        best_xgb_params = {"n_estimators": 160, 
                            "max_depth": 5, 
                            "learning_rate": 0.005, 
                            "subsample": 0.9,
                            "colsample_bytree": 0.9, 
                            "min_child_weight": 1.0, 
                            "gamma": 0.0,
                            "reg_alpha": 1e-8, 
                            "reg_lambda": 1.0
                            }
        xgb_time = 0.0
    else:
        def xgb_objective(trial: optuna.Trial) -> float:
            params = {
                        "n_estimators": trial.suggest_int("n_estimators", 100, 900, step=100),
                        "max_depth": trial.suggest_int("max_depth", 4, 10),
                        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
                        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
                        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
                        "gamma": trial.suggest_float("gamma", 0.0, 4.0),
                        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                        "eval_metric": "mlogloss", "random_state": SEED, "n_jobs": 8, "tree_method": "hist",
                        }
            model = XGBClassifier(**params)
            model.fit(X_xgb_train, y_xgb_train)
            return float(f1_score(y_xgb_val, model.predict(X_xgb_val), average="macro"))

        start_xgb = time.perf_counter()
        xgb_study = get_or_create_study(study_name=f"{study_prefix}_xgboost")
        ensure_study_trials(xgb_study, xgb_objective, optuna_trials_xgb)
        best_xgb_params = dict(xgb_study.best_params)
        xgb_time = time.perf_counter() - start_xgb

    final_xgb_params = {**best_xgb_params, "eval_metric": "mlogloss", "random_state": SEED,"n_jobs": 8, "tree_method": "hist"}
    start_final_xgb = time.perf_counter()
    xgb = XGBClassifier(**final_xgb_params)
    xgb.fit(X_train, y_train)
    xgb_time += time.perf_counter() - start_final_xgb

    X_train_dense = to_dense_float32(X_train)
    X_test_dense = to_dense_float32(X_test)
    mlp, mlp_time, best_mlp_params = train_multi_mlp(X_train_dense, y_train,optuna_trials=optuna_trials_mlp,study_name=f"{study_prefix}_torch_mlp")
    xgb_pred = xgb.predict(X_test)
    with torch.no_grad():
        mlp_pred = np.argmax(torch.softmax(mlp(torch.tensor(X_test_dense, dtype=torch.float32)), dim=1).numpy(), axis=1)

    metrics = pd.DataFrame([
                            {"Task": "Genre classifier", "Model": "XGBoost",
                                "Accuracy": round(float(accuracy_score(y_test, xgb_pred)), 4),
                                "Macro F1": round(float(f1_score(y_test, xgb_pred, average="macro")), 4),
                                "Train sec incl. Optuna": round(float(xgb_time), 3), "Source": source,
                                "Best params": json.dumps(best_xgb_params, ensure_ascii=False)},
                            {"Task": "Genre classifier", "Model": "Torch MLP",
                                "Accuracy": round(float(accuracy_score(y_test, mlp_pred)), 4),
                                "Macro F1": round(float(f1_score(y_test, mlp_pred, average="macro")), 4),
                                "Train sec incl. Optuna": round(float(mlp_time), 3), "Source": source,
                                "Best params": json.dumps(best_mlp_params, ensure_ascii=False)},
                            ])

    return {"xgb": xgb, "mlp": mlp, "vectorizer": vectorizer, "labels": labels,"metrics": metrics, "source": source, "rows": len(df)}


def predict_genre(description):
    X = GENRE_MODEL["vectorizer"].transform([description or ""])
    X_dense = to_dense_float32(X)
    xgb_probs = GENRE_MODEL["xgb"].predict_proba(X)[0]
    with torch.no_grad():
        mlp_probs = torch.softmax(GENRE_MODEL["mlp"](torch.tensor(X_dense, dtype=torch.float32)), dim=1).numpy()[0]
    labels = GENRE_MODEL["labels"]
    table = pd.DataFrame({
                            "Genre": labels,
                            "XGBoost probability": np.round(xgb_probs, 4),
                            "Torch MLP probability": np.round(mlp_probs, 4),
                            "Difference": np.round(np.abs(xgb_probs - mlp_probs), 4),
                        }).sort_values("XGBoost probability", ascending=False)
    xgb_label = labels[int(np.argmax(xgb_probs))]
    mlp_label = labels[int(np.argmax(mlp_probs))]
    summary = (
                f"XGBoost predicts: {xgb_label} ({np.max(xgb_probs) * 100:.1f} %)\n"
                f"Torch MLP predicts: {mlp_label} ({np.max(mlp_probs) * 100:.1f} %)\n\n"
                )
    summary += ("Models agree. This is an example of a consistent signal." if xgb_label == mlp_label
                else "Models disagree. Different algorithms can read different signals in the same text.")
    return summary, table, GENRE_MODEL["metrics"]

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
        logits = model(**inputs).logits_per_image[0].detach().cpu().numpy()
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

        combined_score = 0.45 * selected_category_score + 0.45 * match_score + 0.10 * sel_cat_prompt
        decision = "Good fit" if combined_score >= 0.65 else "Partial fit" if combined_score >= 0.45 else "Weak fit"

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

print("Training/loading Genre model")
GENRE_MODEL = train_genre_bundle(optuna_trials_xgb=OPTUNA_TRIALS_XGB, optuna_trials_mlp=OPTUNA_TRIALS_MLP)

# ============================================================
# Dashboard
# ============================================================

def build_dashboard_markdown() -> str:
    total_rows = MOVIE["rows"] + GENRE_MODEL["rows"]
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
<div class="small-note">IMDb catalog + genre training data</div>
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
    return pd.concat([clip_row, GENRE_MODEL["metrics"]], ignore_index=True)

def get_training_sources() -> pd.DataFrame:
    return pd.DataFrame([
                        {"Dataset": "Movie recommender", "Source": MOVIE["source"], "Rows": MOVIE["rows"],
                        "Purpose": "Content-based movie recommendations from the IMDb catalog"},
                        {"Dataset": "Product image relevance", "Source": f"Uploaded image + {CLIP_MODEL_ID}",
                        "Rows": "N/A", "Purpose": "Image-category and customer preference matching"},
                        {"Dataset": "Genre classifier", "Source": GENRE_MODEL["source"], "Rows": GENRE_MODEL["rows"],
                        "Purpose": "Genre prediction from description"},
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

        with gr.Tab("3. Genre classifier"):
            gr.Markdown("## Genre classification\n\nThe model receives a movie description and predicts the most likely genre.")
            description = gr.Textbox(value="A starship crew discovers an alien signal on a distant planet.", lines=4, label="Movie description")
            genre_btn = gr.Button("Classify", variant="primary")
            genre_out = gr.Textbox(label="Prediction", lines=6, elem_classes=["prediction-box"])
            genre_probs = gr.Dataframe(label="Class probabilities", interactive=False)
            genre_metrics = gr.Dataframe(label="Genre classifier metrics", interactive=False)
            genre_btn.click(predict_genre, [description], [genre_out, genre_probs, genre_metrics])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", inbrowser=True, share=False, css=CUSTOM_CSS)