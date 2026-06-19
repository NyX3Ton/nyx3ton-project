# ML Demos — Classic vs. Deep Tabular Models + CLIP

An interactive [Gradio](https://www.gradio.app/) application that demonstrates how different machine-learning models behave on real-world data. It compares a gradient-boosted tree (**XGBoost**), a neural network (**Torch MLP**), and a parameter-efficient deep ensemble (**TabM**) side by side, and adds a zero-shot vision demo with **CLIP**. Hyperparameters are tuned automatically with **Optuna**, training uses the GPU when available and falls back to CPU otherwise, and each model family can be switched on or off at startup.

> Built as a hands-on way to show non-experts what model selection, tuning, and ensembling actually do — using movies, music, and product images instead of abstract benchmarks.

<!-- Optional: add a screenshot or GIF of the running app here -->
<!-- ![App overview](docs/screenshot.png) -->

## Features

The app launches a single browser UI with four tabs:

| Tab | What it does | Models |
| --- | --- | --- |
| **Dashboard** | Overview of loaded datasets, model metrics, and training sources | — |
| **1. Movie recommender** | Predicts how likely you are to enjoy a title from your taste sliders + mood, then ranks real IMDb movies | XGBoost · Torch MLP · TabM |
| **2. Product image relevance** | Upload a product image; CLIP scores how well it matches a chosen category and a free-text customer preference | CLIP (zero-shot) |
| **3. Playlist creator** | Pick a target "vibe" and audio profile; each model scores tracks and builds its own 10–20 song playlist so you can compare them | XGBoost · Torch MLP · TabM |

Across the trained tabs you also get:

- **Three-way model comparison** — every enabled model produces its own results table, side by side.
- **Soft-vote ensemble** — when two or more models are enabled, their probabilities are averaged and reported as an extra row in the metrics.
- **Automatic hyperparameter tuning** via Optuna (TPE sampler), with studies cached to SQLite so re-runs are fast.
- **GPU auto-detection** — uses CUDA for both PyTorch and XGBoost if present, otherwise CPU, with no code changes.
- **Per-model enable/disable** through environment flags.

## Tech stack

Python · Gradio · XGBoost · PyTorch · [TabM](https://github.com/yandex-research/tabm) · Optuna · scikit-learn · Transformers (CLIP) · pandas · NumPy. OpenVINO is an optional acceleration backend for CLIP.

## Project structure

```
ML Demos/
├── app.py                 # The full application (UI + models + training)
├── Inputs/                # Datasets live here (see "Datasets" below)
│   ├── IMDb_Genres_real_enriched.csv
│   └── spotify_data.csv
├── Cache/                 # Optuna study database (auto-created)
│   └── optuna_studies.sqlite3
├── requirements.txt       # Python dependencies (see below)
└── README.md
```

## Datasets

The large CSV files are **not** committed to the repository (see `.gitignore`). Place them in the `Inputs/` folder before running:

- **`IMDb_Genres_real_enriched.csv`** — the movie catalog used to surface real titles in the Movie recommender.
- **`spotify_data.csv`** — the [Spotify 1 Million Tracks dataset](https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks) (Kaggle), used by the Playlist creator. Download it from Kaggle and drop it into `Inputs/`.

The Movie recommender's *training labels* are generated synthetically at startup (a built-in deterministic generator), so no extra file is required. To train on your own labeled data instead, add `Inputs/movie_recommendation_full.csv` with columns: `user_action, user_comedy, user_drama, user_scifi, user_romance, user_documentary, movie_genre, movie_rating, movie_age_years, movie_length_min, mood, recommended`.

## Installation

Requires **Python 3.10+**. A virtual environment is recommended.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd "<your-repo>"

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

For **GPU acceleration**, install the CUDA build of PyTorch that matches your driver from the [official selector](https://pytorch.org/get-started/locally/) instead of the default CPU wheel, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

`requirements.txt`:

```
gradio
numpy
pandas
scikit-learn
xgboost
torch
optuna
transformers
pillow
tabm            # optional — enables the TabM model
# optimum[openvino]   # optional — OpenVINO backend for CLIP
```

> If `tabm` is not installed, the app prints a notice and automatically disables the TabM model; the other models still run.

## Usage

```bash
python app.py
```

The app trains/loads the models (first run runs Optuna tuning and caches it), then opens at `http://127.0.0.1:7860`.

## Configuration

Behavior is controlled with environment variables — no code edits needed.

| Variable | Default | Description |
| --- | --- | --- |
| `ENABLE_XGBOOST` | `1` | Train and show the XGBoost model |
| `ENABLE_MLP` | `1` | Train and show the Torch MLP model |
| `ENABLE_TABM` | `1` | Train and show the TabM model (auto-off if `tabm` is missing) |
| `OPTUNA_TRIALS_XGB` | `0` | Optuna trials for XGBoost (`0` = use built-in defaults, no tuning) |
| `OPTUNA_TRIALS_MLP` | `0` | Optuna trials for the Torch MLP |
| `OPTUNA_TRIALS_TABM` | `0` | Optuna trials for TabM |
| `TABM_K` | `16` | Number of TabM ensemble submodels |
| `TOP_GENRES_N` | `14` | Number of most-common Spotify genres to classify |
| `MAX_SPOTIFY_ROWS` | `1048000` | Cap on training rows for the playlist models |
| `PLAYLIST_CANDIDATES` | `10000` | Candidate pool size for building playlists |
| `CLIP_MODEL_ID` | `openai/clip-vit-base-patch32` | CLIP model used for image relevance |

Examples:

```bash
# Run only XGBoost (fastest startup)
ENABLE_MLP=0 ENABLE_TABM=0 python app.py

# Tune every model with 30 Optuna trials each
OPTUNA_TRIALS_XGB=30 OPTUNA_TRIALS_MLP=30 OPTUNA_TRIALS_TABM=30 python app.py
```

(On Windows PowerShell, set variables with `$env:ENABLE_MLP="0"` before running.)

## How it works

- **Tabular models** are trained as classifiers on engineered numeric features (audio features for music; taste + movie attributes for movies). Features are standardized on the training split only, and the scaler is reused for inference.
- **XGBoost** uses early stopping on a validation set; **Torch MLP** and **TabM** track validation macro-F1 each epoch and restore the best weights.
- **Recommendations/playlists** are built by scoring a candidate pool with each model's predicted probability and blending it with the user's chosen profile, then ranking the top results.
- **CLIP** runs zero-shot: it embeds the uploaded image and the candidate text labels and ranks them by similarity — no training required.

## Notes & limitations

- Predicting a fine-grained genre from a handful of audio features (or a "would-recommend" label) has a real accuracy ceiling — many classes overlap. The comparison is meant to be illustrative rather than state-of-the-art.
- First startup can take a while because it trains two model bundles (and tunes them if Optuna trials > 0). Studies are cached, so later runs are faster; disable models you don't need to speed things up.

## Acknowledgements

- **TabM** — Gorishniy et al., *"TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"* (ICLR 2025). [Paper](https://arxiv.org/abs/2410.24210) · [Code](https://github.com/yandex-research/tabm)
- **CLIP** — OpenAI, via Hugging Face Transformers.
- **Spotify 1 Million Tracks** dataset by Amitansh Joshi (Kaggle).

## License

<!-- Choose a license (e.g. MIT) and add a LICENSE file. Replace this section accordingly. -->
This project is released under the **MIT License** — see [`LICENSE`](LICENSE) for details.
