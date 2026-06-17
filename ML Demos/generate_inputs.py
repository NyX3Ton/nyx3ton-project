from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

BASE_DIR = Path(__file__).resolve().parent
INPUTS_DIR = BASE_DIR / "Inputs"
INPUTS_DIR.mkdir(parents=True, exist_ok=True)

GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Fantasy", "Romance", "Documentary"]
MOODS = ["Relax", "Funny", "Emotional", "Adrenaline", "Mind-bending"]
PRODUCT_CATEGORIES = ["Gaming", "Fitness", "Travel", "Finance", "Tech", "Food"]
DEVICES = ["Mobile", "Desktop", "Tablet"]


def generate_movie_recommendation(rows: int = 5000) -> pd.DataFrame:
    data = []
    for _ in range(rows):
        prefs = rng.uniform(0, 1, 6)
        genre_id = int(rng.integers(0, len(GENRES)))
        rating = float(np.clip(rng.normal(7.0, 1.1), 3.5, 9.8))
        movie_age = int(rng.integers(0, 45))
        length = int(np.clip(rng.normal(112, 22), 70, 190))
        mood_id = int(rng.integers(0, len(MOODS)))

        genre_match = [
            prefs[0],
            prefs[1],
            prefs[2],
            prefs[3],
            0.65 * prefs[3] + 0.35 * prefs[0],
            prefs[4],
            prefs[5],
        ][genre_id]

        mood_bonus = 0.12 if (
            (mood_id == 4 and genre_id in [3, 4])
            or (mood_id == 1 and genre_id == 1)
            or (mood_id == 3 and genre_id == 0)
        ) else 0.0

        score = (
            1.85 * genre_match
            + 0.23 * (rating - 6.5)
            - 0.006 * movie_age
            - 0.002 * max(length - 130, 0)
            + mood_bonus
            + rng.normal(0, 0.25)
        )

        data.append(
            {
                "user_action": round(float(prefs[0]), 3),
                "user_comedy": round(float(prefs[1]), 3),
                "user_drama": round(float(prefs[2]), 3),
                "user_scifi": round(float(prefs[3]), 3),
                "user_romance": round(float(prefs[4]), 3),
                "user_documentary": round(float(prefs[5]), 3),
                "movie_genre": GENRES[genre_id],
                "movie_rating": round(rating, 2),
                "movie_age_years": movie_age,
                "movie_length_min": length,
                "mood": MOODS[mood_id],
                "recommended": int(score > 0.95),
            }
        )
    return pd.DataFrame(data)


def generate_product_click(rows: int = 5000) -> pd.DataFrame:
    data = []
    for _ in range(rows):
        device = DEVICES[int(rng.integers(0, len(DEVICES)))]
        category = PRODUCT_CATEGORIES[int(rng.integers(0, len(PRODUCT_CATEGORIES)))]
        previous_clicks = int(rng.poisson(2))
        session_minutes = float(np.clip(rng.gamma(2.2, 3.0), 0.3, 45))
        visits_7d = int(rng.poisson(4))
        price_sensitivity = float(rng.uniform(0, 1))
        brand_interest = float(rng.uniform(0, 1))
        evening = int(rng.integers(0, 2))
        category_affinity = float(rng.uniform(0, 1))

        logit = (
            -1.55
            + 0.37 * previous_clicks
            + 0.075 * visits_7d
            + 1.15 * brand_interest
            + 0.95 * category_affinity
            + 0.025 * session_minutes
            + 0.12 * evening
            - 0.38 * price_sensitivity
        )
        probability = 1 / (1 + np.exp(-(logit + rng.normal(0, 0.42))))

        data.append(
            {
                "device": device,
                "product_category": category,
                "previous_clicks": previous_clicks,
                "session_minutes": round(session_minutes, 2),
                "visits_7d": visits_7d,
                "price_sensitivity": round(price_sensitivity, 3),
                "brand_interest": round(brand_interest, 3),
                "evening": evening,
                "category_affinity": round(category_affinity, 3),
                "clicked": int(rng.uniform(0, 1) < probability),
            }
        )
    return pd.DataFrame(data)


def generate_movie_genre(rows_per_genre: int = 600) -> pd.DataFrame:
    templates = {
        "Action": ["explosion chase mission fight escape", "soldier rescue battle danger", "revenge crime combat survival", "spy operation chase rooftop"],
        "Comedy": ["funny friends awkward jokes", "office misunderstanding humor", "clumsy wedding laugh", "family party silly chaos"],
        "Drama": ["family secret emotional conflict", "relationship struggle personal loss", "character difficult decision", "career pressure moral dilemma"],
        "Sci-Fi": ["space alien robot future", "starship artificial intelligence galaxy", "simulation futuristic society", "time travel scientific experiment"],
        "Fantasy": ["dragon magic kingdom quest", "wizard ancient prophecy forest", "mythical creature enchanted land", "chosen hero magical artifact"],
        "Romance": ["love couple relationship heart", "romantic story destiny", "breakup reunion passion", "unexpected date wedding feelings"],
        "Documentary": ["real story investigation nature", "true events science archive", "facts analysis environment", "historical evidence expert interview"],
    }
    filler = ["unexpected", "journey", "hidden", "modern", "classic", "intense", "slow", "dark", "bright", "mysterious", "personal", "dramatic"]

    data = []
    for genre, examples in templates.items():
        for _ in range(rows_per_genre):
            description = str(rng.choice(examples)) + " " + " ".join(rng.choice(filler, size=int(rng.integers(3, 7))))
            data.append({"description": description, "genre": genre})

    return pd.DataFrame(data).sample(frac=1, random_state=SEED).reset_index(drop=True)


def main() -> None:
    movie = generate_movie_recommendation(rows=5000)
    product = generate_product_click(rows=5000)
    genre = generate_movie_genre(rows_per_genre=600)

    movie.to_csv(INPUTS_DIR / "movie_recommendation_full.csv", index=False)
    product.to_csv(INPUTS_DIR / "product_click_full.csv", index=False)
    genre.to_csv(INPUTS_DIR / "movie_genre_full.csv", index=False)

    print("Generated full input CSV files:")
    print(f"- {INPUTS_DIR / 'movie_recommendation_full.csv'} ({len(movie)} rows)")
    print(f"- {INPUTS_DIR / 'product_click_full.csv'} ({len(product)} rows)")
    print(f"- {INPUTS_DIR / 'movie_genre_full.csv'} ({len(genre)} rows)")


if __name__ == "__main__":
    main()
