#%pip install --upgrade pip
#%pip install --upgrade optuna pandas numpy sqlalchemy pyodbc matplotlib seaborn plotly scikit-learn xgboost joblib gradio psycopg2-binary

# ---------------------------------------------------------
# 1. POSTGRESQL ENGINE & DATA LOAD
#       a. Database Helpers
# 2. DATA PREPROCESSING
# 3. TRAINING LOGIC
#       a. Optuna optimization
# 4. CATEGORIZATION BY RISK THRESHOLD, CLIPPING IRRELEVANT DATA
# 5. ANOMALY DETECTION + FEATURE IMPORTANCE
# 6. GRADIO FRONTEND
#       a. KPI Prep-work
#       b. KPI anomalies
#       c. Heatmap definition
#       d. Callbacks
#       e. UI Definition
#       f. TAB 1 - New employee simulator
#       g. TAB 2: Anomalies dashboard
# 7. MAIN FUNCTION
# ---------------------------------------------------------

import os, warnings, unicodedata, joblib, optuna, urllib, urllib.parse, gradio as gr
warnings.filterwarnings('ignore')

from pathlib import Path

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

INPUT_DIR = BASE_DIR / "Input"
MODEL_DIR = BASE_DIR / "model_save" / "model"

DATA_ENV_PATH = INPUT_DIR / "data.env"
MODEL_PATH = MODEL_DIR / "model_attrition.pkl"
ENCODER_PATH = MODEL_DIR / "encoder_attrition.pkl"
FEATURE_COLS_PATH = MODEL_DIR / "feature_cols.pkl"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("BASE_DIR =", BASE_DIR)
print("INPUT_DIR =", INPUT_DIR)
print("DATA_ENV_PATH =", DATA_ENV_PATH)
print("DATA_ENV exists =", DATA_ENV_PATH.exists())

import pandas as pd
import numpy as np
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.sql.elements import quoted_name

import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

# GPU / CPU Detection
def get_xgboost_device():
    try:
        X_test = np.array([[1, 2], [3, 4]])
        y_test = np.array([0, 1])
        test_model = XGBClassifier(device="cuda", n_estimators=1, tree_method="hist")
        test_model.fit(X_test, y_test)
        print("GPU na CUDA bol detekovany.")
        return "cuda"
    except Exception:
        print("GPU na CUDA nebol detekovany. Prepinam na CPU mode.")
        return "cpu"

ACTIVE_DEVICE = get_xgboost_device()

# 1. POSTGRESQL ENGINE & DATA LOAD

HR_TABLE = "HR_Synth_Data"
PREDICTIONS_TABLE = "Predictions_Attrition"
FEATURE_IMPORTANCE_TABLE = "Model_Feature_Importance"
FEATURE_IMPORTANCE_BY_ROLE_TABLE = "Model_Feature_Importance_By_Role"
ANOMALIES_TABLE = "Anomalies_Department"

def load_data_env(env_path: Path | None = None) -> None:
    if env_path is None:
        env_path = DATA_ENV_PATH

    print(f"Nahravam subor z adresara: {env_path}")
    print(f"Adresar existuje : {env_path.exists()}")

    if not env_path.exists():
        raise FileNotFoundError(f"Data.env subor nebol najdeny: {env_path}")

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
        raise RuntimeError(f"Chýbajú premenné prostredia: {missing}")

    host = os.getenv("PG_HOST")
    port = os.getenv("PG_PORT", "5432")
    database = os.getenv("PG_DATABASE")
    username = os.getenv("PG_USERNAME")
    password = urllib.parse.quote_plus(os.getenv("PG_PASSWORD"))

    return sa.create_engine(
        f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}",
        pool_pre_ping=True
    )

def load_hr_dataframe() -> pd.DataFrame:
    engine = postgre_engine()
    schema = postgre_schema()
    print("Nacitavam data z PostgreSQL")

    query = text(f'SELECT * FROM {qident(schema)}.{qident(HR_TABLE)}')
    df = pd.read_sql(query, engine)

    print(f"Bolo nacitanych {len(df)} riadkov.")
    return df

def save_to_sql(df: pd.DataFrame, table_name: str):
    engine = postgre_engine()
    schema = postgre_schema()
    print(f"Zapisujem tabulku: {table_name} ({len(df)} riadkov)...")
    try:
        df.to_sql(
            quoted_name(table_name, True),
            engine,
            schema=schema,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=1000
        )
        print("Zapis bol uspesny.")
    except Exception as e:
        print(f"Chyba pri zapise do PostgreSQL: {e}")

# a. Database Helpers

def get_next_employee_number(engine):
    schema = postgre_schema()
    try:
        query = text(
            f'SELECT MAX({qident("EmployeeNumber")}) AS max_id '
            f'FROM {qident(schema)}.{qident(HR_TABLE)}'
        )
        max_id = pd.read_sql(query, engine).iloc[0, 0]
        if pd.isna(max_id):
            return 1
        return int(max_id) + 1
    except Exception as e:
        print(f"Chyba pri získavaní ID: {e}")
        return int(np.random.randint(10000, 99999))

def add_new_employee_to_db(input_data: dict, df_template: pd.DataFrame):
    engine = postgre_engine()
    schema = postgre_schema()

    new_row_data = {}
    user_provided_cols = list(input_data.keys())

    for col in df_template.columns:
        if col in user_provided_cols:
            new_row_data[col] = input_data[col]
            continue

        if col == "EmployeeNumber":
            continue

        valid_values = df_template[col].dropna()
        if valid_values.empty:
            new_row_data[col] = 0
            continue

        if df_template[col].dtype.name in ["category", "object", "string"]:
            dist = valid_values.value_counts(normalize=True)
            chosen_val = np.random.choice(dist.index, p=dist.values)
            new_row_data[col] = chosen_val
        else:
            chosen_val = np.random.choice(valid_values.values)

            if isinstance(chosen_val, (np.integer, np.int64)):
                chosen_val = int(chosen_val)
            elif isinstance(chosen_val, (np.floating, np.float64)):
                chosen_val = float(chosen_val)

            new_row_data[col] = chosen_val

    new_row_df = pd.DataFrame([new_row_data])

    next_id = get_next_employee_number(engine)
    new_row_df["EmployeeNumber"] = next_id

    print(f"Pridávam zamestnanca ID {next_id} do PostgreSQL (Smart Sampling)")
    try:
        new_row_df.to_sql(
            quoted_name(HR_TABLE, True),
            engine,
            schema=schema,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000
        )

        return True, next_id, f"Zamestnanec {input_data.get('FullName', '')} (ID: {next_id}) bol úspešne uložený."
    except Exception as e:
        print(f"PostgreSQL Error: {e}")
        return False, None, f"Chyba pri ukladaní do DB: {str(e)}"

# 2. DATA PREPROCESSING

def get_drop_columns():
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
            "JobInvolvement"
            ]

def preprocess_data_for_training(df: pd.DataFrame):
    df_proc = df.copy()
    drop_cols = get_drop_columns()

    y = None
    if "Attrition" in df_proc.columns:
        df_proc["Attrition"] = df_proc["Attrition"].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
        y = df_proc["Attrition"]
        df_proc = df_proc.drop(columns=["Attrition"])

    df_proc = df_proc.drop(columns=[c for c in drop_cols if c in df_proc.columns], errors='ignore')

    encoders = {}
    cat_cols = df_proc.select_dtypes(include=["object", "category", "string"]).columns

    for col in cat_cols:
        le = LabelEncoder()
        df_proc[col] = le.fit_transform(df_proc[col].astype(str))
        encoders[col] = le

    return df_proc, y, encoders

def preprocess_data_for_inference(df: pd.DataFrame, encoders, feature_cols):
    df_proc = df.copy()

    drop_cols = get_drop_columns() + ["Attrition"]

    X = df_proc.drop(columns=[c for c in drop_cols if c in df_proc.columns], errors='ignore')

    for col, le in encoders.items():
        if col in X.columns:
            X[col] = X[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else 0)
            X[col] = X[col].fillna(0).astype(int)

    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    return X

# 3. TRAINING LOGIC

# a. Optuna optimization

def run_optuna_optimization(X_train, y_train, X_test, y_test, n_trials=100):
    def objective(trial: optuna.Trial) -> float:
        params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1600),
                    "max_depth": trial.suggest_int("max_depth", 5, 15),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "gamma": trial.suggest_float("gamma", 0, 5),
                    "random_state": 42,
                    "n_jobs": -1,
                    "tree_method": "hist",
                    "device": ACTIVE_DEVICE,
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "early_stopping_rounds": 25,
                }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test)
        return accuracy_score(y_test, preds)

    print(f"Spustam optimalizaciu hyperparametrov Optuna ({n_trials} pokusov) na {ACTIVE_DEVICE}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Najlepsia presnost: {study.best_value:.4f}")
    return study.best_params

def train_and_save_new_model(df_full: pd.DataFrame, n_trials=100, reuse_prev_params=True):
    print("\n Spustam trening modelu")

    X, y, encoders = preprocess_data_for_training(df_full)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    best_params = None
    if reuse_prev_params and os.path.exists(MODEL_PATH):
        try:
            print(f"Pokus o nacitanie parametrov predoslych treningov z ({MODEL_PATH})")
            old_model = joblib.load(MODEL_PATH)
            best_params = old_model.get_params()
            print("Parametre uspecne nacitane. Preskakujem Optunu.")

            keys_to_remove = ['missing', 'callbacks', 'monotone_constraints', 'interaction_constraints']
            for k in keys_to_remove:
                if k in best_params:
                    del best_params[k]
        except Exception as e:
            print(f"Predosle parametre neboli uspesne nacitane ({e}). Spustam Optunu.")
            best_params = None

    if best_params is None:
        best_params = run_optuna_optimization(X_train, y_train, X_test, y_test, n_trials)

    best_params.update({
                        "random_state": 42,
                        "n_jobs": -1,
                        "tree_method": "hist",
                        "device": ACTIVE_DEVICE,
                        "objective": "binary:logistic",
                        "eval_metric": "logloss",
                        "early_stopping_rounds": 25
                        })

    print(f"Trenujem finalny model ({ACTIVE_DEVICE})...")
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    acc = accuracy_score(y_test, final_model.predict(X_test))
    print(f"Finalna presnost: {acc:.3%}")

    print(f"Ukladam model do {MODEL_PATH}")
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    joblib.dump(feature_names, FEATURE_COLS_PATH)

    return final_model, encoders, feature_names, acc

# 4. CATEGORIZATION BY RISK THRESHOLD, CLIPPING IRRELEVANT DATA

def get_model_and_predictions(df_full: pd.DataFrame):
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH) and os.path.exists(FEATURE_COLS_PATH):
        print(f"\n Model najdeny ({MODEL_PATH}).")
        try:
            model = joblib.load(MODEL_PATH)
            encoders = joblib.load(ENCODER_PATH)
            feature_names = joblib.load(FEATURE_COLS_PATH)
            acc = 0.0
        except Exception as e:
            print(f"Chyba pri nacitani: {e}. Spustam trening modelu.")
            model, encoders, feature_names, acc = train_and_save_new_model(df_full, reuse_prev_params=False)
    else:
        print(f"\nModel nebol najdeny. Spustam trening")
        model, encoders, feature_names, acc = train_and_save_new_model(df_full, reuse_prev_params=False)

    print("Generujem predikcie pre aktualne data")
    X_current = preprocess_data_for_inference(df_full, encoders, feature_names)

    importance_df = pd.DataFrame({"Feature": feature_names,"Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)

    predictions_df = df_full.copy()
    all_probs = model.predict_proba(X_current)[:, 1]

    predictions_df["Risk_Label"] = pd.cut(all_probs, bins=[-0.1, 0.25, 0.50, 1.1], labels=["Low", "Medium", "High"])
    predictions_df["Risk_Label"] = predictions_df["Risk_Label"].astype(str)
    predictions_df["Attrition_Prob"] = (all_probs * 100).round(2)
    predictions_df["Attrition_Prob"] = predictions_df["Attrition_Prob"].astype(str) + "%"

    cols_to_drop = []
    predictions_df.drop(columns=[c for c in cols_to_drop if c in predictions_df.columns], inplace=True)

    cat_columns = predictions_df.select_dtypes(['category']).columns
    for col in cat_columns:
        predictions_df[col] = predictions_df[col].astype(str)

    cols_order = ["EmployeeNumber", "FullName", "Risk_Label", "Attrition_Prob", "Department", "JobRole"]
    remaining_cols = [c for c in predictions_df.columns if c not in cols_order]
    predictions_df = predictions_df[cols_order + remaining_cols]

    return model, encoders, feature_names, acc, importance_df, predictions_df

# 5. ANOMALY DETECTION + FEATURE IMPORTANCE

def detect_department_anomalies(df_full: pd.DataFrame, n_optuna_trials=50):
    print("\n 2. Detekcia Anomalii (Regression + Optuna)")

    df_agg = df_full.copy()
    df_agg["Attrition_Flag"] = df_agg["Attrition"].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    df_agg["OverTime_Flag"] = df_agg["OverTime"].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)

    group_cols = ["Department", "JobRole"]

    dept_stats = df_agg.groupby(group_cols).agg({"EmployeeNumber": "count", "Attrition_Flag": "mean", "OverTime_Flag": "mean", "WorkLifeBalance": lambda x: x.astype(int).mean()}).reset_index()

    dept_stats.rename(columns={"EmployeeNumber": "Headcount","Attrition_Flag": "Actual_Attrition_Rate","OverTime_Flag": "Avg_OverTime"}, inplace=True)

    dept_stats = dept_stats[dept_stats["Headcount"] > 2].copy()

    features = ["Avg_OverTime", "WorkLifeBalance"]
    X = dept_stats[features]
    y = dept_stats["Actual_Attrition_Rate"]

    if len(dept_stats) < 10:
        print("   [INFO] Malo dat pre Optunu. Default config.")
        best_params = {
                        "n_estimators": 600,
                        "max_depth": 12,
                        "learning_rate": 0.001,
                        "objective": "reg:squarederror",
                        "device": ACTIVE_DEVICE
                        }
    else:
        def objective_reg(trial):
            params = {
                        "n_estimators": trial.suggest_int("n_estimators", 100, 1200),
                        "max_depth": trial.suggest_int("max_depth", 5, 15),
                        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                        "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
                        "random_state": 42,
                        "n_jobs": -1,
                        "tree_method": "hist",
                        "device": ACTIVE_DEVICE,
                        "objective": "reg:squarederror"
                    }

            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            return np.sqrt(mean_squared_error(y_val, model.predict(X_val)))

        print(f"Spustam Optunu ({n_optuna_trials} pokusov)")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study_reg = optuna.create_study(direction="minimize")
        study_reg.optimize(objective_reg, n_trials=n_optuna_trials, show_progress_bar=True)
        best_params = study_reg.best_params

    final_params = best_params.copy()
    final_params.update({
                        "random_state": 42,
                        "n_jobs": -1,
                        "objective": "reg:squarederror",
                        "device": ACTIVE_DEVICE
                        })

    reg_model = XGBRegressor(**final_params)
    reg_model.fit(X, y)

    dept_stats["Predicted_Attrition_Rate"] = reg_model.predict(X)
    residuals = dept_stats["Actual_Attrition_Rate"] - dept_stats["Predicted_Attrition_Rate"]
    dept_stats["Anomaly_Score"] = residuals
    threshold = np.mean(residuals) + (2 * np.std(residuals))

    print(f"Anomaly Threshold (Mean + 2*Std): {threshold:.4f}")

    dept_stats["Is_Anomaly"] = dept_stats["Anomaly_Score"] > threshold
    dept_stats = dept_stats.sort_values(by="Anomaly_Score", ascending=False)

    cols_to_round = [
                    "Actual_Attrition_Rate",
                    "Avg_OverTime",
                    "WorkLifeBalance",
                    "Predicted_Attrition_Rate",
                    "Anomaly_Score"
                    ]

    for col in cols_to_round:
        if col in dept_stats.columns:
            dept_stats[col] = dept_stats[col].round(3)

    return dept_stats

def generate_feature_importance_by_role(df_full: pd.DataFrame):
    print("\n Generovanie Feature Importance podla JobRole")
    results = []
    roles = df_full["JobRole"].unique()

    for role in roles:
        df_subset = df_full[df_full["JobRole"] == role].copy()
        if len(df_subset) < 10:
            continue

        X, y, _ = preprocess_data_for_training(df_subset)
        if "JobRole" in X.columns:
            X = X.drop(columns=["JobRole"])
        if y.nunique() < 2:
            continue

        model = XGBClassifier(
            n_estimators=600,
            max_depth=12,
            random_state=42,
            device=ACTIVE_DEVICE,
            tree_method="hist",
            verbose=0
        )
        model.fit(X, y)

        imp_dict = dict(zip(X.columns, model.feature_importances_))
        imp_dict["JobRole"] = role
        results.append(imp_dict)

    if not results:
        return pd.DataFrame()

    df_imp = pd.DataFrame(results).fillna(0)

    num_cols = [c for c in df_imp.columns if c != "JobRole"]
    df_imp[num_cols] = df_imp[num_cols].astype(float).round(3)

    cols = ["JobRole"] + [c for c in df_imp.columns if c != "JobRole"]
    return df_imp[cols]

# 6. GRADIO FRONTEND

def run_gradio_app(
                    model,
                    encoders,
                    feature_names,
                    global_acc,
                    importance_df,
                    df_sample,
                    full_predictions_df,
                    feat_imp_by_role,
                    anomalies_df
):
    edu_map = {
                "1 - Základné": 1,
                "2 - Stredoškolské": 2,
                "3 - Bakalárske": 3,
                "4 - Magisterské/Inžinierske": 4,
                "5 - Doktorandské": 5
                }

    # a. KPI Prep-work
    predictions_safe = full_predictions_df.copy() if full_predictions_df is not None else pd.DataFrame()
    importance_safe = importance_df.copy() if importance_df is not None else pd.DataFrame()
    feat_imp_role_safe = feat_imp_by_role.copy() if feat_imp_by_role is not None else pd.DataFrame()
    anomalies_safe = anomalies_df.copy() if anomalies_df is not None else pd.DataFrame()

    total_employees = len(predictions_safe)

    if "Risk_Label" in predictions_safe.columns and not predictions_safe.empty:
        high_risk_count = int((predictions_safe["Risk_Label"].astype(str) == "High").sum())
        medium_risk_count = int((predictions_safe["Risk_Label"].astype(str) == "Medium").sum())
        low_risk_count = int((predictions_safe["Risk_Label"].astype(str) == "Low").sum())
    else:
        high_risk_count = 0
        medium_risk_count = 0
        low_risk_count = 0

    if "Attrition_Prob" in predictions_safe.columns and not predictions_safe.empty:
        prob_series = (
            predictions_safe["Attrition_Prob"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        prob_series = pd.to_numeric(prob_series, errors="coerce")
        avg_attrition_prob = float(prob_series.mean()) if prob_series.notna().any() else 0.0
    else:
        avg_attrition_prob = 0.0

    # b. KPI anomalies
    anom_kpi = pd.DataFrame([
        {"Metric": "Pocet anomalii", "Value": 0},
        {"Metric": "Max Anomalne skore", "Value": 0.0},
        {"Metric": "Priemerne Anomalne skore", "Value": 0.0},
    ])

    anom_table = pd.DataFrame(columns=[
                                        "Department",
                                        "JobRole",
                                        "Headcount",
                                        "Actual_Attrition_Rate",
                                        "Predicted_Attrition_Rate",
                                        "Anomaly_Score",
                                        "Is_Anomaly"
    ])

    anom_fig = px.bar(
        pd.DataFrame({"Info": ["Anomalie neboli detekovane"], "Value": [0]}),
        x="Value",
        y="Info",
        orientation="h",
        title="Anomalie podla oddeleni vs. pozicii"
    )
    anom_fig.update_layout(height=350)

    if not anomalies_safe.empty:
        anom_table = anomalies_safe.copy()

        if "Is_Anomaly" in anom_table.columns:
            anomaly_only = anom_table[anom_table["Is_Anomaly"] == True].copy()
        else:
            anomaly_only = anom_table.copy()

        if "Anomaly_Score" in anom_table.columns and not anom_table.empty:
            max_score = float(pd.to_numeric(anom_table["Anomaly_Score"], errors="coerce").max())
            mean_score = float(pd.to_numeric(anom_table["Anomaly_Score"], errors="coerce").mean())
        else:
            max_score = 0.0
            mean_score = 0.0

        anomaly_count = int(len(anomaly_only))

        anom_kpi = pd.DataFrame([
            {"Metric": "Pocet anomalii", "Value": anomaly_count},
            {"Metric": "Max Anomalne skore", "Value": round(max_score, 4)},
            {"Metric": "Priemerne Anomalne skore", "Value": round(mean_score, 4)},
        ])

        plot_source = anomaly_only.copy() if not anomaly_only.empty else anom_table.copy()

        if (
            "Department" in plot_source.columns
            and "JobRole" in plot_source.columns
            and "Anomaly_Score" in plot_source.columns
            and not plot_source.empty
        ):
            plot_source["Label"] = (
                plot_source["Department"].astype(str) + " | " + plot_source["JobRole"].astype(str)
            )
            plot_source["Anomaly_Score"] = pd.to_numeric(plot_source["Anomaly_Score"], errors="coerce").fillna(0)

            hover_cols = [
                c for c in [
                    "Headcount",
                    "Actual_Attrition_Rate",
                    "Predicted_Attrition_Rate"
                ]
                if c in plot_source.columns
            ]

            anom_fig = px.bar(
                plot_source.sort_values("Anomaly_Score", ascending=True).tail(25),
                x="Anomaly_Score",
                y="Label",
                orientation="h",
                title="Top anomalie podla oddelenia a pozicie",
                hover_data=hover_cols
            )
            anom_fig.update_layout(
                height=500,
                yaxis_title="",
                xaxis_title="Anomaly Score"
            )

    # c. Heatmap definition
    heatmap_fig = None
    if not feat_imp_role_safe.empty and "JobRole" in feat_imp_role_safe.columns:
        try:
            heatmap_df = feat_imp_role_safe.set_index("JobRole").copy()
            for col in heatmap_df.columns:
                heatmap_df[col] = pd.to_numeric(heatmap_df[col], errors="coerce").fillna(0)

            heatmap_fig = px.imshow(
                heatmap_df,
                labels=dict(x="Faktor", y="Pozicia", color="Dolezitost"),
                x=heatmap_df.columns,
                y=heatmap_df.index,
                aspect="auto",
                color_continuous_scale="Viridis"
            )
            heatmap_fig.update_layout(
                title="Mapa dolezitosti faktorov podla pozicie",
                height=700
            )
        except Exception:
            heatmap_fig = None

    # d. Callbacks
    def analyze_new_employee(
        fname, lname, gender, age, marital, edu_level_str, edu_field, dist_home,
        dept, role, num_comp, total_years,
        years_at_company, years_curr_role, years_since_promo, years_curr_manager,
        perf_rating, income, overtime, percent_hike, train_times, travel
    ):
        full_name = f"{fname} {lname}".strip()

        def strip_accents(s):
            return ''.join(
                c for c in unicodedata.normalize('NFD', str(s))
                if unicodedata.category(c) != 'Mn'
            )

        clean_fname = strip_accents(fname).lower().replace(" ", "")
        clean_lname = strip_accents(lname).lower().replace(" ", "")
        email = f"{clean_fname}.{clean_lname}@company.com"
        username = f"{clean_fname}.{clean_lname}"

        edu_level_int = edu_map[edu_level_str]

        input_dict = {
                        "Age": age,
                        "Gender": gender,
                        "MaritalStatus": marital,
                        "Education": edu_level_int,
                        "EducationField": edu_field,
                        "DistanceFromHome": dist_home,
                        "Department": dept,
                        "JobRole": role,
                        "NumCompaniesWorked": num_comp,
                        "TotalWorkingYears": total_years,
                        "YearsAtCompany": years_at_company,
                        "YearsInCurrentRole": years_curr_role,
                        "YearsSinceLastPromotion": years_since_promo,
                        "YearsWithCurrManager": years_curr_manager,
                        "PerformanceRating": perf_rating,
                        "MonthlyIncome": income,
                        "OverTime": overtime,
                        "PercentSalaryHike": percent_hike,
                        "TrainingTimesLastYear": train_times,
                        "BusinessTravel": travel
                    }

        full_data_for_db = input_dict.copy()
        full_data_for_db.update({"FirstName": fname, "LastName": lname, "FullName": full_name, "Email": email, "Username": username})

        pred_df = df_sample.iloc[0:1].copy()

        for k, v in input_dict.items():
            if k in pred_df.columns:
                pred_df[k] = v

        X_new = preprocess_data_for_inference(pred_df, encoders, feature_names)
        prob = float(model.predict_proba(X_new)[0, 1])

        if prob <= 0.25:
            decision = "NIZKE RIZIKO"
            color = "#16a34a"
            badge_bg = "#dcfce7"
        elif prob <= 0.33:
            decision = "MIERNE RIZIKO"
            color = "#d97706"
            badge_bg = "#fef3c7"
        else:
            decision = "VYSOKE RIZIKO"
            color = "#dc2626"
            badge_bg = "#fee2e2"

        result_md = f"""
### Vysledok pre: {full_name}
**Email:** {email}  
**Pravdepodobnost odchodu:** {prob:.2%}  

<span style="
display:inline-block;
padding:8px 14px;
border-radius:999px;
background:{badge_bg};
color:{color};
font-weight:700;
font-size:15px;
">
{decision}
</span>
"""

        role_specific_data = (
            feat_imp_role_safe[feat_imp_role_safe["JobRole"] == role].copy()
            if not feat_imp_role_safe.empty and "JobRole" in feat_imp_role_safe.columns
            else pd.DataFrame()
        )

        if not role_specific_data.empty:
            row = role_specific_data.iloc[0].drop("JobRole")
            local_imp_df = pd.DataFrame({
                "Feature": row.index,
                "Importance": row.values
            })
            local_imp_df["Importance"] = pd.to_numeric(local_imp_df["Importance"], errors="coerce").fillna(0)
            local_imp_df = local_imp_df.sort_values(by="Importance", ascending=False)
            title_txt = f"Top faktory pre pozíciu: {role}"
        else:
            local_imp_df = importance_safe.copy()
            if not local_imp_df.empty and "Importance" in local_imp_df.columns:
                local_imp_df["Importance"] = pd.to_numeric(local_imp_df["Importance"], errors="coerce").fillna(0)
                local_imp_df = local_imp_df.sort_values(by="Importance", ascending=False)
            title_txt = "Top faktory (globálny model)"

        fig = px.bar(
            local_imp_df.head(10), x="Importance", y="Feature", orientation="h", title=title_txt, color="Importance", color_continuous_scale="Viridis")
        fig.update_layout(yaxis={"categoryorder": "total ascending"},height=450)

        return (
            result_md,
            fig,
            full_data_for_db,
            "Analyza dokoncena. Mozte zamestnanca zapisat do databazy."
        )

    def save_to_database(data_from_state):
        if data_from_state is None:
            return "Najskor kliknite na Analyzovat, az potom mozte ukladat."

        success, new_id, msg = add_new_employee_to_db(data_from_state, df_sample)
        if success:
            return f"Uspesne ulozene. Nove zamestnanecke cislo: {new_id}"
        return f"Chyba pri ukladani: {msg}"

    def filter_database(name_query, dept_query, id_query):
        try:
            df_filtered = predictions_safe.copy()

            if name_query and "FullName" in df_filtered.columns:
                df_filtered = df_filtered[
                    df_filtered["FullName"].astype(str).str.contains(name_query, case=False, na=False)
                ]

            if dept_query and dept_query != "Všetky" and "Department" in df_filtered.columns:
                df_filtered = df_filtered[df_filtered["Department"].astype(str) == dept_query]

            if id_query and "EmployeeNumber" in df_filtered.columns:
                df_filtered = df_filtered[
                    df_filtered["EmployeeNumber"].astype(str).str.contains(str(id_query), na=False)
                ]

            return df_filtered, f"Nájdených **{len(df_filtered)}** záznamov."
        except Exception as e:
            return predictions_safe.head(0), f"Chyba pri filtrovaní: {str(e)}"

    # e. UI Definition

    custom_css = """
    .app-header {
        padding: 18px 8px 8px 8px;
        margin-bottom: 6px;
    }
    .app-subtitle {
        color: #6b7280;
        margin-top: -8px;
        margin-bottom: 18px;
    }
    .metric-card {
        border-radius: 18px;
        padding: 16px;
        border: 1px solid #e5e7eb;
        background: linear-gradient(180deg, #ffffff 0%, #f9fafb 100%);
        box-shadow: 0 4px 18px rgba(0,0,0,0.04);
    }
    """

    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="HR Analytics AI") as demo:
        gr.HTML("""
        <div class="app-header">
            <h1 style="margin-bottom:6px;">HR Analytics AI Platform</h1>
            <div class="app-subtitle">
                Analyticky panel pre pravdepodobnost odchodu, anomalie a vplyv faktorov zamestnania.
            </div>
        </div>
        """)

        current_data_state = gr.State(None)

        with gr.Tabs():

            # f. TAB 1 - New employee simulator

            with gr.Tab("Simulator zamestnanca"):
                gr.Markdown("### Vstupne udaje pre individualnu analyzu")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 1. Osobne udaje")
                        in_fname = gr.Textbox(label="Meno", value="")
                        in_lname = gr.Textbox(label="Priezvisko", value="")

                        with gr.Row():
                            in_gender = gr.Dropdown(["Male", "Female"], label="Pohlavie", value="Male")
                            in_age = gr.Slider(18, 65, value=30, step=1, label="Vek")

                        in_marital = gr.Dropdown(
                            sorted(list(df_sample["MaritalStatus"].dropna().unique())),
                            label="Rodinny stav",
                            value="Single" if "Single" in df_sample["MaritalStatus"].astype(str).unique() else None
                        )

                        in_edu_level = gr.Dropdown(
                            list(edu_map.keys()),
                            label="Vzdelanie",
                            value="3 - Bakalarske"
                        )

                        in_edu_field = gr.Dropdown(
                            sorted(list(df_sample["EducationField"].dropna().unique())),
                            label="Zameranie vzdelania",
                            value=sorted(list(df_sample["EducationField"].dropna().unique()))[0]
                        )

                        in_dist = gr.Slider(1, 30, value=5, step=1, label="Vzdialenosť z domu (km)")

                    with gr.Column():
                        gr.Markdown("#### 2. Historia pozicie")
                        in_dept = gr.Dropdown(
                            sorted(list(df_sample["Department"].dropna().unique())),
                            label="Oddelenie",
                            value=sorted(list(df_sample["Department"].dropna().unique()))[0]
                        )

                        in_role = gr.Dropdown(
                            sorted(list(df_sample["JobRole"].dropna().unique())),
                            label="Rola zamestnanca",
                            value=sorted(list(df_sample["JobRole"].dropna().unique()))[0]
                        )

                        in_total_years = gr.Slider(0, 40, value=10, step=0.5, label="Celkova prax")
                        in_num_comp = gr.Slider(0, 10, value=1, step=1, label="Počet predchádzajucich firiem")

                        with gr.Row():
                            in_years_co = gr.Slider(0, 40, value=5, step=0.5, label="Roky vo firme")
                            in_years_role = gr.Slider(0, 20, value=3, step=0.5, label="Roky v aktualnej roli")

                        with gr.Row():
                            in_years_promo = gr.Slider(0, 20, value=1, step=0.5, label="Roky od povysenia")
                            in_years_man = gr.Slider(0, 20, value=2, step=0.5, label="Roky s aktualnym manazerom")

                    with gr.Column():
                        gr.Markdown("#### 3. Vykon a odmena")

                        with gr.Row():
                            in_perf = gr.Slider(1, 4, value=3, step=1, label="Vykonnostne hodnotenie")
                            in_training = gr.Slider(0, 6, value=2, step=1, label="Pocet skoleni")

                        in_income = gr.Slider(1000, 25000, value=5000, step=100, label="Mesacny prijem")
                        in_hike = gr.Slider(0, 30, value=10, step=0.5, label="Rocny Percentualny narast mzdy")

                        with gr.Row():
                            in_over = gr.Radio(["Yes", "No"], label="Nadcasy", value="No")
                            in_travel = gr.Dropdown(
                                sorted(list(df_sample["BusinessTravel"].dropna().unique())),
                                label="Frekvencia sluzobnych ciest",
                                value=sorted(list(df_sample["BusinessTravel"].dropna().unique()))[0]
                            )

                gr.Markdown("---")

                with gr.Row():
                    btn_analyze = gr.Button("1. Analyzovat", variant="primary")
                    btn_save = gr.Button("2. Zapisat do Databazy", variant="secondary")

                with gr.Row():
                    with gr.Column(scale=1):
                        out_txt = gr.Markdown()
                        out_status = gr.Textbox(label="Stav procesu", interactive=False)
                    with gr.Column(scale=2):
                        out_plt = gr.Plot()

                inputs_list = [
                    in_fname, in_lname, in_gender, in_age, in_marital, in_edu_level, in_edu_field, in_dist,
                    in_dept, in_role, in_num_comp, in_total_years,
                    in_years_co, in_years_role, in_years_promo, in_years_man,
                    in_perf, in_income, in_over, in_hike, in_training, in_travel
                ]

                btn_analyze.click(
                    fn=analyze_new_employee,
                    inputs=inputs_list,
                    outputs=[out_txt, out_plt, current_data_state, out_status]
                )

                btn_save.click(
                    fn=save_to_database,
                    inputs=[current_data_state],
                    outputs=[out_status]
                )

            # g. TAB 2: Anomalies dashboard

            with gr.Tab("Anomalie"):
                gr.Markdown("Manazersky prehlad")

                with gr.Row():
                    gr.Markdown(
                        f"""
                        <div class="metric-card">
                            <div style="font-size:13px;color:#6b7280;">Celkový počct zamestnancov</div>
                            <div style="font-size:28px;font-weight:700;">{total_employees}</div>
                        </div>
                        """
                    )
                    gr.Markdown(
                        f"""
                        <div class="metric-card">
                            <div style="font-size:13px;color:#6b7280;">High Risk</div>
                            <div style="font-size:28px;font-weight:700;color:#dc2626;">{high_risk_count}</div>
                        </div>
                        """
                    )
                    gr.Markdown(
                        f"""
                        <div class="metric-card">
                            <div style="font-size:13px;color:#6b7280;">Medium Risk</div>
                            <div style="font-size:28px;font-weight:700;color:#d97706;">{medium_risk_count}</div>
                        </div>
                        """
                    )
                    gr.Markdown(
                        f"""
                        <div class="metric-card">
                            <div style="font-size:13px;color:#6b7280;">Priemerná pravdepodobnosť attrition</div>
                            <div style="font-size:28px;font-weight:700;">{avg_attrition_prob:.2f}%</div>
                        </div>
                        """
                    )

                gr.Markdown("### Vyhladavanie")

                with gr.Row():
                    search_name = gr.Textbox(label="Meno", placeholder="Zadajte meno alebo priezvisko")
                    search_dept = gr.Dropdown(
                        choices=["Všetky"] + (
                            sorted(list(predictions_safe["Department"].dropna().unique()))
                            if "Department" in predictions_safe.columns else []
                        ),
                        label="Oddelenie",
                        value="Všetky"
                    )
                    search_id = gr.Textbox(label="EmployeeNumber", placeholder="Napr. 1024")

                lbl_count = gr.Markdown(value=f"Najdenych **{len(predictions_safe)}** zaznamov.")
                data_table = gr.Dataframe(value=predictions_safe, interactive=False, label="Predictions_Attrition")

                inputs_search = [search_name, search_dept, search_id]
                outputs_search = [data_table, lbl_count]

                search_name.change(filter_database, inputs_search, outputs_search)
                search_dept.change(filter_database, inputs_search, outputs_search)
                search_id.change(filter_database, inputs_search, outputs_search)

                gr.Markdown("---")
                gr.Markdown("### Anomalie v oddeleniach a poziciach")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Dataframe(
                            value=anom_kpi,
                            interactive=False,
                            label="KPI anomálii"
                        )
                    with gr.Column(scale=2):
                        gr.Plot(value=anom_fig)

                gr.Dataframe(
                    value=anom_table,
                    interactive=False,
                    label="Detail anomálii"
                )

            # h. TAB 3: Heatmap of factor based on job position

            with gr.Tab("Faktory podla pozicie"):
                gr.Markdown("### Dolezitost faktorov odchodu podla Pracovnej role")

                if heatmap_fig is not None:
                    gr.Plot(value=heatmap_fig)
                    gr.Dataframe(
                        value=feat_imp_role_safe,
                        interactive=False,
                        label="Model_Feature_Importance_By_Role"
                    )
                else:
                    gr.Markdown("*Heatmapu nebolo mozne vygenerovať — pravdepodobne chybaju data.*")

    return demo

# ==========================
# 7. MAIN FUNCTION
# ==========================
if __name__ == "__main__":
    print("=== HR Analytics Pipeline Started ===")

    try:
        df = load_hr_dataframe()
    except Exception as e:
        print(f"CRITICAL ERROR (DB): {e}")
        exit(1)

    # 1. Classification
    model, encoders, feature_names, acc, feat_imp, predictions_df = get_model_and_predictions(df)

    # 2. Regression
    anomalies_df = detect_department_anomalies(df, n_optuna_trials=100)

    # 3. Feature importance by role
    feat_imp_by_role = generate_feature_importance_by_role(df)

    # Save Results
    save_to_sql(predictions_df, "Predictions_Attrition")
    save_to_sql(feat_imp, "Model_Feature_Importance")
    save_to_sql(feat_imp_by_role, "Model_Feature_Importance_By_Role")
    save_to_sql(anomalies_df, "Anomalies_Department")

    print("\nVsetko je hotove. Spustam front-end.")
    app = run_gradio_app(
        model=model,
        encoders=encoders,
        feature_names=feature_names,
        global_acc=acc,
        importance_df=feat_imp,
        df_sample=df,
        full_predictions_df=predictions_df,
        feat_imp_by_role=feat_imp_by_role,
        anomalies_df=anomalies_df
    )
    app.launch(share=False)