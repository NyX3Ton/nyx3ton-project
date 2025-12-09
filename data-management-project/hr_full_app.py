#%pip install --upgrade pip
#%pip install --upgrade optuna pandas numpy sqlalchemy pyodbc matplotlib seaborn plotly scikit-learn xgboost joblib gradio
# ---------------------------------------------------------
# Komplexný HR Analytický Nástroj (Final Version)
# 1. Load Data (SQL)
# 2. Smart Model Training (Reuse Params + Optuna + GPU)
# 3. Anomaly Detection (Regression)
# 4. Feature Importance per Job Role
# 5. Gradio Frontend (Extended Input + Search + Heatmap)
# ---------------------------------------------------------
import os, warnings, unicodedata, joblib, optuna, urllib.parse

from pathlib import Path
import pandas as pd
import numpy as np
import sqlalchemy as sa

# Vizualizácie
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Machine Learning
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor
import optuna

cwd = os.getcwd()
#os.makedirs('./Output', exist_ok=True) 
#os.makedirs('./Input', exist_ok=True)  
os.makedirs('./model_save/model', exist_ok=True) 

# Frontend
import gradio as gr

# Konfigurácia
warnings.filterwarnings('ignore')
DATA_ENV_PATH = Path("data.env")

# Cesty k súborom
MODEL_PATH = "./model_save/model_attrition.pkl"
ENCODER_PATH = "./model_save/encoder_attrition.pkl"
FEATURE_COLS_PATH = "./model_save/feature_cols.pkl"

# ==========================
# 0. GPU / CPU Detection
# ==========================

def get_xgboost_device():
    try:
        X_test = np.array([[1, 2], [3, 4]])
        y_test = np.array([0, 1])
        # Skúsime tréning na GPU
        test_model = XGBClassifier(device="cuda", n_estimators=1, tree_method="hist")
        test_model.fit(X_test, y_test)
        print("[SYSTEM] GPU (CUDA) detekované! Výpočty pobežia na grafickej karte.")
        return "cuda"
    except Exception:
        print("[SYSTEM] GPU (CUDA) nedostupné. Prepínam na CPU.")
        return "cpu"

ACTIVE_DEVICE = get_xgboost_device()

# ==========================
# 1. SQL Engine & Data Load
# ==========================

def load_env_from_file(env_path: Path = DATA_ENV_PATH) -> None:
    if not env_path.exists():
        raise FileNotFoundError(f"Env file not found: {env_path}")
    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()

def make_sql_engine() -> sa.Engine:
    load_env_from_file()
    required = ["SQL_SERVER", "SQL_DATABASE", "SQL_USERNAME", "SQL_PASSWORD", "SQL_DRIVER"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Chýbajú premenné prostredia: {missing}")

    params = urllib.parse.quote_plus(
        f"DRIVER={{{os.getenv('SQL_DRIVER')}}};"
        f"SERVER={os.getenv('SQL_SERVER')};"
        f"DATABASE={os.getenv('SQL_DATABASE')};"
        f"UID={os.getenv('SQL_USERNAME')};"
        f"PWD={os.getenv('SQL_PASSWORD')};"
        f"Encrypt={os.getenv('SQL_ENCRYPT', 'no')};"
        f"TrustServerCertificate={os.getenv('SQL_TRUSTCERT', 'yes')};"
    )
    return sa.create_engine(f"mssql+pyodbc:///?odbc_connect={params}", fast_executemany=True)

def load_hr_dataframe() -> pd.DataFrame:
    engine = make_sql_engine()
    print("--> Načítavam dáta z SQL...")
    query = "SELECT * FROM dbo.HR_Synth_Data"
    df = pd.read_sql(query, engine)
    print(f"    Načítaných {len(df)} riadkov.")
    return df

def save_to_sql(df: pd.DataFrame, table_name: str):
    engine = make_sql_engine()
    print(f"--> Zapisujem tabuľku: {table_name} ({len(df)} riadkov)...")
    try:
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print("    Zápis úspešný.")
    except Exception as e:
        print(f"    Chyba pri zápise do SQL: {e}")

# ==========================
# 2. Database Helpers (Add New Employee)
# ==========================

def get_next_employee_number(engine):
    """Zistí maximálne ID v databáze a vráti +1"""
    try:
        query = "SELECT MAX(EmployeeNumber) FROM dbo.HR_Synth_Data"
        max_id = pd.read_sql(query, engine).iloc[0, 0]
        if pd.isna(max_id):
            return 1
        return int(max_id) + 1
    except Exception as e:
        print(f"Chyba pri získavaní ID: {e}")
        return np.random.randint(10000, 99999)

def add_new_employee_to_db(input_data: dict, df_template: pd.DataFrame):
    engine = make_sql_engine()
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

        if df_template[col].dtype.name in ['category', 'object', 'string']:
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
    
    # Priradenie ID
    next_id = get_next_employee_number(engine)
    new_row_df["EmployeeNumber"] = next_id
    
    print(f"--> Pridávam zamestnanca ID {next_id} do DB (Smart Sampling)...")
    try:
        new_row_df.to_sql("HR_Synth_Data", engine, if_exists='append', index=False)
        return True, next_id, f"Zamestnanec {input_data.get('FullName', '')} (ID: {next_id}) bol úspešne uložený."
    except Exception as e:
        print(f"SQL Error: {e}")
        return False, None, f"Chyba pri ukladaní do DB: {str(e)}"

# ==========================
# 3. Preprocessing
# ==========================

def get_drop_columns():
    return ["EmployeeNumber", "EmployeeCount", "StandardHours", "Over18", 
            "FirstName", "LastName", "FullName", "Email", "Username","DailyRate", 
            "HourlyRate","MonthlyRate", "RelationshipSatisfaction","JobLevel", "WorkLifeBalance", "JobInvolvement"]
    
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
    cat_cols = df_proc.select_dtypes(include=["object", "category"]).columns

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
            # Bezpečné mapovanie: ak hodnota nie je známa, dáme 0
            X[col] = X[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else 0)
            X[col] = X[col].fillna(0).astype(int)
            
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    return X

# ==========================
# 4. Training Logic (Reuse Params or Optuna)
# ==========================

def run_optuna_optimization(X_train, y_train, X_test, y_test, n_trials=25):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
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
            "early_stopping_rounds": 5,
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test)
        return accuracy_score(y_test, preds)

    print(f"--> Spúšťam Optuna optimalizáciu ({n_trials} pokusov) na {ACTIVE_DEVICE}...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"    Najlepšia Accuracy: {study.best_value:.4f}")
    return study.best_params

def train_and_save_new_model(df_full: pd.DataFrame, n_trials=25, reuse_prev_params=True):
    print("\n--- Spúšťam tréning modelu ---")

    X, y, encoders = preprocess_data_for_training(df_full)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    best_params = None
    if reuse_prev_params and os.path.exists(MODEL_PATH):
        try:
            print(f"--> Pokúšam sa načítať parametre z existujúceho modelu ({MODEL_PATH})...")
            old_model = joblib.load(MODEL_PATH)
            best_params = old_model.get_params()
            print(" Parametre úspešne načítané. Preskakujem Optunu.")
            
            # Vyčistenie a update pre aktuálny device
            keys_to_remove = ['missing', 'callbacks', 'monotone_constraints', 'interaction_constraints']
            for k in keys_to_remove:
                if k in best_params: del best_params[k]
        except Exception as e:
            print(f"Nepodarilo sa načítať staré parametre ({e}). Spustím Optunu.")
            best_params = None

    if best_params is None:
        best_params = run_optuna_optimization(X_train, y_train, X_test, y_test, n_trials)
    
    best_params.update({
        "random_state": 42, "n_jobs": -1, "tree_method": "hist",
        "device": ACTIVE_DEVICE, "objective": "binary:logistic", 
        "eval_metric": "logloss", "early_stopping_rounds":5
    })
    
    print(f"--> Trénujem finálny model na celom datasete ({ACTIVE_DEVICE})...")
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    acc = accuracy_score(y_test, final_model.predict(X_test))
    print(f"    Finálna Presnosť: {acc:.2%}")
    
    print(f"--> Ukladám model do {MODEL_PATH}...")
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    joblib.dump(feature_names, FEATURE_COLS_PATH)
    
    return final_model, encoders, feature_names, acc

# ==========================
# 5. Smart Loading Logic
# ==========================

def get_model_and_predictions(df_full: pd.DataFrame):
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH) and os.path.exists(FEATURE_COLS_PATH):
        print(f"\n[INFO] Model nájdený ({MODEL_PATH}). Načítavam...")
        try:
            model = joblib.load(MODEL_PATH)
            encoders = joblib.load(ENCODER_PATH)
            feature_names = joblib.load(FEATURE_COLS_PATH)
            acc = 0.0 
        except Exception as e:
            print(f"[ERROR] Chyba pri načítaní: {e}. Spustím nový tréning.")
            model, encoders, feature_names, acc = train_and_save_new_model(df_full, reuse_prev_params=False)
    else:
        print(f"\n[INFO] Model nenájdený. Spúšťam tréning...")
        model, encoders, feature_names, acc = train_and_save_new_model(df_full, reuse_prev_params=False)

    print("--> Generujem predikcie pre aktuálne dáta...")
    X_current = preprocess_data_for_inference(df_full, encoders, feature_names)
    
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    predictions_df = df_full.copy()
    all_probs = model.predict_proba(X_current)[:, 1]
    
    predictions_df["Risk_Label"] = pd.cut(
        all_probs, 
        bins=[-0.1, 0.25, 0.50, 1.1], 
        labels=["Low", "Medium", "High"]
    )
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

# ==========================
# 6. Analytics (Anomalies & Feature Imp by Role)
# ==========================

def detect_department_anomalies(df_full: pd.DataFrame, n_optuna_trials=25):
    print("\n--- 2. Detekcia Anomálií (Regression + Optuna) ---")

    df_agg = df_full.copy()
    df_agg["Attrition_Flag"] = df_agg["Attrition"].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    df_agg["OverTime_Flag"] = df_agg["OverTime"].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)

    group_cols = ["Department", "JobRole"]
    
    dept_stats = df_agg.groupby(group_cols).agg({
        "EmployeeNumber": "count",
        "Attrition_Flag": "mean",       
        "OverTime_Flag": "mean",        
        "EnvironmentSatisfaction": lambda x: x.astype(int).mean(),
        "WorkLifeBalance": lambda x: x.astype(int).mean()
    }).reset_index()

    dept_stats.rename(columns={
        "EmployeeNumber": "Headcount", 
        "Attrition_Flag": "Actual_Attrition_Rate",
        "OverTime_Flag": "Avg_OverTime"
    }, inplace=True)

    dept_stats = dept_stats[dept_stats["Headcount"] > 2].copy()

    features = ["Avg_OverTime", "EnvironmentSatisfaction", "WorkLifeBalance"]
    X = dept_stats[features]
    y = dept_stats["Actual_Attrition_Rate"]

    if len(dept_stats) < 10:
        print("   [INFO] Málo dát pre Optunu. Default config.")
        best_params = {"n_estimators": 300, "max_depth": 10, "learning_rate": 0.001, "objective": "reg:squarederror", "device": ACTIVE_DEVICE}
    else:
        def objective_reg(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 800),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
                "random_state": 42, "n_jobs": -1, "tree_method": "hist",
                "device": ACTIVE_DEVICE, "objective": "reg:squarederror"
            }
            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            return np.sqrt(mean_squared_error(y_val, model.predict(X_val)))

        print(f"--> Spúšťam Optunu pre Regresiu ({n_optuna_trials} pokusov)...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study_reg = optuna.create_study(direction="minimize")
        study_reg.optimize(objective_reg, n_trials=n_optuna_trials, show_progress_bar=True)
        best_params = study_reg.best_params

    final_params = best_params.copy()
    final_params.update({"random_state": 42, "n_jobs": -1, "objective": "reg:squarederror", "device": ACTIVE_DEVICE})
    
    reg_model = XGBRegressor(**final_params)
    reg_model.fit(X, y)
    
    dept_stats["Predicted_Attrition_Rate"] = reg_model.predict(X)
    residuals = dept_stats["Actual_Attrition_Rate"] - dept_stats["Predicted_Attrition_Rate"]
    dept_stats["Anomaly_Score"] = residuals
    threshold = np.mean(residuals) + (2 * np.std(residuals))
    
    print(f"   [INFO] Anomaly Threshold (Mean + 2*Std): {threshold:.4f}")
    
    dept_stats["Is_Anomaly"] = dept_stats["Anomaly_Score"] > threshold
    dept_stats = dept_stats.sort_values(by="Anomaly_Score", ascending=False)

    cols_to_round = ["Actual_Attrition_Rate", "Avg_OverTime", "EnvironmentSatisfaction", 
                        "WorkLifeBalance", "Predicted_Attrition_Rate", "Anomaly_Score"]
    
    for col in cols_to_round:
        if col in dept_stats.columns:
            dept_stats[col] = dept_stats[col].round(3)

    return dept_stats

def generate_feature_importance_by_role(df_full: pd.DataFrame):
    print("\n--- 3. Generovanie Feature Importance podľa JobRole ---")
    results = []
    roles = df_full["JobRole"].unique()
    
    for role in roles:
        df_subset = df_full[df_full["JobRole"] == role].copy()
        if len(df_subset) < 10: continue
            
        X, y, _ = preprocess_data_for_training(df_subset)
        if "JobRole" in X.columns: X = X.drop(columns=["JobRole"])
        if y.nunique() < 2: continue

        model = XGBClassifier(n_estimators=300, max_depth=10, random_state=42, 
                                device=ACTIVE_DEVICE, tree_method="hist", verbose=0)
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

# ==========================
# 7. Gradio Frontend
# ==========================
def run_gradio_app(model, encoders, feature_names, global_acc, importance_df, df_sample, full_predictions_df, feat_imp_by_role):
    
    # Mapovanie vzdelania
    edu_map = {
        "1 - Základné (Elementary)": 1,
        "2 - Stredoškolské (High School)": 2,
        "3 - Bakalárske (Bachelor)": 3, 
        "4 - Magisterské/Inžinierske (Master)": 4,
        "5 - Doktorandské (Doctor)": 5
    }

    # --- 1. FUNKCIA: LEN ANALÝZA (Uloží dáta do State) ---
    def analyze_new_employee(
        fname, lname, gender, age, marital, edu_level_str, edu_field, dist_home,
        dept, role, num_comp, total_years,
        years_at_company, years_curr_role, years_since_promo, years_curr_manager,
        env_sat, job_sat, perf_rating, 
        income, overtime, percent_hike, train_times, travel
    ):
        full_name = f"{fname} {lname}"
        
        # Generovanie emailu
        def strip_accents(s):
            return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        clean_fname = strip_accents(fname.lower()).replace(" ", "")
        clean_lname = strip_accents(lname.lower()).replace(" ", "")
        email = f"{clean_fname}.{clean_lname}@company.com"
        username = f"{clean_fname}.{clean_lname}"

        edu_level_int = edu_map[edu_level_str]

        # Príprava dát
        input_dict = {
            "Age": age, "Gender": gender, "MaritalStatus": marital,
            "Education": edu_level_int, "EducationField": edu_field, "DistanceFromHome": dist_home,
            "Department": dept, "JobRole": role, 
            "NumCompaniesWorked": num_comp, "TotalWorkingYears": total_years,
            "YearsAtCompany": years_at_company, "YearsInCurrentRole": years_curr_role,
            "YearsSinceLastPromotion": years_since_promo, "YearsWithCurrManager": years_curr_manager,
            "EnvironmentSatisfaction": env_sat, "JobSatisfaction": job_sat, 
            "PerformanceRating": perf_rating, 
            "MonthlyIncome": income, "OverTime": overtime, 
            "PercentSalaryHike": percent_hike, "TrainingTimesLastYear": train_times, "BusinessTravel": travel
        }

        # Obohatíme dict o mená pre DB
        full_data_for_db = input_dict.copy()
        full_data_for_db.update({"FirstName": fname, "LastName": lname, "FullName": full_name, "Email": email, "Username": username})

        # --- PREDIKCIA ---
        pred_df = df_sample.iloc[0:1].copy()
        for k, v in input_dict.items():
            if k in pred_df.columns:
                pred_df[k] = v
        
        # Preprocessing
        X_new = preprocess_data_for_inference(pred_df, encoders, feature_names)
        prob = model.predict_proba(X_new)[0, 1]
        
        # Logika pre farby
        if prob <= 0.25: decision, color = "NÍZKE RIZIKO", "green"
        elif prob <= 0.50: decision, color = "MIERNE RIZIKO", "orange"
        else: decision, color = "VYSOKÉ RIZIKO", "red"

        result_md = f"### Výsledok pre: {full_name}\n**Email:** {email}\n**Pravdepodobnosť odchodu:** {prob:.2%}\n**Verdikt:** <span style='color:{color}; font-weight:bold; font-size:16px'>{decision}</span>"
        
        # Graf (Dynamický podľa role)
        role_specific_data = feat_imp_by_role[feat_imp_by_role["JobRole"] == role]
        if not role_specific_data.empty:
            row = role_specific_data.iloc[0].drop("JobRole")
            local_imp_df = pd.DataFrame({"Feature": row.index, "Importance": row.values}).sort_values(by="Importance", ascending=False)
            local_imp_df["Importance"] = local_imp_df["Importance"].astype(float)
            title_txt = f"Top faktory pre: {role}"
        else:
            local_imp_df = importance_df.copy()
            title_txt = "Top faktory (Global)"

        fig = px.bar(local_imp_df.head(10), x='Importance', y='Feature', orientation='h', title=title_txt, color='Importance', color_continuous_scale="Viridis")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        
        return result_md, fig, full_data_for_db, "Analýza hotová. Skontrolujte výsledok a kliknite na 'Zapísať do DB'."

    # --- 2. FUNKCIA: LEN ZÁPIS ---
    def save_to_database(data_from_state):
        if data_from_state is None:
            return "Najskôr musíte kliknúť na 'Analyzovať', až potom môžete ukladať."
        
        success, new_id, msg = add_new_employee_to_db(data_from_state, df_sample)
        if success:
            return f"Úspešne uložené! (ID: {new_id})"
        else:
            return f"Chyba pri ukladaní: {msg}"

    # --- UI LAYOUT ---
    with gr.Blocks(title="HR Analytics AI") as demo:
        gr.Markdown(f"# HR AI Analytics Platform")
        
        current_data_state = gr.State(None)

        with gr.Tabs():
            # ================= TAB 1: SIMULATOR =================
            with gr.Tab("Detailný Simulátor"):
                with gr.Row():
                    # STĹPEC 1: Osobné
                    with gr.Column():
                        gr.Markdown("### 1. Osobné údaje")
                        in_fname = gr.Textbox(label="Meno", value="")
                        in_lname = gr.Textbox(label="Priezvisko", value="")
                        with gr.Row():
                            in_gender = gr.Dropdown(["Male", "Female"], label="Pohlavie", value="Male")
                            in_age = gr.Slider(18, 65, value=30, label="Vek")
                        in_marital = gr.Dropdown(sorted(list(df_sample["MaritalStatus"].unique())), label="Stav", value="Single")
                        in_edu_level = gr.Dropdown(list(edu_map.keys()), label="Vzdelanie", value="3 - Bakalárske (Bachelor)")
                        in_edu_field = gr.Dropdown(sorted(list(df_sample["EducationField"].unique())), label="Odbor", value="Life Sciences")
                        in_dist = gr.Slider(1, 30, value=5, label="Vzdialenosť z domu (km)")

                    # STĹPEC 2: Práca
                    with gr.Column():
                        gr.Markdown("### 2. Pozícia & História")
                        in_dept = gr.Dropdown(sorted(list(df_sample["Department"].unique())), label="Oddelenie", value="Sales")
                        in_role = gr.Dropdown(sorted(list(df_sample["JobRole"].unique())), label="Pozícia", value="Sales Executive")
                        
                        gr.Markdown("#### Skúsenosti (Roky)")
                        in_total_years = gr.Slider(0, 40, value=10, label="Celková prax", step=0.5)
                        in_num_comp = gr.Slider(0, 10, value=1, label="Počet firiem", step=1)
                        with gr.Row():
                            in_years_co = gr.Slider(0, 40, value=5, label="Vo firme", step=0.5)
                            in_years_role = gr.Slider(0, 20, value=3, label="V roli", step=0.5)
                        with gr.Row():
                            in_years_promo = gr.Slider(0, 20, value=1, label="Od povýšenia", step=0.5)
                            in_years_man = gr.Slider(0, 20, value=2, label="S manažérom", step=0.5)

                    # STĹPEC 3: Spokojnosť & Peniaze
                    with gr.Column():
                        gr.Markdown("### 3. Spokojnosť & Odmeňovanie")
                        with gr.Row():
                            in_env_sat = gr.Slider(1, 4, value=3, label="Env. Sat.", step=1)
                            in_job_sat = gr.Slider(1, 4, value=3, label="Job Sat.", step=1)
                        with gr.Row():
                            in_perf = gr.Slider(1, 4, value=3, label="Performance", step=1)
                        
                        gr.Markdown("#### Financie")
                        in_income = gr.Slider(1000, 25000, value=5000, label="Mesačný Plat", step=100)
                        in_hike = gr.Slider(0, 30, value=10, label="% Zvýšenie platu", step=0.5)
                        
                        with gr.Row():
                            in_over = gr.Radio(["Yes", "No"], label="Nadčasy", value="No")
                            in_travel = gr.Dropdown(sorted(list(df_sample["BusinessTravel"].unique())), label="Cestovanie", value="Travel_Rarely")
                        
                        in_training = gr.Slider(0, 6, value=2, label="Počet školení (minulý rok)")

                # TLAČIDLÁ
                gr.Markdown("---")
                with gr.Row():
                    btn_analyze = gr.Button("1. Analyzovať", variant="primary")
                    btn_save = gr.Button("2. Zapísať do DB", variant="secondary")
                
                # VÝSTUPY
                with gr.Row():
                    with gr.Column(scale=1):
                        out_txt = gr.Markdown()
                        out_status = gr.Textbox(label="Stav procesu", interactive=False)
                    with gr.Column(scale=2):
                        out_plt = gr.Plot()
                
                # --- PREPOJENIE UI ---
                inputs_list = [
                    in_fname, in_lname, in_gender, in_age, in_marital, in_edu_level, in_edu_field, in_dist,
                    in_dept, in_role, in_num_comp, in_total_years, 
                    in_years_co, in_years_role, in_years_promo, in_years_man,
                    in_env_sat, in_job_sat, in_perf, 
                    in_income, in_over, in_hike, in_training, in_travel
                ]
                
                btn_analyze.click(
                    analyze_new_employee, 
                    inputs=inputs_list, 
                    outputs=[out_txt, out_plt, current_data_state, out_status]
                )
                
                btn_save.click(
                    save_to_database, 
                    inputs=[current_data_state], 
                    outputs=[out_status]
                )

            # ================= TAB 2: DASHBOARD =================
            with gr.Tab(" Dashboard & Anomálie"):
                gr.Markdown("### Vyhľadávanie v predikciách")
                
                def filter_database(name_query, dept_query, id_query):
                    try:
                        df_filtered = full_predictions_df.copy()
                        if name_query: 
                            df_filtered = df_filtered[df_filtered["FullName"].astype(str).str.contains(name_query, case=False, na=False)]
                        if dept_query and dept_query != "Všetky": 
                            df_filtered = df_filtered[df_filtered["Department"].astype(str) == dept_query]
                        if id_query: 
                            df_filtered = df_filtered[df_filtered["EmployeeNumber"].astype(str).str.contains(id_query, na=False)]
                        return df_filtered, f"Nájdených **{len(df_filtered)}** záznamov."
                    except Exception as e: 
                        return full_predictions_df.head(0), f"Chyba: {str(e)}"
                
                with gr.Row():
                    search_name = gr.Textbox(label="Meno", placeholder="Hľadať...")
                    search_dept = gr.Dropdown(choices=["Všetky"] + sorted(list(full_predictions_df["Department"].unique())), label="Oddelenie", value="Všetky")
                    search_id = gr.Textbox(label="ID", placeholder="123")
                
                lbl_count = gr.Markdown(value=f"Nájdených **{len(full_predictions_df)}** záznamov.")
                data_table = gr.Dataframe(value=full_predictions_df, interactive=False)
                
                inputs_search = [search_name, search_dept, search_id]
                outputs_search = [data_table, lbl_count]
                
                search_name.change(filter_database, inputs_search, outputs_search)
                search_dept.change(filter_database, inputs_search, outputs_search)
                search_id.change(filter_database, inputs_search, outputs_search)
                
                gr.Markdown("### Anomálie v oddeleniach")
                try: 
                    gr.Dataframe(value=anomalies_df, label="Prehľad anomálií")
                except: 
                    pass

            # ================= TAB 3: HEATMAP (FAKTORY) =================
            with gr.Tab("Faktory podľa Pozície"):
                gr.Markdown("### Čo ovplyvňuje odchod na jednotlivých pozíciách?")
                if not feat_imp_by_role.empty:
                    # Graf Heatmap
                    plot_df = feat_imp_by_role.set_index("JobRole")
                    fig_heat = px.imshow(
                        plot_df, 
                        labels=dict(x="Faktor", y="Pozícia", color="Dôležitosť"), 
                        x=plot_df.columns, 
                        y=plot_df.index, 
                        aspect="auto", 
                        color_continuous_scale="Viridis"
                    )
                    fig_heat.update_layout(title="Mapa dôležitosti faktorov")
                    gr.Plot(fig_heat)
                    
                    # Tabuľka
                    gr.Dataframe(value=feat_imp_by_role, interactive=False, label="Detailné dáta")
                else:
                    gr.Markdown("*Nedostatok dát na generovanie heatmapy.*")

    return demo
# ==========================
# MAIN EXECUTION
# ==========================

if __name__ == "__main__":
    print(f"=== HR Analytics Pipeline Started ===")
    
    try:
        df = load_hr_dataframe()
    except Exception as e:
        print(f"CRITICAL ERROR (DB): {e}")
        exit(1)
    
    # 1. Classification
    model, encoders, feature_names, acc, feat_imp, predictions_df = get_model_and_predictions(df)
    
    # 2. Regression
    anomalies_df = detect_department_anomalies(df, n_optuna_trials=25)
    
    # 3. Feature Imp by Role
    feat_imp_by_role = generate_feature_importance_by_role(df)
    
    # Save Results
    save_to_sql(predictions_df, "Predictions_Attrition")
    save_to_sql(feat_imp, "Model_Feature_Importance")
    save_to_sql(feat_imp_by_role, "Model_Feature_Importance_By_Role")
    save_to_sql(anomalies_df, "Anomalies_Department")
    
    print("\nVšetko hotovo. Spúšťam Gradio")
    app = run_gradio_app(model, encoders, feature_names, acc, feat_imp, df, predictions_df, feat_imp_by_role)
    app.launch(share=False)
