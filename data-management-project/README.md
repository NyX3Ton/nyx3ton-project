# HR Attrition tool (backend + frontend)

Povinne kniznice na beh
```text

!pip install --upgrade pip
!pip install --upgrade optuna pandas numpy sqlalchemy pyodbc matplotlib seaborn plotly scikit-learn xgboost joblib gradio psycopg2-binary


```

## 1. Konfiguracia
```text
ACTIVE_DEVICE = get_xgboost_device()

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


Cez Optunu XGBClassifier (n_trials = 100)

n_estimators = <100, 1600>
max_depth = <5, 15>
earning_rate = <0.01, 0.3>
subsample = <0.6, 1.0>
colsample_bytree = <0.6, 1.0?
min_child_weight = <1, 10>
gamma = <0, 5>
n_jobs = -1,
tree_method = hist
device = ACTIVE_DEVICE
objective = binary:logistic
eval_metric = aucpr
early_stopping_rounds = 25

Cez Optunu XGBRegressor (n_trials = 100)

n_estimators = <100, 1200>
max_depth = <5, 15>
earning_rate = <0.001, 0.3>
subsample = <0.5, 1.0>
colsample_bytree = <0.5, 1.0?
min_child_weight = <1, 5>
n_jobs = -1,
tree_method = hist
device = ACTIVE_DEVICE
objective = binary:logistic
eval_metric = aucpr
early_stopping_rounds = 25
```

## 2. Ako funguje pipeline

```text
CSV konvertovany do Pandas dataframe (script csv_odbc.ipynb)
        |
        v
Simulovane data pridane pomocou Fakeru (podla pohlavia a narodnosti)
        |
        v
Namnozene simulovane data ulozene do PostgreSQL
        |
        v
XGBoost klasifikacny model (hr_full_app.py)
        |
        v
XGBoost regresivny model (hr_full_app.py)
        |
        v
Zapis vysledkov z modelov do PostgreSQL
        |
        v
Spustenie frontendu cedz Gradio
```

## 3. Obsah
```text
1. POSTGRESQL ENGINE & DATA LOAD
      a. Database Helpers
2. DATA PREPROCESSING
3. TRAINING LOGIC
      a. Optuna optimization
4. CATEGORIZATION BY RISK THRESHOLD, CLIPPING IRRELEVANT DATA
5. ANOMALY DETECTION + FEATURE IMPORTANCE
6. GRADIO FRONTEND
      a. KPI Prep-work
      b. KPI anomalies
      c. Heatmap definition
      d. Callbacks
      e. UI Definition
      f. TAB 1: New employee simulator
      g. TAB 2: Anomalies dashboard
7. MAIN FUNCTION
```
