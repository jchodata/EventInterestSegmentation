import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
from xgboost import cv
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

def create_dataframe(csv_file):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
    return df

def binarize_positive(df_input, column_names):
    df = df_input.copy()
    for column in column_names:
        if column in df.columns:
            df[column] = (pd.to_numeric(df[column], errors='coerce').fillna(0) > 0).astype(int)
    return df

def clean_currency(df_input, col_name='lifetime_raised'):
    df = df_input.copy()
    if col_name in df.columns:
        df[col_name] = (
            df[col_name].astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
        )
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    return df

def multiple_degrees(df_input, first_degree_col='first_degree', last_degree_col='last_degree', new_col_name="multiple_degree"):
    df = df_input.copy()
    if first_degree_col in df.columns and last_degree_col in df.columns:
        df[new_col_name] = (df[first_degree_col] != df[last_degree_col]).astype(int)
    return df

def event_count_to_segment(n):
    if n == 0:
        return 0  # Low
    elif n <= 2:
        return 1  # Moderate
    elif n <= 5:
        return 2  # High
    else:
        return 3  # Very High

# UPDATE THIS PATH
csv_file_path = r'D:\Downloads\event_model_sample_v2.csv'

df = create_dataframe(csv_file_path)
if df is None:
    raise SystemExit(1)

df = binarize_positive(df, ['belongs_to_groups', 'belongs_to_household', 'educational_involvement'])
df = clean_currency(df, col_name='lifetime_raised')
df = multiple_degrees(df, first_degree_col='first_degree', last_degree_col='last_degree')

# Target/Features
target_variable = 'events_target_year'
if target_variable not in df.columns:
    raise ValueError(f"Missing required target column: {target_variable}")

# Drop non-features
drop_cols = [target_variable, 'person_id', 'first_degree', 'last_degree', 'event_count_all', 'event_count_5years']
X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
y = df[target_variable]

# Convert counts -> segment labels
y = y.apply(event_count_to_segment).astype(int)

print(y.value_counts().sort_index())

# Encode categoricals with category codes (style matched to original)
for col in X.select_dtypes(include=['category', 'object']):
    X[col] = X[col].astype('category').cat.codes

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify=y)

# Sample weights for class balance
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# XGBoost parameters (multi-class)
params = {
    "objective": "multi:softprob",
    "n_estimators": 1000,
    "verbosity": 0,
    "learning_rate": 0.05,
    "max_depth": 7,
    "subsample": 0.7,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
    "random_state": 0
}

# Instantiate and fit
xgb_clf = XGBClassifier(**params)
xgb_clf.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

y_pred = xgb_clf.predict(X_test)

print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# Feature importance plot (saved to file)
fig, ax = plt.subplots(figsize=(10, 14))
plot_importance(xgb_clf, ax=ax, max_num_features=30)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=160)
plt.close(fig)

# Create Segment on full dataset
y_all_pred = xgb_clf.predict(X)

segment_map = {
    0: 'Low',
    1: 'Moderate',
    2: 'High',
    3: 'Very High'
}
segments = [segment_map.get(int(pred), 'Unknown') for pred in y_all_pred]

df_out = df.copy()
df_out['Event Interest Segment'] = y_all_pred
df_out['Segment'] = segments

segment_counts = df_out['Segment'].value_counts()
print(segment_counts)

# Excel and CSV outputs
# Excel requires openpyxl; CSV is always safe
df_out.to_excel('constituent_segments.xlsx', index=False)
df_out.to_csv('constituent_segments.csv', index=False)

# Weights summary (optional diagnostic saved to CSV)
df_train = X_train.copy()
df_train['Segment Label'] = y_train
df_train['Sample Weight'] = sample_weights
weights_summary = df_train.groupby('Segment Label')['Sample Weight'].describe()
weights_summary.to_csv('sample_weights_by_segment.csv', index=True)


# (Optional) Optuna — disabled by default to keep deps minimal.
# To use: install optuna and set RUN_OPTUNA = True.

RUN_OPTUNA = False

if RUN_OPTUNA:
    from sklearn.metrics import accuracy_score
    import optuna

    def objective(trial):
        params_t = {
            "objective": "multi:softprob",
            "n_estimators": 800,
            "verbosity": 0,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "tree_method": "hist",
            "eval_metric": "mlogloss",
            "random_state": 0,
        }
        model = XGBClassifier(**params_t)
        model.fit(X_train, y_train, verbose=False)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        return acc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    print('Best hyperparameters:', study.best_params)
    print('Best accuracy model:', study.best_value)
