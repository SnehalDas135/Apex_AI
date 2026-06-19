"""
APEX AI — compare_models_v2.py
================================
Expanded model benchmark for F1 tyre degradation prediction.
Tests 15 models across: Linear, Tree-based, Boosting, SVM, KNN,
Neural (MLP, LSTM, GRU, 1D-CNN) families — per compound + overall.

Run:
    python compare_models_v2.py

Output:  per-compound tables  +  grand summary table at the end.
Paste the terminal output and send it back for the final analysis.
"""

import fastf1
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

# ── Sklearn ────────────────────────────────────────────────────
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Boosting libraries ─────────────────────────────────────────
import xgboost as xgb
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[WARN] LightGBM not installed — skipping. Install: pip install lightgbm")

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("[WARN] CatBoost not installed — skipping. Install: pip install catboost")

# ── Deep Learning ──────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Conv1D, Flatten,
    MaxPooling1D, Dropout, Input, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping
tf.get_logger().setLevel('ERROR')

# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  APEX AI — Model Comparison v2")
print("  F1 2023 Bahrain GP | Driver: VER | Tyre Degradation")
print("═" * 60)

fastf1.Cache.enable_cache('f1_cache')
session = fastf1.get_session(2023, 'Bahrain', 'R')
session.load()

laps = session.laps
ver = laps.pick_drivers('VER').copy()
ver['LapTimeSeconds'] = ver['LapTime'].dt.total_seconds()
ver_clean = ver[ver['LapTimeSeconds'] < 105].copy()
ver_clean = ver_clean.dropna(subset=['TyreLife', 'LapTimeSeconds'])

print(f"\n✓ Data loaded: {len(ver_clean)} clean laps across "
      f"{ver_clean['Compound'].nunique()} compounds\n")

# ══════════════════════════════════════════════════════════════
# HELPER — DEEP LEARNING BUILDERS
# ══════════════════════════════════════════════════════════════

ES = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=0)

def build_lstm(input_shape=(1, 1)):
    m = Sequential([
        LSTM(32, input_shape=input_shape, return_sequences=False),
        Dense(16, activation='relu'),
        Dense(1)
    ], name='LSTM')
    m.compile(optimizer='adam', loss='mse')
    return m

def build_gru(input_shape=(1, 1)):
    m = Sequential([
        GRU(32, input_shape=input_shape),
        Dense(16, activation='relu'),
        Dense(1)
    ], name='GRU')
    m.compile(optimizer='adam', loss='mse')
    return m

def build_cnn1d(input_shape=(1, 1)):
    m = Sequential([
        Conv1D(32, kernel_size=1, activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(1)
    ], name='CNN-1D')
    m.compile(optimizer='adam', loss='mse')
    return m

# ══════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ══════════════════════════════════════════════════════════════

def get_sklearn_models():
    models = {
        # ── Linear family ─────────────────────────────────────
        'Linear Regression'  : LinearRegression(),
        'Ridge'              : Ridge(alpha=1.0),
        'Lasso'              : Lasso(alpha=0.01, max_iter=5000),
        'ElasticNet'         : ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        'Huber Regressor'    : HuberRegressor(epsilon=1.35, max_iter=500),
        # ── Tree family ───────────────────────────────────────
        'Decision Tree'      : DecisionTreeRegressor(max_depth=4, random_state=42),
        'Random Forest'      : RandomForestRegressor(n_estimators=100, max_depth=6,
                                                     random_state=42, n_jobs=-1),
        'Extra Trees'        : ExtraTreesRegressor(n_estimators=100, max_depth=6,
                                                   random_state=42, n_jobs=-1),
        'Bagging'            : BaggingRegressor(n_estimators=50, random_state=42,
                                                n_jobs=-1),
        # ── Boosting ──────────────────────────────────────────
        'Gradient Boosting'  : GradientBoostingRegressor(n_estimators=100,
                                                         max_depth=3, learning_rate=0.1,
                                                         random_state=42),
        'AdaBoost'           : AdaBoostRegressor(n_estimators=50, random_state=42),
        'XGBoost'            : xgb.XGBRegressor(n_estimators=100, max_depth=4,
                                                 learning_rate=0.05, verbosity=0,
                                                 random_state=42),
        # ── SVM / KNN ─────────────────────────────────────────
        'SVR (RBF)'          : SVR(kernel='rbf', C=10, epsilon=0.1),
        'SVR (Poly)'         : SVR(kernel='poly', degree=3, C=5, epsilon=0.1),
        'KNN'                : KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        # ── MLP ───────────────────────────────────────────────
        'MLP'                : MLPRegressor(hidden_layer_sizes=(64, 32),
                                            activation='relu', max_iter=500,
                                            random_state=42),
    }
    if HAS_LGB:
        models['LightGBM'] = lgb.LGBMRegressor(n_estimators=100, max_depth=4,
                                                learning_rate=0.05, verbose=-1,
                                                random_state=42)
    if HAS_CAT:
        models['CatBoost'] = CatBoostRegressor(iterations=100, depth=4,
                                               learning_rate=0.05, verbose=0,
                                               random_state=42)
    return models

DEEP_MODELS = ['LSTM', 'GRU', 'CNN-1D']

# ══════════════════════════════════════════════════════════════
# BENCHMARK LOOP
# ══════════════════════════════════════════════════════════════

all_results = []   # one dict per (compound, model)

for compound in ver_clean['Compound'].unique():
    stint = ver_clean[ver_clean['Compound'] == compound].copy()
    if len(stint) < 5:
        continue

    X = stint[['TyreLife']].values
    y = stint['LapTimeSeconds'].values
    n = len(stint)

    print(f"\n{'═' * 60}")
    print(f"  COMPOUND: {compound}   ({n} laps)")
    print(f"{'═' * 60}")
    print(f"\n  {'Model':<22} {'MAE (s)':>9} {'RMSE (s)':>10} {'R²':>8} {'Time (ms)':>12}")
    print(f"  {'─'*22} {'─'*9} {'─'*10} {'─'*8} {'─'*12}")

    # ── Sklearn / Boosting models ──────────────────────────────
    sklearn_models = get_sklearn_models()
    for name, model in sklearn_models.items():
        t0 = time.time()
        model.fit(X, y)
        elapsed = (time.time() - t0) * 1000

        preds = model.predict(X)
        mae  = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2   = r2_score(y, preds)

        print(f"  {name:<22} {mae:>9.4f} {rmse:>10.4f} {r2:>8.4f} {elapsed:>11.1f}")
        all_results.append({
            'Compound': compound, 'Model': name,
            'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Time_ms': elapsed
        })

    # ── Deep Learning models ───────────────────────────────────
    sx = MinMaxScaler(); sy = MinMaxScaler()
    X_s = sx.fit_transform(X)
    y_s = sy.fit_transform(y.reshape(-1, 1))
    X_seq = X_s.reshape(n, 1, 1)   # (samples, timesteps, features)

    dl_builders = {
        'LSTM'   : build_lstm,
        'GRU'    : build_gru,
        'CNN-1D' : build_cnn1d,
    }
    for name, builder in dl_builders.items():
        model = builder(input_shape=(1, 1))
        t0 = time.time()
        model.fit(X_seq, y_s, epochs=150, batch_size=max(2, n // 4),
                  callbacks=[ES], verbose=0)
        elapsed = (time.time() - t0) * 1000

        preds_s = model.predict(X_seq, verbose=0)
        preds   = sy.inverse_transform(preds_s).flatten()
        mae  = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2   = r2_score(y, preds)

        print(f"  {name:<22} {mae:>9.4f} {rmse:>10.4f} {r2:>8.4f} {elapsed:>11.1f}")
        all_results.append({
            'Compound': compound, 'Model': name,
            'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Time_ms': elapsed
        })

# ══════════════════════════════════════════════════════════════
# GRAND SUMMARY — averaged across all compounds
# ══════════════════════════════════════════════════════════════

df = pd.DataFrame(all_results)
summary = (
    df.groupby('Model')
      .agg(
          Avg_MAE   = ('MAE',     'mean'),
          Avg_RMSE  = ('RMSE',    'mean'),
          Avg_R2    = ('R2',      'mean'),
          Avg_Time  = ('Time_ms', 'mean'),
      )
      .sort_values('Avg_MAE')
      .reset_index()
)

print(f"\n\n{'═' * 75}")
print("  GRAND SUMMARY — AVERAGE ACROSS ALL COMPOUNDS (sorted by MAE ↑)")
print(f"{'═' * 75}")
print(f"\n  {'#':<4} {'Model':<22} {'Avg MAE':>9} {'Avg RMSE':>10} {'Avg R²':>8} {'Avg Time':>11}")
print(f"  {'─'*4} {'─'*22} {'─'*9} {'─'*10} {'─'*8} {'─'*11}")

for i, row in summary.iterrows():
    rank  = i + 1
    medal = " ★" if rank == 1 else ("  " if rank > 3 else "  •")
    print(f"  {rank:<4} {row['Model']:<22} {row['Avg_MAE']:>9.4f} "
          f"{row['Avg_RMSE']:>10.4f} {row['Avg_R2']:>8.4f} "
          f"{row['Avg_Time']:>9.1f}ms{medal}")

# ── Category winners ──────────────────────────────────────────
print(f"\n{'─' * 75}")
print("  CATEGORY WINNERS")
print(f"{'─' * 75}")

categories = {
    'Best MAE (accuracy)'    : ('Avg_MAE',  True ),
    'Best RMSE'              : ('Avg_RMSE', True ),
    'Best R² (fit quality)'  : ('Avg_R2',   False),
    'Fastest training'       : ('Avg_Time', True ),
}
for label, (col, asc) in categories.items():
    winner = summary.sort_values(col, ascending=asc).iloc[0]
    val    = winner[col]
    unit   = 'ms' if 'Time' in col else ('s' if 'MAE' in col or 'RMSE' in col else '')
    print(f"  {label:<28} → {winner['Model']:<22}  ({val:.4f}{unit})")

print(f"\n{'═' * 75}")
print("  Done. Copy the full terminal output and send it back for analysis.")
print(f"{'═' * 75}\n")
