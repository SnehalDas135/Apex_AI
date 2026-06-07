import fastf1
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ── Data Loading ──────────────────────────────────────────────
fastf1.Cache.enable_cache('f1_cache')
session = fastf1.get_session(2023, 'Bahrain', 'R')
session.load()

laps = session.laps
ver = laps.pick_drivers('VER').copy()
ver['LapTimeSeconds'] = ver['LapTime'].dt.total_seconds()
ver_clean = ver[ver['LapTimeSeconds'] < 105].copy()
ver_clean = ver_clean.dropna(subset=['TyreLife', 'LapTimeSeconds'])

results = []

# ── Run Per Compound ───────────────────────────────────────────
for compound in ver_clean['Compound'].unique():
    stint = ver_clean[ver_clean['Compound'] == compound].copy()
    if len(stint) < 5:
        continue

    X = stint[['TyreLife']].values
    y = stint['LapTimeSeconds'].values

    print(f"\n{'='*50}")
    print(f"COMPOUND: {compound} ({len(stint)} laps)")
    print(f"{'='*50}")

    # ── 1. Linear Regression ──────────────────────────────────
    start = time.time()
    lr = LinearRegression()
    lr.fit(X, y)
    lr_time = (time.time() - start) * 1000  # ms
    lr_mae = mean_absolute_error(y, lr.predict(X))
    lr_deg = lr.coef_[0]

    print(f"\n[Linear Regression]")
    print(f"  Training time : {lr_time:.3f} ms")
    print(f"  MAE           : {lr_mae:.4f} seconds")
    print(f"  Degradation   : {lr_deg:.4f} sec/lap")

    # ── 2. XGBoost ────────────────────────────────────────────
    start = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        verbosity=0
    )
    xgb_model.fit(X, y)
    xgb_time = (time.time() - start) * 1000
    xgb_mae = mean_absolute_error(y, xgb_model.predict(X))

    print(f"\n[XGBoost]")
    print(f"  Training time : {xgb_time:.3f} ms")
    print(f"  MAE           : {xgb_mae:.4f} seconds")

    # ── 3. LSTM ───────────────────────────────────────────────
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # Reshape for LSTM [samples, timesteps, features]
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, 1))

    start = time.time()
    lstm_model = Sequential([
        LSTM(16, input_shape=(1, 1)),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_lstm, y_scaled, epochs=100, verbose=0)
    lstm_time = (time.time() - start) * 1000

    y_pred_scaled = lstm_model.predict(X_lstm, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    lstm_mae = mean_absolute_error(y, y_pred)

    print(f"\n[LSTM]")
    print(f"  Training time : {lstm_time:.3f} ms")
    print(f"  MAE           : {lstm_mae:.4f} seconds")

    # ── Summary Table ─────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"{'Model':<20} {'MAE (s)':<15} {'Train Time (ms)'}")
    print(f"{'─'*50}")
    print(f"{'Linear Regression':<20} {lr_mae:<15.4f} {lr_time:.3f}")
    print(f"{'XGBoost':<20} {xgb_mae:<15.4f} {xgb_time:.3f}")
    print(f"{'LSTM':<20} {lstm_mae:<15.4f} {lstm_time:.3f}")

    results.append({
        'Compound': compound,
        'LR_MAE': lr_mae, 'LR_Time': lr_time,
        'XGB_MAE': xgb_mae, 'XGB_Time': xgb_time,
        'LSTM_MAE': lstm_mae, 'LSTM_Time': lstm_time
    })

# ── Overall Winner ─────────────────────────────────────────────
print(f"\n{'='*50}")
print("OVERALL AVERAGE ACROSS ALL COMPOUNDS")
print(f"{'='*50}")
df = pd.DataFrame(results)
print(f"Linear Regression → MAE: {df['LR_MAE'].mean():.4f}s  |  Time: {df['LR_Time'].mean():.3f}ms")
print(f"XGBoost           → MAE: {df['XGB_MAE'].mean():.4f}s  |  Time: {df['XGB_Time'].mean():.3f}ms")
print(f"LSTM              → MAE: {df['LSTM_MAE'].mean():.4f}s  |  Time: {df['LSTM_Time'].mean():.3f}ms")