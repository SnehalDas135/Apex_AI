import fastf1
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# Load cached data (already downloaded so will be instant)
fastf1.Cache.enable_cache('f1_cache')
session = fastf1.get_session(2023, 'Bahrain', 'R')
session.load()

laps = session.laps
ver = laps.pick_drivers('VER')

# Convert LapTime from timedelta to plain seconds
ver = ver.copy()
ver['LapTimeSeconds'] = ver['LapTime'].dt.total_seconds()

# Remove pit laps — they spike to 1:58 and will confuse the model
# Any lap over 105 seconds is a pit lap or safety car, remove it
ver_clean = ver[ver['LapTimeSeconds'] < 105].copy()

# Drop rows where TyreLife or LapTime is missing
ver_clean = ver_clean.dropna(subset=['TyreLife', 'LapTimeSeconds'])

# Input (X) = TyreLife
# Output (y) = LapTime in seconds
X = ver_clean[['TyreLife']]
y = ver_clean['LapTimeSeconds']

# Train the model using XGBoost
model = xgb.XGBRegressor(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    verbosity=0
)
model.fit(X, y)

# See how accurate it is on the same data
predictions = model.predict(X)
mae = mean_absolute_error(y, predictions)
print(f"Mean Absolute Error: {mae:.3f} seconds")

# Now ask it to predict lap times for TyreLife 1 through 25
print("\nPredicted lap times by tyre age:")
for tyre_age in range(1, 26):
    predicted = model.predict([[tyre_age]])[0]
    print(f"TyreLife {tyre_age:2d} → {predicted:.3f} seconds")
