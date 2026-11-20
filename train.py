# Trains the Random Forest on hour.csv and saves model.pkl (chronological split, baseline comparison)
import pandas as pd, joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
F = ["season","mnth","hr","holiday","weekday","workingday","weathersit","temp","hum","windspeed"]
df = pd.read_csv("hour.csv"); cut = int(len(df) * 0.8)
tr, te = df.iloc[:cut], df.iloc[cut:]
base = tr.groupby("hr").cnt.mean()
print("Baseline MAE (mean per hour):", round(mean_absolute_error(te.cnt, te.hr.map(base)), 1))
rf = RandomForestRegressor(n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=42).fit(tr[F], tr.cnt)
print("Random Forest MAE:", round(mean_absolute_error(te.cnt, rf.predict(te[F])), 1))
rf.fit(df[F], df.cnt); joblib.dump(rf, "model.pkl", compress=3); print("Saved model.pkl")
