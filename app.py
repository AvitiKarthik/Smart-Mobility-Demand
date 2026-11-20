
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. LOAD MODEL AND DATASET
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "model.pkl")

FEATURES = list(model.feature_names_in_)

DATA_PATH = BASE_DIR / "hour.csv"

app = FastAPI(
    title="BikePulse API",
    description="Smart Mobility Demand Forecasting API",
    version="1.0.0"
)


# --------------------------------------------------
# 2. TEMPORARY CUSTOMER PREDICTION STORAGE
# --------------------------------------------------

customer_predictions = defaultdict(list)


# --------------------------------------------------
# 3. INPUT VALIDATION
# --------------------------------------------------

class PredictionRequest(BaseModel):

    customer_id: str = "guest"

    date: str

    hour: int = Field(ge=0, le=23)

    season: int = Field(ge=1, le=4)

    temp: float = Field(ge=0, le=41)

    humidity: float = Field(ge=0, le=100)

    windspeed: float = Field(ge=0, le=67)

    weather: int = Field(ge=1, le=4)

    workingday: int = Field(ge=0, le=1)

    holiday: int = Field(default=0, ge=0, le=1)


# --------------------------------------------------
# 4. FEATURE ENGINEERING
# --------------------------------------------------

def create_features(data, hours, workingday=None):

    try:
        dt = datetime.strptime(data.date, "%Y-%m-%d")

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Date must be YYYY-MM-DD"
        )

    wd = data.workingday if workingday is None else workingday

    if wd == 1 and data.holiday == 1:
        raise HTTPException(
            status_code=400,
            detail="A holiday cannot be a working day"
        )

    rows = []

    for h in hours:

        rows.append({

            "season": data.season,

            "mnth": dt.month,

            "hr": h,

            "holiday": data.holiday,

            "weekday": (dt.weekday() + 1) % 7,

            "workingday": wd,

            "weathersit": data.weather,

            "temp": data.temp / 41,

            "hum": data.humidity / 100,

            "windspeed": data.windspeed / 67

        })

    return pd.DataFrame(rows)[FEATURES]


# --------------------------------------------------
# 5. HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "project": "BikePulse",
        "description": "Smart Mobility Demand Forecasting API",
        "endpoints": [
            "/predict",
            "/health",
            "/dashboard/customer",
            "/dashboard/investor",
            "/docs"
        ]
    }


# --------------------------------------------------
# 6. HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "database": "Not required"
    }


# --------------------------------------------------
# 7. CUSTOMER DEMAND PREDICTION
# --------------------------------------------------

@app.post("/predict")
def predict(data: PredictionRequest):

    X = create_features(data, range(24))

    predictions = model.predict(X)

    hourly = [
        max(0, round(float(value)))
        for value in predictions
    ]

    # Alternative working-day scenario
    alternative_wd = 1 - data.workingday

    alternative = model.predict(
        create_features(
            data,
            [data.hour],
            workingday=alternative_wd
        )
    )

    alt_prediction = max(
        0,
        round(float(alternative[0]))
    )

    peak_hour = hourly.index(max(hourly))

    low_hour = min(
        range(6, 23),
        key=lambda h: hourly[h]
    )

    result = {

        "customer_id": data.customer_id,

        "date": data.date,

        "predicted_demand": hourly[data.hour],

        "hourly_predictions": hourly,

        "peak_hour": peak_hour,

        "low_hour": low_hour,

        "alternative_prediction": alt_prediction

    }

    # Keep only the latest 100 records per customer
    customer_predictions[data.customer_id].append({

        **result,

        "created_at": datetime.now().isoformat()

    })

    customer_predictions[data.customer_id] = (
        customer_predictions[data.customer_id][-100:]
    )

    return result


# --------------------------------------------------
# 8. CUSTOMER DASHBOARD
# --------------------------------------------------

@app.get("/dashboard/customer")
def customer_dashboard(customer_id: str = "guest"):

    records = customer_predictions.get(customer_id, [])

    if not records:

        return {
            "customer_id": customer_id,
            "total_predictions": 0,
            "average_predicted_demand": 0,
            "recent_predictions": []
        }

    average = sum(
        record["predicted_demand"]
        for record in records
    ) / len(records)

    return {

        "customer_id": customer_id,

        "total_predictions": len(records),

        "average_predicted_demand": round(average, 2),

        "recent_predictions": records[-10:][::-1]

    }


# --------------------------------------------------
# 9. INVESTOR ANALYTICS
# --------------------------------------------------

@app.get("/dashboard/investor")
def investor_dashboard():

    if not DATA_PATH.exists():

        raise HTTPException(
            status_code=500,
            detail="hour.csv dataset not found"
        )

    df = pd.read_csv(DATA_PATH)

    required = [
        "season",
        "weathersit",
        "workingday",
        "hr",
        "cnt"
    ]

    if not set(required).issubset(df.columns):

        raise HTTPException(
            status_code=500,
            detail="Dataset is missing required columns"
        )

    def grouped_analysis(column):

        result = (
            df.groupby(column)["cnt"]
            .agg(["mean", "sum", "count"])
            .reset_index()
        )

        return result.rename(columns={
            "mean": "average_demand",
            "sum": "total_rentals",
            "count": "records"
        }).round(2).to_dict(orient="records")

    return {

        "data_source": "Historical hour.csv dataset",

        "total_records": int(len(df)),

        "total_historical_rentals": int(df["cnt"].sum()),

        "average_hourly_demand": round(
            float(df["cnt"].mean()), 2
        ),

        "season_analysis": grouped_analysis("season"),

        "weather_analysis": grouped_analysis("weathersit"),

        "workingday_analysis": grouped_analysis("workingday"),

        "hourly_analysis": grouped_analysis("hr")

    }
