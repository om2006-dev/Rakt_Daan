import math
from datetime import date
from pathlib import Path
from typing import Mapping

import joblib
import pandas as pd
from flask import Flask, render_template, request


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "blood_demand_random_forest.joblib"
DATA_PATH = APP_DIR / "blood_bank_data.csv"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_columns = bundle["feature_columns"]
data = pd.read_csv(DATA_PATH)

history_metrics = (
    "units_requested_month",
    "units_transfused",
    "units_expired",
    "units_collected_month",
)
lag_periods = (1, 12)

app = Flask(__name__)


def _choices(column: str) -> list[str]:
    return sorted(data[column].dropna().astype(str).unique().tolist())


def _build_features(form: Mapping[str, str]) -> pd.DataFrame:
    facility_level = form.get("facility_level", "")
    facility_id = form.get("facility_id", "")
    product_name = form.get("product_name", "")
    blood_group = form.get("blood_group", "")

    allowed_values = {
        "facility_level": _choices("facility_level"),
        "facility_id": _choices("facility_id"),
        "product_name": _choices("product_name"),
        "blood_group": _choices("blood_group"),
    }
    for name, value in (
        ("facility_level", facility_level),
        ("facility_id", facility_id),
        ("product_name", product_name),
        ("blood_group", blood_group),
    ):
        if value not in allowed_values[name]:
            raise ValueError(f"Choose a valid {name.replace('_', ' ')}.")

    try:
        month = int(form.get("month", ""))
    except ValueError as exc:
        raise ValueError("Choose a valid prediction month.") from exc
    if month < 1 or month > 12:
        raise ValueError("Choose a valid prediction month.")

    try:
        year = int(form.get("year", ""))
    except ValueError as exc:
        raise ValueError("Enter a valid prediction year.") from exc
    if year < 2000 or year > 2100:
        raise ValueError("Enter a prediction year between 2000 and 2100.")

    fridge_value = form.get("has_blood_bank_fridge", "")
    if fridge_value not in {"0", "1"}:
        raise ValueError("Choose whether the facility has a blood bank fridge.")

    row: dict[str, str | int | float] = {
        "facility_level": facility_level,
        "facility_code": f"{facility_level}::{facility_id}",
        "product_name": product_name,
        "blood_group": blood_group,
        "year": year,
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
        "has_blood_bank_fridge": int(fridge_value),
    }

    has_previous_month_record = False
    has_previous_year_record = False
    for lag in lag_periods:
        for metric in history_metrics:
            name = f"{metric}_lag_{lag}"
            raw_value = form.get(name, "").strip()
            if not raw_value:
                row[name] = 0.0
                continue
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"{name.replace('_', ' ')} must be a number.") from exc
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name.replace('_', ' ')} must be zero or greater.")
            row[name] = value
            if metric == "units_requested_month":
                if lag == 1:
                    has_previous_month_record = True
                else:
                    has_previous_year_record = True

    row["has_previous_month_record"] = int(has_previous_month_record)
    row["has_previous_year_record"] = int(has_previous_year_record)
    return pd.DataFrame([row])[feature_columns]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    form_data = request.form if request.method == "POST" else {}

    if request.method == "POST":
        try:
            features = _build_features(form_data)
        except ValueError as exc:
            error = str(exc)
        else:
            prediction = max(0.0, float(model.predict(features)[0]))

    today = date.today()
    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        form_data=form_data,
        facility_levels=_choices("facility_level"),
        facility_ids=_choices("facility_id"),
        products=_choices("product_name"),
        blood_groups=_choices("blood_group"),
        default_month=today.month,
        default_year=today.year,
        months=[
            (1, "January"), (2, "February"), (3, "March"), (4, "April"),
            (5, "May"), (6, "June"), (7, "July"), (8, "August"),
            (9, "September"), (10, "October"), (11, "November"), (12, "December"),
        ],
        history_metrics=history_metrics,
    ), (400 if error else 200)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
