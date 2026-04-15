import os
import logging
from datetime import date
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from mlflow.tracking import MlflowClient
from pydantic import BaseModel
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MYSQL_CONN_STRING = (
    f"mysql+pymysql://{os.environ['MYSQL_USER']}:{os.environ['MYSQL_PASSWORD']}"
    f"@{os.environ['MYSQL_HOST']}:{os.environ['MYSQL_PORT']}/{os.environ['MYSQL_DATABASE']}"
)
DB_ENGINE = create_engine(MYSQL_CONN_STRING)
MODEL_NAME = "rf_prod_pet"
MODEL_ALIAS = "production"
FEATURE_COLS = [
    "tipoextraccion",
    "avg_prod_gas_10m",
    "avg_prod_pet_10m",
    "last_prod_gas",
    "last_prod_pet",
    "n_readings",
]

# Pydantic models
class ModelMetrics(BaseModel):
    mae: Optional[float]
    mse: Optional[float]
    r2: Optional[float]


class ModelMetadataResponse(BaseModel):
    model_name: str
    model_version: str
    alias: str
    run_id: str
    feature_store_version: Optional[str]
    feature_store_table: Optional[str]
    target: Optional[str]
    n_estimators: Optional[int]
    metrics: ModelMetrics


class ForecastPoint(BaseModel):
    date: date
    prod: float


class ForecastResponse(BaseModel):
    id_well: str
    data: List[ForecastPoint]


class WellItem(BaseModel):
    id_well: str


# App & lazy-loaded state
app = FastAPI(
    title="Oil & Gas Forecast API",
    version="1.0.0",
    description="API para consultar el listado de pozos y sus pronósticos de producción.",
)


def get_model():
    """Load the production model and its metadata from MLflow on first call and cache both."""
    logger.info(f"Loading model models:/{MODEL_NAME}@{MODEL_ALIAS} from MLflow...")
    try:
        client = MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        run = client.get_run(mv.run_id)

        _model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
        logger.info(f"Run tags: {run.data.tags}")
        _model_metadata = {
            "model_name": MODEL_NAME,
            "model_version": mv.version,
            "alias": MODEL_ALIAS,
            "run_id": mv.run_id,
            "feature_store_version": run.data.tags.get("feature_store_version"),
            "feature_store_table": run.data.tags.get("feature_store_table"),
            "target": run.data.params.get("target"),
            "n_estimators": int(run.data.params.get("n_estimators", 0)),
            "metrics": {
                "mae": run.data.metrics.get("mae"),
                "mse": run.data.metrics.get("mse"),
                "r2": run.data.metrics.get("r2"),
            },
        }
        logger.info(f"Model loaded: version={mv.version} run_id={mv.run_id}")
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}")
        raise HTTPException(
            status_code=503,
            detail=f"Model not available. Make sure the training pipeline has run: {exc}",
        )
    return _model, _model_metadata


# Endpoints
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/v1/model", response_model=ModelMetadataResponse)
def get_model_metadata():
    """
    Return metadata of the current production model.

    Exposes which feature store version was used for training, the MLflow run ID
    for full reproducibility, and the evaluation metrics that determined promotion.
    """
    production_ml_model, production_ml_model_metadata =  get_model()
    if production_ml_model is None:
        raise HTTPException(status_code=503, detail="Model metadata not available")
    return ModelMetadataResponse(
        model_name=production_ml_model_metadata["model_name"],
        model_version=str(production_ml_model_metadata["model_version"]),
        alias=production_ml_model_metadata["alias"],
        run_id=production_ml_model_metadata["run_id"],
        feature_store_version=production_ml_model_metadata["feature_store_version"],
        feature_store_table=production_ml_model_metadata["feature_store_table"],
        target=production_ml_model_metadata["target"],
        n_estimators=production_ml_model_metadata["n_estimators"],
        metrics=ModelMetrics(**production_ml_model_metadata["metrics"]),
    )


@app.get("/api/v1/wells", response_model=List[WellItem])
def get_wells(
    date_query: date = Query(..., description="Fecha para la cual se hace la consulta (YYYY-MM-DD)"),
):
    """Return distinct wells that have production records on or before date_query."""
    _, production_ml_model_metadata = get_model()
    table_name = production_ml_model_metadata["feature_store_table"]

    query = text(
        f"SELECT DISTINCT idpozo FROM `{table_name}` WHERE DATE(fecha) <= :date_query"
    )
    with DB_ENGINE.connect() as conn:
        rows = conn.execute(query, {"date_query": date_query}).fetchall()

    return [{"id_well": str(row[0])} for row in rows]


@app.get("/api/v1/forecast", response_model=ForecastResponse)
def get_forecast(
    id_well: str = Query(..., description="Identificador del pozo"),
    date_start: date = Query(..., description="Fecha de inicio (YYYY-MM-DD)"),
    date_end: date = Query(..., description="Fecha de fin (YYYY-MM-DD)"),
):
    """
    Return the production forecast for a well between date_start and date_end.

    Features are taken from the well's online store record (the pre-computed row
    for the next inference period). The same prediction is returned for every
    month in the requested date range, since the model produces a single forward
    estimate based on the most recent feature snapshot.
    """
    if date_start > date_end:
        raise HTTPException(status_code=400, detail="date_start must be <= date_end")

    production_ml_model, production_ml_model_metadata =  get_model()
    table_name = production_ml_model_metadata["feature_store_table"]

    cols = ", ".join(FEATURE_COLS)

    query_online = text(
        f"SELECT {cols} FROM `{table_name}` "
        "WHERE idpozo = :well_id AND prod_pet IS NULL "
        "ORDER BY fecha DESC LIMIT 1"
    )
    # Fall back to the most recent 10 historical records
    query_hist = text(
        f"SELECT {cols} FROM `{table_name}` "
        "WHERE idpozo = :well_id AND prod_pet IS NOT NULL "
        "ORDER BY fecha DESC LIMIT 10"
    )

    try:
        well_id_param = int(id_well)
    except ValueError:
        well_id_param = id_well

    with DB_ENGINE.connect() as conn:
        row = conn.execute(query_online, {"well_id": well_id_param}).fetchone()
        if row is None:
            hist_rows = conn.execute(query_hist, {"well_id": well_id_param}).fetchall()
        else:
            hist_rows = None

    if row is None and not hist_rows:
        raise HTTPException(status_code=404, detail=f"Well '{id_well}' not found in feature store")

    if row is not None:
        # Online store row available
        features = dict(zip(FEATURE_COLS, row))
        recent_pet = [features["avg_prod_pet_10m"]] * 10
    else:
        # Use latest historical record as feature base, last 10 for rolling avg
        features = dict(zip(FEATURE_COLS, hist_rows[0]))
        recent_pet = [float(r[FEATURE_COLS.index("last_prod_pet")]) for r in reversed(hist_rows)]
        while len(recent_pet) < 10:
            recent_pet.insert(0, recent_pet[0])
        features["avg_prod_pet_10m"] = sum(recent_pet) / len(recent_pet)

    data: List[ForecastPoint] = []
    current = date(date_start.year, date_start.month, 1)
    end_month = date(date_end.year, date_end.month, 1)
    while current <= end_month:
        df = pd.DataFrame([features], columns=FEATURE_COLS)
        pred_value = float(production_ml_model.predict(df)[0])
        data.append(ForecastPoint(date=current, prod=pred_value))

        # Feed prediction back into features for next month
        recent_pet.append(pred_value)
        recent_pet.pop(0)
        features["last_prod_pet"] = pred_value
        features["avg_prod_pet_10m"] = sum(recent_pet) / len(recent_pet)
        features["n_readings"] = features["n_readings"] + 1

        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

    return ForecastResponse(id_well=id_well, data=data)
