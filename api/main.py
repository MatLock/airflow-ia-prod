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
GET_LATEST_FS_VERSION = "SELECT latest_feature_store_preffix FROM fs_metadata"
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

_engine = None
_model = None
_model_metadata: Optional[Dict[str, Any]] = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(MYSQL_CONN_STRING)
    return _engine


def get_model():
    """Load the production model and its metadata from MLflow on first call and cache both."""
    global _model, _model_metadata
    if _model is None:
        logger.info(f"Loading model models:/{MODEL_NAME}@{MODEL_ALIAS} from MLflow...")
        try:
            client = MlflowClient()
            mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            run = client.get_run(mv.run_id)

            _model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
            _model_metadata = {
                "model_name": MODEL_NAME,
                "model_version": mv.version,
                "alias": MODEL_ALIAS,
                "run_id": mv.run_id,
                "feature_store_version": run.data.params.get("feature_store_version"),
                "feature_store_table": run.data.params.get("feature_store_table"),
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
    return _model


def get_latest_fs_table() -> str:
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(GET_LATEST_FS_VERSION)).fetchone()
    if result is None:
        raise HTTPException(
            status_code=503,
            detail="Feature store not initialized. Run the ingest DAG first.",
        )
    return f"{result[0]}_feature_store"


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
    get_model()  # ensures _model_metadata is populated
    if _model_metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not available")
    return ModelMetadataResponse(
        model_name=_model_metadata["model_name"],
        model_version=str(_model_metadata["model_version"]),
        alias=_model_metadata["alias"],
        run_id=_model_metadata["run_id"],
        feature_store_version=_model_metadata.get("feature_store_version"),
        feature_store_table=_model_metadata.get("feature_store_table"),
        target=_model_metadata.get("target"),
        n_estimators=_model_metadata.get("n_estimators"),
        metrics=ModelMetrics(**_model_metadata["metrics"]),
    )


@app.get("/api/v1/wells", response_model=List[WellItem])
def get_wells(
    date_query: date = Query(..., description="Fecha para la cual se hace la consulta (YYYY-MM-DD)"),
):
    """Return distinct wells that have production records on or before date_query."""
    table_name = get_latest_fs_table()
    engine = get_engine()

    query = text(
        f"SELECT DISTINCT idpozo FROM `{table_name}` WHERE DATE(fecha) <= :date_query"
    )
    with engine.connect() as conn:
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

    model = get_model()
    table_name = get_latest_fs_table()
    engine = get_engine()

    cols = ", ".join(FEATURE_COLS)

    query_online = text(
        f"SELECT {cols} FROM `{table_name}` "
        "WHERE idpozo = :well_id AND prod_pet IS NULL "
        "ORDER BY fecha DESC LIMIT 1"
    )
    # Fall back to the most recent historical record
    query_hist = text(
        f"SELECT {cols} FROM `{table_name}` "
        "WHERE idpozo = :well_id AND prod_pet IS NOT NULL "
        "ORDER BY fecha DESC LIMIT 1"
    )

    try:
        well_id_param = int(id_well)
    except ValueError:
        well_id_param = id_well

    with engine.connect() as conn:
        row = conn.execute(query_online, {"well_id": well_id_param}).fetchone()
        if row is None:
            row = conn.execute(query_hist, {"well_id": well_id_param}).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Well '{id_well}' not found in feature store")

    features = pd.DataFrame([dict(zip(FEATURE_COLS, row))], columns=FEATURE_COLS)
    pred_value = float(model.predict(features)[0])

    # Build one data point per calendar month in [date_start, date_end]
    data: List[ForecastPoint] = []
    current = date(date_start.year, date_start.month, 1)
    end_month = date(date_end.year, date_end.month, 1)
    while current <= end_month:
        data.append(ForecastPoint(date=current, prod=pred_value))
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

    return ForecastResponse(id_well=id_well, data=data)
