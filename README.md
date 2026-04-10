# Oil & Gas Production Prediction Pipeline

End-to-end ML pipeline that ingests Argentina's unconventional oil & gas well production data, engineers features, stores them in a versioned MySQL feature store, trains Random Forest models tracked with MLflow, and promotes the best model to production.

## Architecture

```
Data Source (API) -> Airflow (Ingest DAG) -> MySQL Feature Store -> Airflow (Train DAG) -> MLflow Model Registry
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| Airflow UI | 8080 | DAG orchestration and monitoring |
| MLflow UI | 9191 | Experiment tracking and model registry |
| MySQL | 3307 | Versioned feature store |
| PostgreSQL | 5432 | Airflow metadata database |
| Redis | 6379 | Celery broker |

## DAGs

### `ingest_dag`

Ingestion pipeline that runs the following tasks sequentially:

1. **download_dataset** - Downloads the raw CSV from Argentina's energy open data portal
2. **preprocess** - Cleans data, encodes categorical variables, and engineers features using a 10-reading sliding window per well (rolling averages, last values)
3. **insert_into_online_store** - Versions the feature store (`v1_feature_store`, `v2_feature_store`, ...) and writes processed data to MySQL
4. **delete_raw_data** - Cleans up temporary CSV files
5. **trigger_train_dag** - Triggers the training DAG via `TriggerDagRunOperator`

### `train_with_online_feature_store_and_promote_best_model`

Training and model promotion pipeline:

1. **train_with_online_feature_store** - Reads the latest feature store version from MySQL, trains 3 Random Forest regressors (`n_estimators` = 25, 50, 100) to predict `prod_pet` (oil production), and logs parameters, metrics (MAE, MSE, R2), and model artifacts to MLflow
2. **promote_best_model** - Selects the best model by lowest MSE, registers it in the MLflow model registry, and tags it with the `production` alias

## Feature Store

The feature store is versioned in MySQL. Each ingestion creates a new table (`v1_feature_store`, `v2_feature_store`, etc.) and updates the `fs_metadata` table with the latest version.

### Engineered Features

| Feature | Description |
|---------|-------------|
| `avg_prod_gas_10m` | Average gas production over last 10 readings |
| `avg_prod_pet_10m` | Average oil production over last 10 readings |
| `last_prod_gas` | Most recent gas production reading |
| `last_prod_pet` | Most recent oil production reading |
| `n_readings` | Number of readings in the window |
| `tipoextraccion` | Extraction type (label encoded) |

## Setup

### Prerequisites

- Docker and Docker Compose

### Run

```bash
docker compose up -d
```

### Environment Variables

Configuration is managed via the `.env` file:

| Variable | Description |
|----------|-------------|
| `AIRFLOW_UID` | User ID for Airflow containers |
| `MYSQL_USER` | MySQL username |
| `MYSQL_PASSWORD` | MySQL password |
| `MYSQL_HOST` | MySQL host |
| `MYSQL_PORT` | MySQL port |
| `MYSQL_DATABASE` | MySQL database name |

## Data Source

[Unconventional Oil & Gas Well Production - Argentina Energy Open Data](http://datos.energia.gob.ar/dataset/c846e79c-026c-4040-897f-1ad3543b407c/resource/b5b58cdc-9e07-41f9-b392-fb9ec68b0725/download/produccin-de-pozos-de-gas-y-petrleo-no-convencional.csv)

## Authors

- **Flores Jorge Federico** - jfflores90@gmail.com
- **Nicolas Velazquez** - nicoj.velazquez@gmail.com