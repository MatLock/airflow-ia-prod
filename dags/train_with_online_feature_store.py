import os
import logging

from sqlalchemy import create_engine, text
from airflow.sdk import dag, task
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import mlflow.sklearn


MYSQL_CONN_STRING = (
    f"mysql+mysqldb://{os.environ['MYSQL_USER']}:{os.environ['MYSQL_PASSWORD']}"
    f"@{os.environ['MYSQL_HOST']}:{os.environ['MYSQL_PORT']}/{os.environ['MYSQL_DATABASE']}"
)
GET_LATEST_FS_VERSION = 'SELECT latest_feature_store_preffix FROM fs_metadata'
TARGET = 'prod_pet'
FEATURE_COLS = [
    "tipoextraccion",
    "avg_prod_gas_10m",
    "avg_prod_pet_10m",
    "last_prod_gas",
    "last_prod_pet",
    "n_readings",
]


@task
def train_with_online_feature_store():
  engine = create_engine(MYSQL_CONN_STRING)

  with engine.connect() as conn:
    result = conn.execute(text(GET_LATEST_FS_VERSION)).fetchone()
    if result is None:
      raise ValueError("No feature store version found in fs_metadata")
    fs_version = result[0]

  table_name = f"{fs_version}_feature_store"
  logging.info(f"Reading from table {table_name}")
  df = pd.read_sql_table(table_name, engine)
  df = df.dropna(subset=[TARGET])

  X = df[FEATURE_COLS]
  y = df[TARGET]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  for n_estimators in [25, 50, 100]:
    with mlflow.start_run(run_name=f"rf_{TARGET}_n{n_estimators}"):
      model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
      model.fit(X_train, y_train)

      y_pred = model.predict(X_test)
      mae = mean_absolute_error(y_test, y_pred)
      mse = mean_squared_error(y_test, y_pred)
      r2 = r2_score(y_test, y_pred)

      mlflow.log_param("target", TARGET)
      mlflow.log_param("n_estimators", n_estimators)
      mlflow.log_param("feature_store_version", fs_version)
      mlflow.log_param("feature_store_table", table_name)
      mlflow.log_metric("mae", mae)
      mlflow.log_metric("mse", mse)
      mlflow.log_metric("r2", r2)
      mlflow.sklearn.log_model(model, artifact_path=f"rf_{TARGET}_n{n_estimators}")
      logging.info(f"rf_n{n_estimators} — MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")


@dag(
    dag_id='train_with_online_feature_store',
    description='Train RF models using the latest online feature store',
    schedule=None,
)
def train_dag():
  train_with_online_feature_store()

train_dag()
