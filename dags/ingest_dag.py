import os

from airflow.sdk import dag, task
from airflow.api.client.local_client import Client
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine, text

import urllib.request
import pandas as pd
import logging

DATASET_DOWNLOAD_URL = "http://datos.energia.gob.ar/dataset/c846e79c-026c-4040-897f-1ad3543b407c/resource/b5b58cdc-9e07-41f9-b392-fb9ec68b0725/download/produccin-de-pozos-de-gas-y-petrleo-no-convencional.csv"
DATASET_PATH = '/opt/airflow/raw_data/raw_data.csv'
PREPROCESSED_PATH = '/opt/airflow/raw_data/preprocessed_data.csv'
MYSQL_CONN_STRING = (
    f"mysql+mysqldb://{os.environ['MYSQL_USER']}:{os.environ['MYSQL_PASSWORD']}"
    f"@{os.environ['MYSQL_HOST']}:{os.environ['MYSQL_PORT']}/{os.environ['MYSQL_DATABASE']}"
)
PREPROCESS_COLUMNS = ['idpozo', 'fecha_data', 'prod_pet', 'prod_gas', 'prod_agua', 'tef', 'profundidad', 'tipoextraccion']
GET_LATEST_FS_VERSION = 'SELECT latest_feature_store_preffix FROM fs_metadata'
INSERT_LATEST_FS_VERSION = 'INSERT INTO fs_metadata (creation_date, latest_feature_store_preffix) VALUES (CURRENT_TIMESTAMP(), :version)'
UPDATE_LATEST_FS_VERSION = 'UPDATE fs_metadata SET latest_feature_store_preffix = :version, creation_date = CURRENT_TIMESTAMP()'

@task
def download_dataset():
  logging.info("Downloading dataset...")
  os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
  urllib.request.urlretrieve(DATASET_DOWNLOAD_URL, DATASET_PATH)
  logging.info("Dataset downloaded")

@task
def preprocess():
  df = pd.read_csv(DATASET_PATH)
  columns = PREPROCESS_COLUMNS
  df = df[columns].dropna()
  le = LabelEncoder()
  df['tipoextraccion'] = le.fit_transform(df['tipoextraccion'])
  df['fecha'] = pd.to_datetime(df['fecha_data'], utc=True)
  df = df.sort_values(['idpozo', 'fecha']).reset_index(drop=True)
  records = []
  for well_id, group in df.groupby('idpozo', sort=False):
    group = group.sort_values('fecha').reset_index(drop=True)

    # Calculo los features con ventana deslizante de 10 readings por pozo
    for i in range(len(group)):
      window = group.iloc[max(0, i - 9): i + 1]
      rec = group.iloc[i].to_dict()
      rec['avg_prod_gas_10m'] = float(window['prod_gas'].mean())
      rec['avg_prod_pet_10m'] = float(window['prod_pet'].mean())
      rec['last_prod_gas'] = float(window['prod_gas'].iloc[-1])
      rec['last_prod_pet'] = float(window['prod_pet'].iloc[-1])
      rec['n_readings'] = int(len(window))
      records.append(rec)

    # Creo una fila que represente la próxima ingesta (para online store)
    tail_window = group.iloc[max(0, len(group) - 10): len(group)]
    if not tail_window.empty:
      online_rec = group.iloc[-1].to_dict()
      online_rec['fecha'] = online_rec['fecha'] + pd.DateOffset(months=1)

      online_rec['prod_gas'] = None
      online_rec['prod_pet'] = None
      online_rec['avg_prod_gas_10m'] = float(tail_window['prod_gas'].mean())
      online_rec['avg_prod_pet_10m'] = float(tail_window['prod_pet'].mean())
      online_rec['last_prod_gas'] = float(tail_window['prod_gas'].iloc[-1])
      online_rec['last_prod_pet'] = float(tail_window['prod_pet'].iloc[-1])
      online_rec['n_readings'] = int(len(tail_window))

      records.append(online_rec)

  feat_df = pd.DataFrame(records)
  feat_df.to_csv(PREPROCESSED_PATH, index=False)

@task
def insert_into_online_store():
  logging.info("Inserting into online store...")
  df = pd.read_csv(PREPROCESSED_PATH)
  engine = create_engine(MYSQL_CONN_STRING)

  with engine.connect() as conn:
    result = conn.execute(text(GET_LATEST_FS_VERSION)).fetchone()

    if result is None:
      new_version = "v1"
      conn.execute(
        text(INSERT_LATEST_FS_VERSION),
        {"version": new_version}
      )
    else:
      current_version = result[0]
      version_num = int(current_version.replace("v", "")) + 1
      new_version = f"v{version_num}"
      conn.execute(
        text(UPDATE_LATEST_FS_VERSION),
        {"version": new_version}
      )

    table_name = f"{new_version}_feature_store"
    conn.commit()

  logging.info(f"Creating table {table_name} and inserting {len(df)} rows")
  df.to_sql(table_name, engine, if_exists='replace', index=False)
  logging.info("Done inserting into online store")

@task
def delete_raw_data():
  logging.info("Deleting raw data...")
  for path in [DATASET_PATH, PREPROCESSED_PATH]:
    if os.path.exists(path):
      os.remove(path)
      logging.info(f"Deleted {path}")
  logging.info("Raw data cleanup complete")


@task
def trigger_train_dag():
  client = Client(None, None)
  client.trigger_dag('train_with_online_feature_store')
  logging.info("Triggered train_with_online_feature_store DAG")


@dag(
    dag_id='ingest_dag',
    description='Ingestion into online Store DAG',
    schedule=None,
)
def ingest_dag():
  download_dataset() >> preprocess() >> insert_into_online_store() >> delete_raw_data() >> trigger_train_dag()

ingest_dag()