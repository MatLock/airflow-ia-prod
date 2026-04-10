FROM apache/airflow:3.1.8

RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    mlflow \
    apache-airflow-providers-fab \
    apache-airflow-providers-mysql