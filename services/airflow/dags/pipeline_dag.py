"""Single automated pipeline (orchestrator): data engineering -> model engineering -> deployment.

Runs every 5 minutes. load_data.py is not part of this schedule -- the raw dataset is static, and
no need to hit the Kaggle API every 5 minutes. Run it once manually before starting Airflow.

AIRFLOW_HOME is services/airflow, so this file must stay at
services/airflow/dags/pipeline_dag.py for the repo-root path math below to
hold.
"""
from pathlib import Path

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

REPO_ROOT = Path(__file__).resolve().parents[3]
VENV_PYTHON = REPO_ROOT / "services" / "airflow_venv" / "bin" / "python3"
COMPOSE_FILE = REPO_ROOT / "code" / "deployment" / "docker-compose.yml"

with DAG(
    dag_id="walmart_sales_pipeline",
    description="Data engineering -> model engineering -> deployment",
    schedule="*/5 * * * *",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
) as dag:
    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=f"{VENV_PYTHON} {REPO_ROOT}/code/datasets/prepare_data.py",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"{VENV_PYTHON} {REPO_ROOT}/code/models/train.py",
    )

    deploy = BashOperator(
        task_id="deploy",
        bash_command=(
            f"docker compose -f {COMPOSE_FILE} up -d && "
            f"docker compose -f {COMPOSE_FILE} restart api"
        ),
    )

    prepare_data >> train_model >> deploy
