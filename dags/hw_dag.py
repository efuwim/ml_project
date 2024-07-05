import datetime as dt
import os

from airflow import DAG
from airflow.operators.python import PythonOperator
from modules.pipeline import pipeline
from modules.predict import predict, latest_model

# Указываем путь к корневой папке проекта
project_path = os.path.abspath(os.getenv('PROJECT_PATH', '/opt/airflow/dags'))

# Устанавливаем переменную окружения PROJECT_PATH для доступа к проекту из DAG
os.environ['PROJECT_PATH'] = project_path

# Указываем путь к папке с моделями
models_path = os.path.join(project_path, 'modules/data/models')

# Проверяем существование пути к моделям
if not os.path.exists(models_path):
    raise FileNotFoundError(f"Models directory does not exist: {models_path}")

# Настройки DAG
default_args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2024, 6, 26),  # Указываем точную дату и время начала DAG
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1),
    'depends_on_past': False,
}

# Определение DAG
with DAG(
    dag_id='sberautopodpiska_prediction',
    schedule_interval="00 15 * * *",  # Запускать каждый день в 15:00 UTC
    default_args=default_args,
) as dag:
    # Определение задачи для выполнения pipeline
    run_pipeline = PythonOperator(
        task_id='pipeline',
        python_callable=pipeline,  # Функция pipeline из модуля modules.pipeline
    )

    # Определение задачи для выполнения predict
    run_predict = PythonOperator(
        task_id='predict',
        python_callable=predict,  # Функция predict из модуля modules.predict
        op_args=[latest_model(models_path)],  # Передаем путь к последней модели в качестве аргумента
    )

    # Установка зависимости между задачами pipeline и predict
    run_pipeline >> run_predict
