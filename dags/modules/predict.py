import dill
import json
import pandas as pd
from datetime import datetime
import logging
import os
from os import listdir, makedirs
from os.path import isfile, join, exists

path = os.environ.get('PROJECT_PATH', '/opt/airflow/dags')


def latest_model(models_path):
    # Получить путь к последнему файлу модели в каталоге models
    if not exists(models_path):
        raise FileNotFoundError(f"Models directory does not exist: {models_path}")
    models = sorted(os.listdir(models_path))
    if not models:
        raise FileNotFoundError(f"No model files found in {models_path}")
    latest_model_filename = models[-1]
    latest_model_path = os.path.join(models_path, latest_model_filename)
    return latest_model_path


def predict(model_path):
    try:
        # Загрузить последнюю модель
        with open(model_path, 'rb') as file:
            model = dill.load(file)

        data = []
        mypath = os.path.join(path, 'modules/data/test/')
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

        for file_name in onlyfiles:
            with open(os.path.join(mypath, file_name)) as file:
                content = json.load(file)
            data.append(content)

        df = pd.DataFrame.from_dict(data, orient='columns')
        result_of_prediction = [model['model'].predict(df.loc[[i]])[0] for i in range(len(df))]

        result_df = pd.DataFrame({
            'pred': result_of_prediction
        })

        preds_dir = os.path.join(path, 'modules/data/predictions')
        if not exists(preds_dir):
            makedirs(preds_dir)
        preds_filename = os.path.join(preds_dir, f'preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv')
        result_df.to_csv(preds_filename, index=False)

        logging.info(f'Predictions are saved as {preds_filename}')

    except Exception as e:
        logging.error(f"Error occurred during prediction: {e}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s - %(message)s')
    try:
        models_path = os.path.join(path, 'modules/data/models')
        model_path = latest_model(models_path)
        predict(model_path)
    except FileNotFoundError as e:
        logging.error(e)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
