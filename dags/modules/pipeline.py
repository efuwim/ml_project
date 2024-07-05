import logging
import os
from datetime import datetime

import dill
import pandas as pd
from imblearn.over_sampling import SMOTE

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder

path = os.environ.get('PROJECT_PATH', '/opt/airflow')

def read_data():
    # Загрузка данных сессий и событий
    df_sessions = pd.read_csv(os.path.join(path, 'dags/modules/data/files/ga_sessions.csv'), low_memory=False)
    df_hits = pd.read_csv(os.path.join(path, 'dags/modules/data/files/ga_hits.csv'))

    # Подготовка датасета с целевой переменной: приведение целевой переменной в бинарный вид и группировка действий
    # в рамках одной сессии
    success = ['sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
               'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click',
               'sub_submit_success', 'sub_car_request_submit_click']

    df_hits['event_action_bin'] = df_hits['event_action'].isin(success).astype(int)

    stats_hits = df_hits.groupby(['session_id'], as_index=False)['event_action_bin'].max()

    # Объединение данных на основе session_id
    df_full = pd.merge(left=df_sessions, right=stats_hits, on='session_id', how='inner')

    # Приведение типу целевой переменной к целому
    df_full['event_action_bin'] = df_full['event_action_bin'].astype(int)

    return df_full

def fill_nan(df_full):
    df_full = df_full.drop(columns=['device_model'])

    # Замена пропусков в марке устройства и отсутствующих данных
    df_full.loc[df_full['device_os'] == 'Macintosh', 'device_brand'] = 'Apple'
    df_full.device_brand = df_full.device_brand.fillna('unknown')
    for column in df_full.columns:
        df_full.loc[:, column] = df_full.loc[:, column].apply(lambda x: 'unknown' if x == '(not set)' else x)

    # Восстановление отсутствующих записей об операционной системе устройства
    def os_recovery(df):
        stats = df.groupby([df.columns[0], df.columns[1]], as_index=False)[df.columns[2]].agg(pd.Series.mode)
        for i in range(len(stats)):
            try:
                index_list = list(df[(df[df.columns[2]].isna()) & (df[df.columns[0]] == stats.loc[i, df.columns[0]]) & \
                                     (df[df.columns[1]] == stats.loc[i, df.columns[1]])].index)
                df.loc[index_list, df.columns[2]] = stats.loc[i, df.columns[2]]
            except:
                pass
        df[df.columns[2]][df[df.columns[2]].isna()] = 'unknown'
        return df[df.columns[2]]

    df_full['device_os'] = os_recovery(df_full[['device_brand', 'device_category', 'device_os']])

    # Удаление строк датафрейма с 3-мя и более отсутствующими UTM-метками, а также отсутствующими значениями utm_source
    utm_nan = list(df_full[(df_full['utm_keyword'].isna()) & (df_full['utm_campaign'].isna()) & (df_full['utm_adcontent'].isna())].index) + \
              list(df_full[df_full['utm_source'].isna()].index)
    df_full = df_full.drop(axis=0, index=utm_nan)

    # Разбиение признаков utm-меток на группы и заполнение пропусков признаков utm-меток модой для соответствующей группы
    def utm_recovery(df):
        stats = df.groupby([df.columns[0], df.columns[1], df.columns[2]], as_index=False)[df.columns[3]].agg(pd.Series.mode)
        for i in range(len(stats)):
            try:
                index_list = list(df[(df[df.columns[3]].isna()) & (df[df.columns[0]] == stats.loc[i, df.columns[0]]) & \
                                     (df[df.columns[1]] == stats.loc[i, df.columns[1]]) & (df[df.columns[2]] == stats.loc[i, df.columns[2]])].index)
                df.loc[index_list, df.columns[3]] = stats.loc[i, df.columns[3]]
            except:
                pass
        df[df.columns[3]][df[df.columns[3]].isna()] = 'unknown'
        return df[df.columns[3]]

    df_full['utm_campaign'] = utm_recovery(df_full[['utm_source', 'utm_medium', 'utm_adcontent', 'utm_campaign']])
    df_full['utm_adcontent'] = utm_recovery(df_full[['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent']])
    df_full['utm_keyword'] = utm_recovery(df_full[['utm_adcontent', 'utm_medium', 'utm_campaign', 'utm_keyword']])
    df_full['device_browser'] = df_full['device_browser'].apply(lambda x: x.split(' ')[0].lower()).replace(
        {'helloworld': 'other', '[fban': 'other', 'mrchrome': 'chrome', 'MyApp': 'other', 'nokiax2-02': 'other',
         'nokia501': 'other', 'com.zhiliaoapp.musically': 'other', 'threads': 'other'})
    df_full = df_full.drop(columns=['session_id', 'client_id'])

    return df_full

def targetencoding(df_full):
    data = ['utm_source', 'utm_campaign', 'utm_medium', 'utm_keyword', 'utm_adcontent', 'device_browser', 'device_os',
            'device_brand', 'device_category', 'geo_country', 'geo_city']
    encoder = TargetEncoder()
    for elem in data:
        df_full[elem + '_encoding'] = encoder.fit_transform(df_full[elem], df_full['event_action_bin'])

    # Удаление первоначальных признаков до преобразований
    columns_for_drop = ['utm_source', 'utm_keyword', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'device_browser',
                        'device_os', 'device_category', 'geo_city', 'device_brand', 'geo_country',
                        'device_screen_resolution', 'visit_date', 'visit_time']
    df_full = df_full.drop(columns=columns_for_drop)
    return df_full

def pipeline():
    df = read_data()
    if df is None:
        return

    preprocessor = Pipeline(steps=[
        ('fill_nan', FunctionTransformer(fill_nan)),
        ('targetencoding', FunctionTransformer(targetencoding)),
    ])

    X = df.drop(['event_action_bin'], axis=1)
    y = df['event_action_bin']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # Обучение модели Gradient Boosting на всём датасете
    final_gb_model = GradientBoostingClassifier(random_state=42)
    final_gb_model.fit(X, y)

    model_filename = f'{path}/dags//modules/data/models/avtopodpiska_pipe_{datetime.now().strftime("%Y%m%d%H")}.pkl'

    with open(model_filename, 'wb') as file:
        dill.dump({
            'predict': final_gb_model,
            'model': final_gb_model,
            'metadata': {
                'name': 'avtopodpiska prediction model',
                'author': 'Yumeame',
                'version': 1,
                'data': datetime.now(),
                'type': 'Gradient Boosting',
            }
        }, file)

    logging.info(f'Model is saved as {model_filename}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s - %(message)s')
    pipeline()
