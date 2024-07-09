import logging
import os
from datetime import datetime

import dill
import pandas as pd
from imblearn.over_sampling import SMOTE

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

path = os.environ.get('PROJECT_PATH', '/opt/airflow')

def read_data():
    df_sessions = pd.read_csv(os.path.join(path, 'dags/modules/data/files/ga_sessions.csv'), low_memory=False)
    df_hits = pd.read_csv(os.path.join(path, 'dags/modules/data/files/ga_hits.csv'))

    success = ['sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
               'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click',
               'sub_submit_success', 'sub_car_request_submit_click']

    df_hits['event_action_bin'] = df_hits['event_action'].isin(success).astype(int)

    stats_hits = df_hits.groupby(['session_id'], as_index=False)['event_action_bin'].max()

    df_full = pd.merge(left=df_sessions, right=stats_hits, on='session_id', how='inner')

    df_full['event_action_bin'] = df_full['event_action_bin'].astype(int)

    # Уменьшаем размер датасета в два раза случайным образом
    df_full = df_full.sample(frac=0.3, random_state=42)

    return df_full

def fill_nan(df_full):
    df_full = df_full.drop(columns=['device_model'])

    df_full.loc[df_full['device_os'] == 'Macintosh', 'device_brand'] = 'Apple'
    df_full.device_brand = df_full.device_brand.fillna('unknown')
    df_full = df_full.applymap(lambda x: 'unknown' if x == '(not set)' else x)

    df_full['device_os'] = df_full.groupby(['device_brand', 'device_category'])['device_os'].transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 'unknown'))

    utm_nan = df_full[(df_full['utm_keyword'].isna()) & (df_full['utm_campaign'].isna()) & (df_full['utm_adcontent'].isna())].index.tolist() + \
              df_full[df_full['utm_source'].isna()].index.tolist()
    df_full = df_full.drop(axis=0, index=utm_nan)

    df_full['utm_campaign'] = utm_recovery(df_full, ['utm_source', 'utm_medium', 'utm_adcontent'], 'utm_campaign')
    df_full['utm_adcontent'] = utm_recovery(df_full, ['utm_source', 'utm_medium', 'utm_campaign'], 'utm_adcontent')
    df_full['utm_keyword'] = utm_recovery(df_full, ['utm_adcontent', 'utm_medium', 'utm_campaign'], 'utm_keyword')

    df_full['device_browser'] = df_full['device_browser'].str.split(' ').str[0].str.lower().replace({
        'helloworld': 'other', '[fban': 'other', 'mrchrome': 'chrome', 'MyApp': 'other', 'nokiax2-02': 'other',
        'nokia501': 'other', 'com.zhiliaoapp.musically': 'other', 'threads': 'other'
    })
    df_full = df_full.drop(columns=['session_id', 'client_id'])

    return df_full

def utm_recovery(df, group_cols, target_col):
    stats = df.groupby(group_cols)[target_col].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown')
    df[target_col] = df.apply(
        lambda row: stats[row[group_cols[0]], row[group_cols[1]], row[group_cols[2]]] if pd.isna(row[target_col]) else row[target_col],
        axis=1
    )
    df[target_col] = df[target_col].fillna('unknown')
    return df[target_col]

def target_encoding(df_full):
    categorical_cols = ['utm_source', 'utm_campaign', 'utm_medium', 'utm_keyword', 'utm_adcontent', 'device_browser',
                        'device_os', 'device_category', 'geo_country', 'geo_city']

    encoder = OneHotEncoder(drop='first')
    encoded_data = encoder.fit_transform(df_full[categorical_cols])
    df_encoded = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(categorical_cols))

    df_full = pd.concat([df_full.drop(columns=categorical_cols), df_encoded], axis=1)
    return df_full

def pipeline():
    df = read_data()
    if df is None:
        return

    preprocessor = Pipeline(steps=[
        ('fill_nan', FunctionTransformer(fill_nan)),
        ('target_encoding', FunctionTransformer(target_encoding)),
    ])

    X = df.drop(['event_action_bin'], axis=1)
    y = df['event_action_bin']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    final_gb_model = GradientBoostingClassifier(random_state=42)
    final_gb_model.fit(X, y)

    model_filename = f'{path}/dags/modules/data/models/avtopodpiska_pipe_{datetime.now().strftime("%Y%m%d%H")}.pkl'

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
