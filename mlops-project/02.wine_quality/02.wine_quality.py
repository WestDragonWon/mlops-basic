# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: mlops-basic
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## 와인 데이터
# - 와인의 등급에 영향을 미치는 성분들이 어떤 것들이 있는지 실험/연구 할 수 있는 데이터
# - 
#
# Q.
# - 화이트 와인을 달까?
# - 와인의 등급에 영향을 미치는 성분들이 어떤 것들이 있는지 -> corr()
# - 9등급 와인과 3등급 와인의 각각 성분차이가 가장 큰 값은 무엇인가?
# - 알콜 도수가 높을 수록 와인 등급은 높아지는가?

# %% [markdown]
# 1. 라이브러리 import / 데이터로드
# 2. 피쳐 엔지니어링 (데이터 EDA)
# 3. 모델 설정 및 학습
# 4. MLflow 설정 및 실행 -> metrics, params 기록
# 5. 모델 결과 출력 및 모델 저장

# %%
# 필요한 import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data load
path = '/mnt/e/git/mlops-basic/mlops-project/02.wine_quality/winequality.csv'
df  = pd.read_csv(path, index_col=0)
df.head()


# %%
df['quality'].value_counts().sort_index(ascending=False)

# %%
df.groupby('quality').size()

# %%
corr = df.corr(numeric_only=True)
import seaborn as sns

plt.Figure(figsize=(10,6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')

# %%
# 와인의 등급별 ㅍ ㅕㅇ균 알코올 도수와 평균 당도를 구하시오.
df.groupby('quality')[['residual sugar', 'alcohol']].mean()

# %%
# 조건부 필터링: fixed acidity, ph가 각각 3이상, 4이상인 데이터만 추출 하시오.
df[(df['fixed acidity'] >= 3) & (df['pH'] >= 4)] # boolean indexing

# %%
# 모델링
# (1) 학습데이터(모의고사), 실습데이터(수능) 분리
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.25)

# (2) 분리한 데이터를 csv 형태로 저장
train.to_csv('/mnt/e/git/mlops-basic/mlops-project/02.wine_quality/wine_train.csv')
test.to_csv('wine_test.csv')


# %%
X_train = train.drop(['quality'], axis=1)
X_test = test.drop(['quality'], axis=1)

y_train = train['quality']
y_test = test['quality']

X_train.shape, X_test.shape
y_train.shape, y_test.shape


# %%
# (4) mlflow load
import mlflow
import mlflow.sklearn
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import product

mlflow.autolog()

mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('wine_quality_experiment')

alpha = [0.2, 0.5, 0.7, 1.0]
l1_ratio = [0.2, 0.5, 0.7, 1.0]
# 4x4 = 16

for a, l in product(alpha, l1_ratio):  
    with mlflow.start_run(nested=True) as run:
        lr = ElasticNet(alpha=a, l1_ratio=l, random_state=123)
        lr.fit(X_train, y_train)

        predict = lr.predict(X_test)

        # 모델 성능 평가 => SageMaker
        rmse = np.sqrt(mean_squared_error(y_test, predict)) #MSE
        mae = mean_absolute_error(y_test, predict) #MAE
        r2 = r2_score(y_test, predict) #R2

        # log 기록
        mlflow.log_param('alpha', a)
        mlflow.log_param('l1_ratio', l)

        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mae', mae)
        mlflow.log_metric('r2', r2)

        mlflow.sklearn.log_model(lr, 'mwine_qulity_model')



# %%
from itertools import product

alpha = [0.2, 0.5, 0.7, 1.0]
l1_ratio = [0.2, 0.5, 0.7, 1.0]

for a, l in product(alpha, l1_ratio):
    print(a, l)


# %%
# Model Serving
# http://127.0.0.1:5000/#/experiments/444658877168831610/runs/7775b52691914258becdf7371875cdb3

# mlflow models serve -m /mnt/e/git/mlops-basic/mlops-project/mlartifacts/444658877168831610/7775b52691914258becdf7371875cdb3/artifacts/model -p 5002 --no-conda

# %%
# 데이터 셋 준비 -> 요청
import requests
import json

test_data = pd.read_csv('/mnt/e/git/mlops-basic/mlops-project/02.wine_quality/wine_test.csv')
input_data = test_data.drop(['quality'], axis=1)[:10] # json - streamit(prototyping tool)

data = {
    'dataframe_split': input_data.to_dict(orient='split')
}

url = 'http://127.0.0.1:5002/invocations'

headers = {'Content-Type': 'application/json'}
res = requests.post(url, headers=headers, data=json.dumps(data))

result = res.json()

for i in result['predictions']:
    print(i)

# frontend - REST API 형태로 서버에 요청(json)

# %%
requests.post(url, headers=headers, data=json.dumps(data))
