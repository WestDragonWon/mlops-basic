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

# %%
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# 데이터 준비
iris = load_iris() # 꽃 받침과 꽃 잎 사이즈를 가지고 꽃의 종류를 결정

X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 학습 데이터와 테스트 데이터로 분리 => train_test_split()
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=123)


X_scaled


# 얼굴인식 => ??? => 사람의 얼굴을 수치화 => Open CV => 무인차 (Open CV)


# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train) # train=모의고사 # 학습을 시킬 때는 학습 데이터만 제공

model.predict(X_test) # 예측을 시킬 때는 테스트 데이터만 제공

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"정확도 : {accuracy * 100}")


# %% [markdown]
# 모델 학습과 모델 성능
# 심플하게 모든 것은 ML flow에게 맡긴다. => mlflow.autolog()
# autolog에서 추적하지 못하는 다른 파라미터,메트릭,메타데이터 등등의 값을 수동으로 기록

# %%
mlflow.set_tracking_uri("http://127.0.0.1:5000")
print("",mlflow.get_tracking_uri())

# %%
import mlflow.sklearn

mlflow.autolog()

with mlflow.start_run(nested=True):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"정확도 : {accuracy * 100}")


# %%
exp = mlflow.set_experiment(experiment_name='iris_classification_experiment')

print(f"Name: {exp.name}")
print(f"ID: {exp.experiment_id}")
print(f"Location: {exp.artifact_location}")
print(f"Tags: {exp.tags}")
print(f"Lifecycle: {exp.lifecycle_stage}")
print(f"Create Timestamp: {exp.creation_time}")


# %%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

models = {
    "LogisticRegression" :LogisticRegression(
        max_iter=200, #최대 반복횟수"
        C=1.0, # 규제 강도(C값이 적을 수록 규제가 강화됨)
        solver='lbfgs', #최적화 알고리즘
        random_state=123
    ),
    "RandomForest" : RandomForestClassifier(
        n_estimators=100, #트리의 갯수
        max_depth=None,
        random_state=123
    ),
    "SVC" : SVC(
        kernel='linear', # linear, sigmoid, poly, rbf
        random_state=123
    ),
}   


# %%
# 위 모델들을 한번씩 불러와서 (반복문) => 최고의 모델을 찾아내고, 해당 파라미터를 기록합니다.

mlflow.autolog()

best_accuracy = 0
best_model = None
best_model_name = None

with mlflow.start_run(nested=True):
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name

        print(f"Model Name: {model_name}, Accuracy: {accuracy}")

        mlflow.log_param('best_model', best_model_name) # 파라미터 로그
        mlflow.log_metric('best_accuracy', best_accuracy) # 메트릭 로그

    print(f"Best Model Name: {best_model_name}, Best Accuracy: {best_accuracy}")
    

# %%
mlflow.autolog()
# 전체 모델에 대해서 기록을 하고 싶은데?

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name, nested=True):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        #모델을 mlflow에 저장
        model_path = f"model/{model_name}"
        mlflow.sklearn.log_model(model, model_path)

        mlflow.log_param(f'{model_name}_param', model.get_params())
        mlflow.log_metric(f'{model_name}_accuracy', accuracy)

        print(f"Model Name: {model_name}, Accuracy: {accuracy}")



# %%
# 모델 관리
from mlflow.tracking import MlflowClient
client = MlflowClient()
# 모델을 등록하고, 해당 모델의 버전을 반환
def register_model(model_name, run_id, model_uri='model'): # 모델 등록
    model_uri = f"runs:/{run_id}/{model_uri}"
    model_version = mlflow.register_model(model_uri, model_name)
    return model_version
# 등록된 모델을 stage 단계로 승격
def promote_to_staging(model_name, run_id, model_uri): # stage
    model_version = register_model(model_name, run_id, model_uri)
    client.set_model_version_tag(
        name=model_name,
        version=model_version.version,
        key='stage',
        value='staging'
    )
    print(f"Model: {model_name}, version: {model_version} promoted to Staging...")
def promote_to_production(model_name, version): # production
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key='stage',
        value='production'
    )
    print(f"Model: {model_name}, version: {version} promoted to Production...")
def archive_model(model_name, version): # archive: 모델 폐기 단계
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key='stage',
        value='archived'
    )
    print(f"Model: {model_name}, version: {version} Archived ...")


# %%
# http://127.0.0.1:5000/#/experiments/787161780912017658/runs/cbd04b96570b4d28aa996ea6e5e43e00
# 실험ID: 787161780912017658
# 실행ID: cbd04b96570b4d28aa996ea6e5e43e00
# Model Name: LogisticRegression
# (1) 모델 등록
run_id = 'cbd04b96570b4d28aa996ea6e5e43e00'
model_name = 'LogisticRegression'

model_version = register_model(model_name, run_id)
print(model_version)

# %%
# (2) 모델을 staging 단계로 승격
promote_to_staging(model_name, run_id, 'model')

# %%
# (3) 모델을 Production 단계로 승격
promote_to_production(model_name, '3')

# %%
# (4) 새로운 버전의 모델을 Production으로 승격시키고, 기존의 Production 버전은 Archived
promote_to_production(model_name, '4') # 4 staging -> production
archive_model(model_name, '3') # production -> archive

# %% [markdown]
# ### 모델 Serving
# - FastAPI, Flask ... => API로 언제만들지?
# - mlflow가 해결해줌
# - inference: 값을 전달하고, 그 값에 대한 예측값을 return (API)

# %%
# PM 결과를 보여줘야하는데 PM은 모름 눈으로 보여줘야함
# (1) Model Load
model_name = 'LogisticRegression'
model_version = 4

model_uri = f'models:/{model_name}/{model_version}'

loaded_model = mlflow.pyfunc.load_model(model_uri)

test_input = X_test[:10]
loaded_model.predict(test_input)

# %% [markdown]
# ### Model API Serving
# - 서버가 하나 더 필요합니다.Rest API
# - mlflow 설치 할 때 flask=>API 내려줄 flask 서버를 하나 더 띄워줘야 합니다.
#
# http://127.0.0.1:5000/#/experiments/787161780912017658/runs/d3102882f781401b967ad332d678940c
#
# [text](../mlartifacts/)
#
# 로컬실행
# mlflow models serve -m ./mlartifacts/968704052837447115/2b8120e167474469b9b9cbe753cb643b/artifacts/model -p 5001 --no-conda
#
# => 로컬에서 돌리고 있는데, AWS Sage Maker 올려서 운영을 하시면 됩니다.

# %%
import pandas as pd

X_text_df = pd.DataFrame(X_test, columns=iris.feature_names)

data = {
    'dataframe_split': X_text_df[:10].to_dict(orient='split'),
} # data type: dict -> json

url = "http://127.0.0.1:5001/invocations"

headers = {"Content-Type":"application/json"}

import requests
import json

res = requests.post(url, headers=headers, data=json.dumps(data))


print("Server response(infernece):", res.json())

# %%
