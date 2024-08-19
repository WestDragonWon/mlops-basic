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
# 1. 데이터 로드
import pandas as pd
import numpy as np

train = pd.read_csv('titanic_train.csv')
test = pd.read_csv('titanic_test.csv')

train.head()
train.shape, test.shape # test: label 데이터가 없음

# %%
# EDA (Exploratory Data Analysis)
train.info()
train.describe()

# %%
train.isna().sum() # Age, Cabin, Embarked에 결측치 존재
test.isna().sum() # Age, Fare, Cabin에 결측치 존재

# %%
# 시각화
# - 코드로 확인: 생존자 수, 성별, 나이, 객실 등급, 승선 항구, 형제자매, 부모자식 수, 요금 등이 생존에 영향을 미치는가?

survived = train[train['Survived'] == 1]['Pclass'].value_counts()
dead = train[train['Survived'] == 0]['Pclass'].value_counts()

merged_df = pd.DataFrame({
    'Survived': survived,
    'Dead': dead
})

merged_df.plot(kind='bar', stacked=True, figsize=(10, 5)) # pandas의 plot 함수 사용


# %%

def make_bar_chart(column_names):
    survived = train[train['Survived'] == 1][column_names].value_counts()
    dead = train[train['Survived'] == 0][column_names].value_counts()

    merged_df = pd.DataFrame({
        'Survived': survived,
        'Dead': dead
    })

    merged_df.plot(kind='bar', stacked=True, figsize=(10, 5))


# %%
make_bar_chart('Sex')
make_bar_chart('Parch')

# %%
for col in train.columns:
    print(col)
    make_bar_chart(col)
    plt.show()

# %%
# 가장 높은 요금을 낸 상위 10명과 가장 낮은 요금을 낸 하위 10명의 생존율을 확인

train.sort_values(by=['Fare'])[['Name', 'Sex', 'Embarked', 'Fare', 'Survived']].head(10)
train.sort_values(by=['Fare'])[['Name', 'Sex', 'Embarked', 'Fare', 'Survived']].tail(10)

# %%
# Q. 65세 이상인 사람들의 생존율을 확인
# ~ : not

train[train['Age'] >= 65][['Name', 'Sex', 'Embarked', 'Fare', 'Survived']]
train[~(train['Age'] >= 65)][['Name', 'Sex', 'Embarked', 'Fare', 'Survived']]



# %%
train[(train['Age'] <= 10)][['Survived']].mean()

# %%
# Feature Engineering
# (1) Name

train['Name_Title'] = train['Name'].str.extract('([A-Za-z]+)\.') # 정규표현식 사용
train['Name_Title'].value_counts()

#make_bar_chart('Name_Title')

# %%
title_mapping = {
    "Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3, "Countess": 3,
}

train['Name_Title'] = train['Name_Title'].map(title_mapping).fillna(3)
train

# %%
make_bar_chart('Name_Title')

# %%
test['Name_Title'] = test['Name'].str.extract('([A-Za-z]+)\.') # 정규표현식 사용
test['Name_Title'] = test['Name_Title'].map(title_mapping).fillna(3)
test

# %%
# (2) Sex value mapping
train_test_data = [train, test]

# train['Sex'].replace({
#     'male': 0,
#     'female': 1
# })

for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].replace({
    'male': 0,
    'female': 1
})

test['Sex']


# %%
# (3) Age
# 결측치 처리

train['Age'].isna().sum() # 900 / 177

#train.groupby('Name_Title')['Age'].mean()
train.groupby('Name_Title')['Age'].transform('mean')

train['Age'].fillna(train.groupby('Name_Title')['Age'].transform('mean'), inplace=True)




# %%
test['Age'].fillna(test.groupby('Name_Title')['Age'].transform('mean'), inplace=True)
test['Age'].isna().sum()

# %%
# 결측치 확인 0(없음), 1(있음)
train['Age'].isna().sum()

# %%
# 결측치 확인 0(없음), 1(있음)
test['Name_Title'].isna().sum()

# %%
train['Age'].value_counts().to_frame()

# 데이터의 분포를 볼때는 커널 밀도 함수를 사용
import seaborn as sns
sns.kdeplot(train['Age'])

age_bins = [0, 15, 25, 40, np.inf]
age_labels = [0, 1, 2, 3]

#

train['Age'] = pd.cut(train['Age'], bins=age_bins, labels=age_labels)
test['Age'] = pd.cut(test['Age'], bins=age_bins, labels=age_labels)

# %%
# (4) Embarked
train['Embarked'].value_counts()
train['Embarked'].isna().sum()


# %%
train['Embarked'].fillna('S', inplace=True)

# %%
train['Embarked'].isna().sum()

# %%
# 기계는 문자를 인식하지 못하므로 숫자로 변환
train['Embarked'].isna().sum()

# %%
train['Embarked'].value_counts()

embarked_mapping = {
    'S': 0, 'C': 1, 'Q': 2
}

train['Embarked'] = train['Embarked'].replace(embarked_mapping)
test['Embarked'] = test['Embarked'].replace(embarked_mapping)

# %%
train['Embarked'].value_counts()

# %%
# (5) Fare
train['Fare'].value_counts()

sns.kdeplot(train['Fare'])

fare_bins = [0, 20, 100, np.inf]
fare_labels = [0, 1, 2]


pd.cut(train['Fare'], fare_bins, fare_labels)

# %%
train['Fare'] = train['Fare'].fillna(0)

drop_features = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Name_Title']

df_train = train.drop(drop_features, axis=1)

# %%
train.isna().sum()


# %%
df_train

# %%
df_train.isna().sum()

# %%
test['Fare'] = test['Fare'].fillna(0)

drop_features = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Name_Title']

df_test = test.drop(drop_features, axis=1)

# %%
df_test.isna().sum()

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# %%
from sklearn.model_selection import KFold, cross_val_score

#KFold?  ?붙이면 사용법
kfold = KFold(n_splits=10, shuffle=True, random_state=123)

# %%
dt_clf = DecisionTreeClassifier()
scores = cross_val_score(
    dt_clf,
    df_train.drop('Survived', axis=1),
    df_train['Survived'],
    cv=kfold,
    scoring='accuracy'
)

np.mean(scores) * 100

# %%
knn = KNeighborsClassifier()
scores = cross_val_score(
    knn,
    df_train.drop('Survived', axis=1),
    df_train['Survived'],
    cv=kfold,
    scoring='accuracy'
)

np.mean(scores) * 100

# %%
rfc = RandomForestClassifier()
scores = cross_val_score(
    rfc,
    df_train.drop('Survived', axis=1),
    df_train['Survived'],
    cv=kfold,
    scoring='accuracy'
)

np.mean(scores) * 100

# %%
gnb = GaussianNB()
scores = cross_val_score(
    gnb,
    df_train.drop('Survived', axis=1),
    df_train['Survived'],
    cv=kfold,
    scoring='accuracy'
)

np.mean(scores) * 100

# %%
rfc = RandomForestClassifier()

rfc.fit(df_train.drop('Survived', axis=1), df_train['Survived'])

pred = rfc.predict(df_test)

# %%
df_result = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived':pred
})

df_result.to_csv('submission.csv', index=False)


# %% [markdown]
# # MLflow를 활용한 모델 학습 Tracking -> 모델개선 -> 다시 제출
#
# ## 파라미터 값을 버뮈 설정하고 ...
# ## GridSearchCV 등을 써서 우리대신 최적화시키는 방법알기-

# %% [markdown]
#
