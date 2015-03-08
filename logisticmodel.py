import pandas as pd
import numpy as np
import csv as csv

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.utils import shuffle
from random import randint

X_COLUMNS = ['Title', 'AgeIsNa', 'Pclass', 'Age', 'Fare', 'Embarked', 'Sex']
X_CATEGORICAL_COLUMNS = ['Title', 'AgeIsNa', 'Pclass', 'Embarked', 'Sex']


def convert_data_frame(df):
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    if len(df.Embarked[df.Embarked.isnull()]) > 0:
        df.Embarked[df.Embarked.isnull()] = df.Embarked.dropna().mode().values

    ports_dict = {name: i for i, name in list(enumerate(np.unique(df['Embarked'])))}
    df.Embarked = df.Embarked.map( lambda x: ports_dict[x]).astype(int)

    age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
    df['AgeIsNa'] = df['Age'].apply(lambda x: 1 if np.isnan(x) else 0)
    df['Age'] = df['Age'].apply(lambda x: randint(age_min, age_max) if np.isnan(x) else x)

    fare_min, fare_max = int(df['Fare'].min()), int(df['Fare'].max())
    df['Fare'] = df['Fare'].apply(lambda x: randint(fare_min, fare_max) if np.isnan(x) else x)
    df['Fare'] = df['Fare'].apply(lambda x: randint(fare_min, fare_max) if x == 0 else x)
    df['Fare'] = df['Fare'].apply(lambda x: np.log(x))

    df['Title'] = df['Name'].apply(lambda x: x.split('.')[0].split(', ')[1])
    df['Title'] = df[['Title']].apply(lambda x: pd.factorize(x, na_sentinel=-1)[0])

    drop_columns = [c for c in df.columns.values if c not in X_COLUMNS + ['Survived']]
    df = df.drop(drop_columns, axis=1)

    df = df.reindex_axis(X_COLUMNS + ['Survived'], axis=1)

    return df


if __name__ == '__main__':
    df_train = pd.read_csv('train.csv', header=0)
    df_test = pd.read_csv('test.csv', header=0)
    ids = df_test['PassengerId'].values

    df_train = convert_data_frame(df_train)
    df_test = convert_data_frame(df_test)

    X_train = df_train.drop('Survived', axis=1).values
    y_train = df_train['Survived'].values
    X_train, y_train = shuffle(X_train, y_train)

    X_test = df_test.drop('Survived', axis=1).values

    categorical_indices = [i for i in range(len(X_COLUMNS)) if X_COLUMNS[i] in X_CATEGORICAL_COLUMNS]
    model = Pipeline(steps=[
        ('onehot', OneHotEncoder(categorical_features=categorical_indices, sparse=False, n_values=17)),
        ('scaler', MinMaxScaler(feature_range=(0, 1))),
        ('classifier', LogisticRegression(penalty='l1')),
    ])

    tuned_parameters = {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    }

    model = GridSearchCV(model, tuned_parameters, cv=10, verbose=3)
    model = model.fit(X_train, y_train)

    print 'BEST SCORE: %f' % model.best_score_
    print 'BEST PARAMETERS: ' + str(model.best_params_)

    y_test_pred = model.predict(X_test).astype(int)

    predictions_file = open("submit.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId", "Survived"])
    open_file_object.writerows(zip(ids, y_test_pred))
    predictions_file.close()
