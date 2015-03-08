# kaggle-titanic
A deliverable at [PyData.Tokyo](https://pydata.tokyo/) Kaggle Hackathon ([Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic-gettingStarted)). A Logistic Regression model with a few feature engineering methods hit 79.904% accuracy at the leaderboard.

## Setup
Download the data [here](https://www.kaggle.com/c/titanic-gettingStarted/data). Put train.csv and test.csv under the same directory as [logisticmodel.py](https://github.com/g-votte/kaggle-titanic/blob/master/logisticmodel.py).

## Model Overview
The pipeline is composed of one-of-K encoding, scaling, and Logistic Regression with L1 norm.
```python
model = Pipeline(steps=[
    ('onehot', OneHotEncoder(categorical_features=categorical_indices, sparse=False, n_values=17)),
    ('scaler', MinMaxScaler(feature_range=(0, 1))),
    ('classifier', LogisticRegression(penalty='l1')),
])
```

Ticket, Cabin, SibSp, Parch, and Name are dropped by features. N/A values are filled with random integer for Age. In addition, the number of zero Fare values is significantly large only in train.csv; hence, they are also considered as N/A values and filled randomly.

The model accuracy is further improved with the following three tuning methods.

### Feature Engineering 1: Age is N/A
The number of Age N/A is rather small in survivors’ data. Thus, missing value itself can be considered as an important cue, and is added as a binary feature. 

### Feature Engineering 2: Name Title
Name titles are extracted as a categorical feature. They include the information of male/female, married/unmarried, child/adult, which might be related to priority of lifeboat onboarding.

### Feature Engineering 3: Log Fare
Fare feature shows a long tail distribution, where there is a spike around a low value. The feature is converted into log(Fare) so that it better fit to a linear model.
