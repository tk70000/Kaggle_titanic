import pandas as pd
import csv as csv
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("/Users/ueyamatakuma/desktop/kaggle/titanic/train.csv")
test = pd.read_csv("/Users/ueyamatakuma/desktop/kaggle/titanic/test.csv")

#'Age' の欠損値を埋める
impute_value = train['Age'].median()
train['Age'] = train['Age'].fillna(impute_value)
test['Age'] = test['Age'].fillna(impute_value)

#'Fare' の欠損値を埋める
test['Fare'] = test['Fare'].fillna(train['Fare'].median())

#'Sex'  の数値化
train['IsFemale'] = (train['Sex'] == 'female').astype(int)
test['IsFemale'] = (test['Sex'] == 'female').astype(int)

#学習に使うデータを選択
predictors = ['Age', 'Pclass', 'Fare', 'IsFemale']
X_train = train[predictors].values
y_train = train['Survived']
X_test = test[predictors].values

#学習とフィッティング
model = LogisticRegression()
model.fit(X_train, y_train)
#予測
y_predict = model.predict(X_test)

ids = test["PassengerId"].values

#提出用ファイルの作成
submit_file = open("first_model_submit.csv", "w")
file_object = csv.writer(submit_file)
file_object.writerow(["PassengerId", "Survived"])
file_object.writerows(zip(ids, y_predict))
submit_file.close()
