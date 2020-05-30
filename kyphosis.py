# -*- coding: utf-8 -*-


#Step 0 : Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Step 1: Import dataset
kyphosis_df = pd.read_csv('kyphosis.csv')
kyphosis_df.head(10)
kyphosis_df.tail(10)
kyphosis_df.describe()
kyphosis_df.info()

#Step 2: Visualize the dataset
sns.countplot(kyphosis_df['Kyphosis'], label = 'count')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
kyphosis_df['Kyphosis'] = le.fit_transform(kyphosis_df['Kyphosis'])

kyphosis_False = kyphosis_df[kyphosis_df['Kyphosis'] == 0]
kyphosis_True = kyphosis_df[kyphosis_df['Kyphosis'] == 1]

print("Disease present after operation in percentage = ", (len(kyphosis_True)/len(kyphosis_df)) * 100, '%')
print("Disease absent after operation in percentage = ", (len(kyphosis_False)/len(kyphosis_df)) * 100, '%')

sns.heatmap(kyphosis_df.corr(), annot = True)
sns.pairplot(kyphosis_df, hue = 'Kyphosis', vars = ['Age','Number','Start'])

#Step 3: Create training and testing dataset / Data cleaning
X = kyphosis_df.drop(['Kyphosis'], axis = 1)
y = kyphosis_df['Kyphosis']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#Step 4: Model Training
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

feature_importance = pd.DataFrame(classifier.feature_importances_, index = X_train.columns, columns = ['importance'])
feature_importance
feature_importance = pd.DataFrame(classifier.feature_importances_, index = X_train.columns,
                                  columns = ['importance']).sort_values('importance',ascending = False)
feature_importance

#Step 5: Evaluation
from sklearn.metrics import classification_report, confusion_matrix

y_predict_train = classifier.predict(X_train)
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot = True)

y_predict = classifier.predict(X_test)
cm1 = confusion_matrix(y_test, y_predict)
sns.heatmap(cm1, annot = True)

print(classification_report(y_test, y_predict))

#Step 6: Improving the model
from sklearn.ensemble import RandomForestClassifier
cl = RandomForestClassifier(n_estimators = 300)
cl.fit(X_train, y_train)

y_predict_train1 = classifier.predict(X_train)
cm2 = confusion_matrix(y_train, y_predict_train1)
sns.heatmap(cm2, annot = True)

y_predict1 = classifier.predict(X_test)
cm3 = confusion_matrix(y_test, y_predict1)
sns.heatmap(cm3, annot = True)

print(classification_report(y_test, y_predict1))
