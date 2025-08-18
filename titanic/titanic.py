import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
labenc = preprocessing.LabelEncoder()

df = pd.read_csv("train.csv")

# # Check for null values
# print(df.isnull().sum())

# Drop columns with null values that are more than 35%
coldrop = df.isnull().sum()[df.isnull().sum()>(0.35*df.shape[0])]
icd = coldrop.index
df.drop(icd,axis=1,inplace=True)
# print(df.isnull().sum())

# Since Age is a continuous variable, we can fill null values with their
# respective mean
df['Age'].fillna(df['Age'].mean(),inplace=True)
# print(df.isnull().sum())

# df.describe(include='object')
df['Embarked'].fillna('S',inplace=True)
# print(df.isnull().sum())

sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='RdYlGn')
# plt.show()

# Calculate family size and add Alone column
df['FamilySize'] = df['SibSp']+df['Parch']
df.drop(['SibSp','Parch'],axis=1,inplace=True)
df['Alone'] = [0 if df['FamilySize'][i]>1 else 1 for i in df.index]

# Encode Sex variable
df['Sex'] = [0 if df['Sex'][i]=='male' else 1 for i in df.index]

# Encode Embarked variable
df['Embarked'] = labenc.fit_transform(df['Embarked'])

# Survival based on gender
# print(df.groupby(['Sex'])['Survived'].mean())

# Survival based on family size
# print(df.groupby(['Alone'])['Survived'].mean())

# Survival based on Pclass
# print(df.groupby(['Pclass'])['Survived'].mean())

# Survival based on Embarked
# print(df.groupby(['Embarked'])['Survived'].mean())

x = df.drop(['Name', 'PassengerId', 'Ticket', 'Survived'], axis=1)
y = df['Survived']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.50)

# # Logistic regression
# logreg = LogisticRegression()
# logreg.fit(x_train,y_train)
# test_pred = logreg.predict(x_test)
# print(classification_report(y_test,test_pred))

# # Naive Bayes Classifier
# nbc = GaussianNB()
# nbc.fit(x_train,y_train)
# y_test_pred = nbc.predict(x_test)
# print(classification_report(y_test,y_test_pred))

# # KNN
# knn = KNeighborsClassifier(n_neighbors=2)
# knn.fit(x_train,y_train)
# test_pred = knn.predict(x_test)
# print(classification_report(y_test,test_pred))

# # Decision Tree
# dtree = DecisionTreeClassifier()
# dtree.fit(x_train,y_train)
# test_pred = dtree.predict(x_test)
# print(classification_report(y_test,test_pred))

# # Support Vector Machine
# sv = svm.SVC(kernel='linear')
# sv.fit(x_train,y_train)
# test_pred = sv.predict(x_test)
# print(classification_report(y_test,test_pred))

# Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis()
lda.fit(x_train,y_train)
print(classification_report(y_test,lda.predict(x_test)))
