# Basic Libraries

import numpy as np
import pandas as pd
from warnings import filterwarnings
from collections import Counter

# Visualizations Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.offline as pyo
import plotly.express as px
import plotly.graph_objs as go
pyo.init_notebook_mode()
import plotly.figure_factory as ff
import missingno as msno

# Data Pre-processing Libraries
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,learning_curve

# Modelling Libraries
from sklearn.linear_model import LogisticRegression,RidgeClassifier,SGDClassifier,PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC,LinearSVC,NuSVC
from sklearn.neighbors import KNeighborsClassifier,NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.ensemble import VotingClassifier

# Evaluation & CV Libraries
from sklearn.metrics import precision_score,accuracy_score,roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,RepeatedStratifiedKFold,train_test_split, cross_validate

df = pd.read_csv("/Users/melisacepni/PycharmProjects/miuul_project/term project/water_quality.csv")
#orjinal veri seti + eksik gözlemleri mean le doldurma

df.isnull().sum()

X = df.drop('Potability',axis=1)
y = df['Potability'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

filterwarnings('ignore')
models = [("LR", LogisticRegression(max_iter=1000)), ("SVC", SVC()), ('KNN', KNeighborsClassifier(n_neighbors=10)),
          ("DTC", DecisionTreeClassifier()),
          ("SGDC", SGDClassifier()), ('RF', RandomForestClassifier()), ('ADA', AdaBoostClassifier()),
          ('XGB', GradientBoostingClassifier())]

results = []
names = []
finalResults = []

for name, model in models:
    model.fit(X_train, y_train)
    model_results = model.predict(X_test)
    score = precision_score(y_test, model_results, average='macro')
    results.append(score)
    names.append(name)
    finalResults.append((name, score))

finalResults.sort(key=lambda k: k[1], reverse=True)

finalResults

rf_model = RandomForestClassifier().fit(X_train, y_train)

#confusion matrix için y_pred
y_pred = rf_model.predict(X)

#AUC için y_prob:
y_prob = rf_model.predict_proba(X)[:,1]

#confusion matrix
print(classification_report(y,y_pred))

#AUC
roc_auc_score(y,y_prob)

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))

cv_results = cross_validate(rf_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

print("test accuracy",cv_results['test_accuracy'].mean(),
     "precision",cv_results['test_precision'].mean(),
     "recall",cv_results['test_recall'].mean(),
     "f1",cv_results['test_f1'].mean())

train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(), X, y, cv=5, scoring='recall', train_sizes=np.linspace(0.1, 1.0, 10)
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)


plt.plot(train_sizes, train_scores_mean, label="Learning Score", color="blue")
plt.plot(train_sizes, test_scores_mean, label="Test Score", color="green")
plt.xlabel("Training Set Size")
plt.ylabel("Recall")
plt.title("Learning Curve")
plt.legend()
plt.show()

def plot_importance(model, features, num=len(X)):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature':features.columns})
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1)
    sns.barplot(x="Value",y="Feature",data=feature_imp.sort_values(by="Value",
    ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()

plot_importance(rf_model,X)