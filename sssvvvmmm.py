import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, classification_report

s=load_breast_cancer()
df=pd.DataFrame(s["data"],columns=s['feature_names'])
# print(df.head())
# print(s["feature_names"])
# sns.heatmap(df.isnull())
# plt.show()
# ['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename']


# from sklearn.datasets import
x=df

y=s['target']
# print(x)
# print(y)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
x_trained, x_test , y_trained, y_test = train_test_split(x,y,test_size=.4,random_state=101)
# print(x_test,y_test)
a=SVR()
a.fit(x_trained,y_trained)
pr=a.predict(x_test)
print(pr)
# print(confusion_matrix(y_test,pr))
# print(classification_report(y_test,pr))

