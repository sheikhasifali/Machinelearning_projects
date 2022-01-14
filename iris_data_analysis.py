
# We have given sample Iris dataset of flowers with 3 category to train our Algorithm/classifier and the Purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly.


#imgae show

from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://www.w3resource.com/w3r_images/iris_flower_sepal_and_petal.png", width=1000, height=600)

# import libraries

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Iris.csv')
df.head()

# delete a column
df = df.drop(columns = ['Id'])
df.head()

# to display stats about data
df.describe()

# to basic info about datatype
df.info()

# check for null values
df.isnull().sum()
#correlation
df.corr()

# histograms
df['SepalLengthCm'].hist()

df['SepalWidthCm'].hist()


df['PetalLengthCm'].hist()

df['PetalWidthCm'].hist()

# scatterplot
colors = ['red', 'orange', 'blue']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']

for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()

for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()

for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()


sns.set_theme(style="darkgrid")
sns.pairplot(df, hue="species", palette="icefire")

#label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['Species'] = le.fit_transform(df['Species'])
df.head()

from sklearn.model_selection import train_test_split
# train - 70
# test - 30
X = df.drop(columns=['Species'])
Y = df[['Species']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(x_train,y_train)

# Predict Accuracy Score
y_pred = DT.predict(x_test)
print("Train data accuracy:",accuracy_score(y_true = y_train, y_pred=DT.predict(x_train)))
print("Test data accuracy:",accuracy_score(y_true = y_test, y_pred=y_pred))

print(y_test[0:5],y_pred[0:5])

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

# Print the confusion matrix using Matplotlib

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

#acuracy
accuracy_dt = accuracy_score(y_test,y_pred)
print("Accuracy: {}".format(accuracy_dt))

#classification report
from sklearn.metrics import classification_report
cls_rpt = classification_report(y_test,y_pred)
print("F1 Score: {}".format(cls_rpt))
