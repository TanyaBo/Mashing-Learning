import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

df  = pd.read_csv('titanic.csv', index_col = "PassengerId")
#sns.barplot(x="Sex", y="Survived", data = df);
#plt.show()#Вероятность выжить у мужчин составила 19% и оказалась ниже, чем у женщин - 74%
#sns.barplot(x="Pclass", y="Survived", data = df);
#plt.show()#Самая высокая вероятность выжить у пассажиров класса 1(63%), затем класса 2(47%) и класса 3 (24%)
#sns.boxplot(x="Pclass", y="Fare", data = df);
#plt.show()#стоимость билетов в среднем больше у пассажиров 1-го класса(61), у 2 класса ниже (в среднем 20), 3 класс - 14. Только среди пассажиров первого класса есть большие выбросы - 514, 262
#2)
#sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=df, kind="bar");
#plt.show()# P(FS|C1)= 96%, P(FS|C3)>P(MS|C1)(50%>39%), F - женщина, S - выживший, С - класс
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Sex'] = df['Sex'].map({'female': 0, 'male':1}).astype(int)
x_labels = ['Pclass', 'Fare', 'Age', 'Sex', 'SibSp']
X, y = df[x_labels], df['Survived']
