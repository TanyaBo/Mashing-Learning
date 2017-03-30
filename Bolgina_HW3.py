import re
import pandas
import sklearn
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
lemma = nltk.wordnet.WordNetLemmatizer()
from sklearn.model_selection import train_test_split

path = 'C:/Users/user/Documents/4 курс/программирование 4-й курс/Mashing learning/hw3/SouthParkData-master/SouthParkData-master/All-seasons.csv'
charecters = pandas.read_csv(path, sep=',')
main_charecters = ['Stan', 'Kyle', 'Cartman','Butters'] #Выбираю главных героев, у кого больше всего реплик

Stan = charecters.loc[charecters['Character'] == 'Stan']# балансирую выборку так, чтобы реплик каждого героя было одинаковое количество
Kyle = charecters.loc[charecters['Character'] == 'Kyle']
Cartman = charecters.loc[charecters['Character'] == 'Cartman']
Butters = charecters.loc[charecters['Character'] == 'Butters']
cut_Kyle = Kyle.sample(n = 2602)
cut_Stan = Stan.sample(n = 2602)
cut_Cartman = Cartman.sample(n = 2602)
# print(len(Stan))       # У Butters их меньше всего(2602),поэтому уравниваю количество реплик каждого героя до 2602
# print(len(Kyle))
# print(len(Cartman))
# print(len(Butters))
balanced = pandas.concat([cut_Kyle,cut_Stan,cut_Cartman,Butters], ignore_index = True)
balanced = balanced.reindex(np.random.permutation(balanced.index))# перемешиваю всех персонажей
#print(balanced.head()) # Теперь выборка сбалансирована, приступим к нормализации высказываний

def normalize(text):
    without = []
    text = text.lower()
    text = re.sub(r"['_,!\-\"\\\/}{?\.()<>&*+;:|$%]", '', text).strip()# убираю знаки препинания, как показала практика (ДЗ 2) модель лучше классифицирует без знаков препинания
    text = word_tokenize(text)# делю на слова, токенизирую их
    lemmas = [lemma.lemmatize(word) for word in text] # привожу в начальную форму
    # for word in text:
    #     if word not in stop:
    #         without.append(word)# Сначала хотела удалить все стоп слова, но получилось так, что некоторые реплики удалились совсем - состояли только из стоп-слов, поэтому я решила стоп-слова оставить, они отражают особенность речи героя
    # vectors = CountVectorizer()# Векторизация реплик, превращение текста в матрицу частот
    # if len(lemmas) != 0:
    #     vectors.fit_transform(lemmas)
    #     vectors = vectors.transform(lemmas)
    return lemmas

balanced.Line = balanced.Line.apply(normalize)
print(balanced)

#Делю на тренировочную и тестовую выборку 80:20
# X_train, X_test, y_train, y_test = train_test_split(balanced['Line'], balanced['Character'], test_size=0.2)
#
# print(X_train)