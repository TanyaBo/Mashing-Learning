import re
import pandas
import sklearn
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
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
    #print(lemmas)                                                   #Сначала хотела удалить все стоп-слова, но получилось так, что некоторые реплики удалились совсем - состояли только из стоп-слов, поэтому я решила стоп-слова оставить, они отражают особенность речи героя
    return lemmas

balanced.Line = balanced.Line.apply(normalize)
#print(balanced)
balanced['Line'] = balanced['Line'].map(lambda x: ' '.join(x))# каждую реплику превращаю из массива нормализованных слов в обычную строку для векторизации
#print(balanced['Line'])
vectors = CountVectorizer()# Векторизация реплик, превращение текста в матрицу частот
vectors.fit_transform(balanced['Line'])
vectors = vectors.transform(balanced['Line'])

#Делю на тренировочную и тестовую выборку 80:20
X_train, X_test, y_train, y_test = train_test_split(vectors, balanced['Character'], test_size=0.2)
#print(X_train)

# Обучаю Логистическую регрессию
logist = LogisticRegression()
logist.fit(X_train, y_train)
y_pred = logist.predict(X_test)
#print(classification_report(y_test, y_pred))
#              precision    recall  f1-score   support                   Результат не очень, средняя точность, полнота и f_мера - 0.47
#
#     Butters       0.57      0.56      0.57       520
#     Cartman       0.49      0.44      0.46       521
#        Kyle       0.41      0.48      0.44       518
#        Stan       0.41      0.39      0.40       523
#
# avg / total       0.47      0.47      0.47      2082

#print(confusion_matrix(y_test, logist.predict(X_test))) #Результат TP, FP  - видно, что много объектов классифицируется неправильно
#print(logist.predict_log_proba(X_test)) # смотрю на предсказания по каждой реплике
#print(logist.coef_)# выдает коэффициенты по каждому классу
#print(logist.intercept_) # выводит константное значение по каждому классу
#print(logist.classes_)# Выдает сами классы - то есть имена главных героев

# от коэффициентов через logodds к вероятности
# logodds = logist.intercept_ + logist.coef_[i] * value[i]
# odds = np.exp(logodds)
# prob = odds/(1 + odds)
# print(prob)
# Попробую поварьировать параметры и поискать лучший

# parameters = {'C': (.1, .05, 0.04, 0.03, 0.2, 0.01)}
# gs = grid_search.GridSearchCV(svm.LinearSVC(), parameters)
# gs.fit(data[:, :4], data[:, 5])
# print('Best result is ',gs.best_score_)
# print('Best C is', gs.best_estimator_.C)
# clf = svm.LinearSVC(C=gs.best_estimator_.C)
# clf.fit(train[:, :4], train[:, 5])

# Теперь обучаю Наивного Байеса
# naive = MultinomialNB()
# naive.fit(X_train, y_train)
# y_pred = naive.predict(X_test)
# print(classification_report(y_test, y_pred))

#              precision    recall  f1-score   support
#
#     Butters       0.54      0.61      0.57       487
#     Cartman       0.47      0.51      0.49       534
#        Kyle       0.45      0.30      0.36       560
#        Stan       0.38      0.45      0.41       501
#
# avg / total       0.46      0.46      0.45      2082

# И Лес тоже

# forest = RandomForestClassifier()
# forest.fit(X_train, y_train)
# y_pred = forest.predict(X_test)
# print(classification_report(y_test, y_pred))

#              precision    recall  f1-score   support
#
#     Butters       0.48      0.45      0.47       530
#     Cartman       0.41      0.36      0.38       538
#        Kyle       0.34      0.35      0.34       491
#        Stan       0.34      0.38      0.36       523
#
# avg / total       0.39      0.39      0.39      2082

# Из трех моделей лучшей оказалась логистическая регрессия, немного хуже справился Байес и совсем плохо лес, теперь сравню Логистическую регрессию с Dummy классификатором

dummy = DummyClassifier()
dummy.fit(X_train, y_train)
y_pred = dummy.predict(X_test)
print(classification_report(y_test, y_pred))

#              precision    recall  f1-score   support
#
#     Butters       0.24      0.26      0.25       517      Конечно, Дамми показал практически равновероятный выбор между классами, и понятно, что это гораздо хуже, чем логистическая регрессия
#     Cartman       0.27      0.25      0.26       535
#        Kyle       0.24      0.24      0.24       499
#        Stan       0.22      0.22      0.22       531
#
# avg / total       0.24      0.24      0.24      2082

#Осталось визуализировать результаты логистической регрессии, поварьировать параметры у классификаторов и интерпретировать коэффициенты логит регрессии