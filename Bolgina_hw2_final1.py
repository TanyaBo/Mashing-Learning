import matplotlib.pyplot as plt
import re
import collections
import pandas
import sklearn
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
ps = nltk.stem.PorterStemmer()
stem = nltk.stem.SnowballStemmer('english')
lemma = nltk.wordnet.WordNetLemmatizer()
stop = set(stopwords.words('english'))
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


#Задание 1.

path = 'C:/Users/user/Documents/4 курс/программирование 4-й курс/Mashing learning/hw2/SMSSpamCollection'
messages = pandas.read_csv(path, sep='\t', names=["label", "message"])
#print(messages.groupby('label').describe()) #смотрим на наши данные, сгруппированные по ham/spam
#Видно, что датасет несбалансирован, так как наблюдений класса spam намного меньше (747), чем класса ham(4825). Dummy classifier,
#который всем новым наблюдениям будет присваивать класс ham, в большинстве случаев будет делать это правильно в связи с особенностями нашего датасета:
# - ham встречается в 86% случаев. Но для определения спама это плохо, так как в реальных условиях спам определяться не будет.

#Теперь балансирую выборку и рандомно выбираю 747 сообщений с пометой "ham" (так как "spam"a в нашей выборке ,было 747) и записываю все в новый датафрейм

ham = messages.loc[messages['label'] == 'ham']
spam = messages.loc[messages['label'] == 'spam']
cut_hum = ham.sample(n = 747)
balanced = pandas.concat([cut_hum, spam], ignore_index = True)
#print(balanced)

# Нормализация текста:

def tokenize(text): #токенизация слов со знаками препинания
    text = text.lower()
    return word_tokenize(text)

def tokenize_without_punkt(text): #токенизация слов без знаков препинания
    text = text.lower()
    text = re.sub(r"['_,!\-\"\\\/}{?\.()<>&*+;:|$%]",'',text).strip()
    return word_tokenize(text)

def lemmatize(text): # лемматизация
    text = text.lower()
    text = re.sub(r"['_,!\-\"\\\/}{?\.()<>&*+;:|$%]",'',text).strip()
    text = word_tokenize(text)
    lemmas = [lemma.lemmatize(word) for word in text]
    return lemmas

def stemming(text): # стемминг
    text = text.lower()
    text = re.sub(r"['_,!\-\"\\\/}{?\.()<>&*+;:|$%]",'',text).strip()
    text = word_tokenize(text)
    stems = [ps.stem(word) for word in text]
    return stems

def without_stopwords(text): #удаляю стоп-слова
    without = []
    text = text.lower()
    text = re.sub(r"['_,!\-\"\\\/}{?\.()<>&*+;:|$%]", '', text).strip()
    text = word_tokenize(text)
    for word in text:
        if word not in stop:
            without.append(word)
    return without

def threshhold_df(text): #считаю частоту всех слов в колонке messages;
    lemmas = []
    for text in messages['message']:
        text = text.lower()
        text = re.sub(r"['_,!\-\"\\\/}{?\.()“<>&*+;:|$%]", '', text).strip()
        text = word_tokenize(text)
        for word in text:
            lemma = lemmatize(word)
            lemmas.append(lemma[0])
    #print(len(collections.Counter(lemmas)))
    freq = dict(collections.Counter(lemmas))
    #print(sorted(freq.values())))# вывожу отсортированные частоты всех слов, получаю, что минимальная частота = 1, максимальная = 2251, теперь надо удалить из словаря слова, частота корторых 1 или 2251.
    new_freq = []
    for i in freq:
        if freq[i]!=1 and freq[i]!=2251:#удаляю пороги максимальной/минимальной document frequency(слова с частотой 1 и 2251)
            new_freq.append(i)
    #print(len(new_freq))
    return new_freq

#threshhold_df()

# Преобразуем слова сообщений в матрицу с частотой слов по всем сообщениям, делаем это двумя способами:
# bow = CountVectorizer()
bow1 = TfidfVectorizer()
# bow.fit_transform(balanced['message'])
# bow1.fit_transform(balanced['message'])
# bowed_messages = bow.transform(balanced['message'])
# bowed1_messages = bow1.transform(balanced['message'])
#
# # Запускаем классификатор и отдаем ему на вход полученные двумя способами матрицы
# naive_model = MultinomialNB()
# naive_model.fit(bowed_messages, balanced['label'])
# naive_model.fit(bowed1_messages, balanced['label'])
#
# cv_results1 = cross_val_score(naive_model, bowed_messages, balanced['label'], cv=10, scoring='accuracy')
# #print(round(cv_results1.mean(),3), round(cv_results1.std(),3))# Для countvectorizer значения 0.965 и 0.014(при первом запуске! при повторном прогоне значения становятся меньше)
#
# cv_results2 = cross_val_score(naive_model, bowed1_messages, balanced['label'], cv=10, scoring='accuracy')
#print(round(cv_results2.mean(),3), round(cv_results2.std(),3))# Для TfidfVectorizer значения 0.969 0.012 В целом, получилось, что результаты для tf-idf чуть-чуть лучше по сравнению

# Теперь пройдемся по всем функциям и посмотрим, в каких случаях классификатор показывает наилучшие показатели

# bow_t = CountVectorizer(analyzer=tokenize)# в качестве анализатора беру свою функцию def tokenize(text): #токенизация слов со знаками препинания
# bow_t.fit_transform(balanced['message'])
# bowed_messages = bow_t.transform(balanced['message'])
# naive_model = MultinomialNB()
# naive_model.fit(bowed_messages, balanced['label'])
# cv_results = cross_val_score(naive_model, bowed_messages, balanced['label'], cv=10, scoring='accuracy')
# print(round(cv_results.mean(),3), round(cv_results.std(),3)) # Результат - 0.953    0.013

# bow_tw = CountVectorizer(analyzer=tokenize_without_punkt)# в качестве анализатора беру функцию tokenize_without_punkt(text): #токенизация слов без знаков препинания
# bow_tw.fit_transform(balanced['message'])
# bowed_messages = bow_tw.transform(balanced['message'])
# naive_model = MultinomialNB()
# naive_model.fit(bowed_messages, balanced['label'])
# cv_results = cross_val_score(naive_model, bowed_messages, balanced['label'], cv=10, scoring='accuracy')
# print(round(cv_results.mean(),3), round(cv_results.std(),3)) # Результат - 0.956 0.007, получается, что при токенизации без знаков препинания результат немного лучше  - точность выше на 0,003

# bow_l = CountVectorizer(analyzer=lemmatize)# в качестве анализатора беру функцию def lemmatize(text): # лемматизация
# bow_l.fit_transform(balanced['message'])
# bowed_messages = bow_l.transform(balanced['message'])
# naive_model = MultinomialNB()
# naive_model.fit(bowed_messages, balanced['label'])
# cv_results = cross_val_score(naive_model, bowed_messages, balanced['label'], cv=10, scoring='accuracy')
# print(round(cv_results.mean(),3), round(cv_results.std(),3)) # Результат - 0.954 0.012, то есть хуже, чем при токенизации без знаков препинания, но выше, чем при токенизации со знаками препинания

# bow_s = CountVectorizer(analyzer=stemming)# в качестве анализатора беру функцию def stemming(text): стемминг
# bow_s.fit_transform(balanced['message'])
# bowed_messages = bow_s.transform(balanced['message'])
# naive_model = MultinomialNB()
# naive_model.fit(bowed_messages, balanced['label'])
# cv_results = cross_val_score(naive_model, bowed_messages, balanced['label'], cv=10, scoring='accuracy')
# print(round(cv_results.mean(),3), round(cv_results.std(),3)) # 0.954 0.013 почти такая же точность как и при лемматизации

# bow_ws = CountVectorizer(analyzer=without_stopwords)# в качестве анализатора беру функцию def without_stopwords(text): #удаляю стоп-слова
# bow_ws.fit_transform(balanced['message'])
# bowed_messages = bow_ws.transform(balanced['message'])
# naive_model = MultinomialNB()
# naive_model.fit(bowed_messages, balanced['label'])
# cv_results = cross_val_score(naive_model, bowed_messages, balanced['label'], cv=10, scoring='accuracy')
# print(round(cv_results.mean(),3), round(cv_results.std(),3)) # 0.941 0.013, на данный момент это самая низкая частотность из всех прогонов

# bow_th = CountVectorizer(analyzer=threshhold_df)# в качестве анализатора беру функцию def threshhold_df(text): #считаю частоту всех слов в колонке messages;
# bow_th.fit_transform(balanced['message'])
# bowed_messages = bow_th.transform(balanced['message'])
# naive_model = MultinomialNB()
# naive_model.fit(bowed_messages, balanced['label'])
# cv_results = cross_val_score(naive_model, bowed_messages, balanced['label'], cv=10, scoring='accuracy')
# print(round(cv_results.mean(),3), round(cv_results.std(),3)) # Почему-то очень долго думал,не выдавая ошибки, я не дождалась результатов)

#Итого, из всех прогонов, можно сделать вывод, что лучше всего наивный байес справляется при анализаторе TfidfVectorizer, в этом случае средняя точность = 0.969
# при стандартном отклонении = 0.012. Если выбирать из разных типов токенизации, лучше брать токенизацию без знаков препинания; сравнивая лемматизацию и стемминг

#Задание 2(проверяем самый лучший анализатор на других классификаторах)

# Делю другим способом на тестовую и тренировочную выборки и обучаю классификатор Дерево решений на наших данных с использованием лучшего анализатора TfidfVectorizer:

# msg_train, msg_test, label_train, label_test = train_test_split(balanced['message'], balanced['label'], test_size=0.2)
# msg_train = bow1.fit_transform(msg_train)
# msg_test = bow1.transform(msg_test)
# clf = DecisionTreeClassifier()
# clf.fit(msg_train, label_train)
# pred = clf.predict(msg_test)
# print(classification_report(label_test, pred))
# print(confusion_matrix(label_test, pred))
#                                                               Итак, получается, что точность Дерева решений высокая, но ниже, чем у Наивного Байеса (0,95 по сравнению с 0.969 ~ 0,97)
#                                                               TN = 149, FN = 14, FP = 3, TP = 133, то есть в 14 случаях классификатор определил спам как хам,
                                                                # и в трех случаях он, наоборот, определил хам как спам
#              precision    recall  f1-score   support
#
#         ham       0.91      0.98      0.95       152
#        spam       0.98      0.90      0.94       147
#
# avg / total       0.95      0.94      0.94       299
#
# [[149   3]
#  [ 14 133]]

#msg_train, msg_test, label_train, label_test = train_test_split(balanced['message'], balanced['label'], test_size=0.2)
# msg_train = bow1.fit_transform(msg_train)
# msg_test = bow1.transform(msg_test)
#random = RandomForestClassifier()
# random.fit(msg_train, label_train)
# pred = random.predict(msg_test)
# print(classification_report(label_test, pred))
# print(confusion_matrix(label_test, pred))
                                                            #Random Forest классификатор справился хуже, чем все остальные классификаторы, его точность 0,92 по сравнению с 0,95(Decision tree) и 0,97(naive Bayes)
                                                            #по сравнению с Decision tree False Negative значений стало больше вместо 14 - 22. То есть спам определился как неспам(хам) На моих данных получается,
                                                            # что лучше всего использовать классификатор Naive Bayes
#              precision    recall  f1-score   support
#
#         ham       0.87      0.98      0.92       154
#        spam       0.98      0.85      0.91       145
#
# avg / total       0.92      0.92      0.92       299
#
# [[151   3]
#  [ 22 123]]

# Пытаюсь построить learning_curve и применить ее для классификатора Random Forest

# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
#                         n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
#     plt.figure()
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, title, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.grid()
#
#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                     train_scores_mean + train_scores_std, alpha=0.1,
#                     color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#             label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#             label="Cross-validation score")
#
#     plt.legend(loc="best")
#     return plt
#
# X, y = msg_train.shape(), label_train.shape()
# title = "Learning Curves (Random Forest)"
# estimator = RandomForestClassifier()
# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
# plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=1)
# plt.show()

# пытаюсь построить ROC-кривую

# X = msg_train
# y = label_train
#
# y = label_binarize(y, classes=[0, 1])
# n_classes = y.shape[1]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
# random_state=0)
#
# classifier = RandomForestClassifier()
# y_score = classifier.fit(X_train, y_train).decision_function(X_test)
#
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Задание 3 (Это какой-то кошмар...)

# Делю на тренировочную и тестовую выборку в соотношении 80*20
msg_train, msg_test, label_train, label_test = train_test_split(balanced['message'], balanced['label'], test_size=0.2)
#print(msg_train)
#messages['length'] = messages['message'].map(lambda text: len(text))
# Создаю новый датафрейм и записываю туда только данные тренировочной выборки, все дальнейшие манипуляции буду проводить с ней
df = {'message': pandas.Series([]), 'label' : pandas.Series([])}
only_train = pandas.DataFrame(df)
only_train['message'] = msg_train
only_train['label'] = label_train
ham = only_train.loc[only_train['label'] == 'ham']
spam = only_train.loc[only_train['label'] == 'spam']
i = len(spam)
ham = ham.sample(n = i )
print(len(spam), len(ham))