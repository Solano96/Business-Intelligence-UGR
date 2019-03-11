# -*- coding: utf-8 -*-
"""
Autor:
    Francisco Solano López Rodríguez
Fecha:
    Noviembre/2018
Contenido:
    Práctica 3
    Inteligencia de Negocio
    Grado en Ingeniería Informática
    Universidad de Granada
"""

''' -------------------- IMPORT LIBRARY -------------------- '''

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import datetime

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn import ensemble

''' --- classifiers import --- '''
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import xgboost as xgb
import lightgbm as lgb
from sklearn import tree

from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from catboost import Pool, CatBoostClassifier

''' --- preprocessing import --- '''
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.preprocessing import Normalizer

''' --- metrics import --- '''
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from math import sin, cos, sqrt, atan2, radians


# Obtener datos respecto a la fecha y obtener la edad del pozo
def date_parser(df):
    date_recorder = list(map(lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d'),
                             df['date_recorded'].values))
    df['year_recorder'] = list(map(lambda x: int(x.strftime('%Y')), date_recorder))
    df['weekday_recorder'] = list(map(lambda x: int(x.strftime('%w')), date_recorder))
    df['yearly_week_recorder'] = list(map(lambda x: int(x.strftime('%W')), date_recorder))
    df['month_recorder'] = list(map(lambda x: int(x.strftime('%m')), date_recorder))
    df['age'] = df['year_recorder'].values - df['construction_year'].values
    del df['date_recorded']
    return df


# Obtener a distancia a la coordenada (0,0)
def distancia(lon1, lat1, lon2, lat2):  
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    R = 6371

    return R * c

# Obtener la coordenada cartesiana x a partir de las longitud y la latitud
def cartesian_x(lon, lat):
    lat=radians(lat)
    lon=radians(lon)
    R=6371.0
    x = R * cos(lat) * cos(lon)
    return x

# Obtener la coordenada cartesiana y a partir de las longitud y la latitud
def cartesian_y(lon, lat):
    lat=radians(lat)
    lon=radians(lon)
    R=6371.0
    y = R * cos(lat) * sin(lon)
    return y

# Matriz de confusion
def plot_confusion_matrix(y_test, predictions):
    cm = metrics.confusion_matrix(y_test, predictions)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

# Funcion para realizar la validacion cruzada
def cross_validation(clf, X, y, cv = None, min_max_scaler = False, scaled = False, standard_scaler = False, normalizer = False, poly = False, m_confusion = False):

    if cv == None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123456)

    iteration = 0

    for train, test in cv.split(X, y):

        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]


        if min_max_scaler:
            X_train = MinMaxScaler().fit_transform(X_train)
            X_test = MinMaxScaler().fit_transform(X_test)

        if scaled:
            X_train = scale(X_train)
            X_test = scale(X_test)

        if poly:
            X_train = PolynomialFeatures(degree = 2, interaction_only=True).fit_transform(X_train)
            X_test = PolynomialFeatures(degree = 2, interaction_only=True).fit_transform(X_test)

        if standard_scaler:
            transformer = StandardScaler().fit(X_train)
            X_train = transformer.transform(X_train)
            X_test = transformer.transform(X_test)

        if normalizer:
            transformer = Normalizer().fit(X_train)
            X_train = transformer.transform(X_train)
            X_test = transformer.transform(X_test)

        t = time.time()
        clf = clf.fit(X_train,y_train)
        training_time = time.time() - t

        predictions_train = clf.predict(X_train)
        predictions = clf.predict(X_test)

        print("--------- Iteración ", iteration, " --------- ")
        print("Tiempo :: ", training_time)
        print ("Train Accuracy :: ", accuracy_score(y_train, predictions_train))
        print ("Test Accuracy  :: ", accuracy_score(y_test, predictions))
        print("")

        if m_confusion:
            plot_confusion_matrix(y_test, predictions)

        iteration += 1

''' ------------------------------------------------------------------ '''
''' --------------------------- READ DATA ---------------------------- '''
''' ------------------------------------------------------------------ '''

print("\nWATER PUMP COMPETITION\n")

print("Leyendo datos...")

#los ficheros .csv se han preparado previamente para sustituir ,, y "Not known" por NaN (valores perdidos)
data_x_orig = pd.read_csv('data/water_pump_tra.csv')
data_y = pd.read_csv('data/water_pump_tra_target.csv')
data_x_tst = pd.read_csv('data/water_pump_tst.csv')

print(data_x_orig.shape)
print(data_x_tst.shape)

print("Lectura completada.\n")


''' ------------------------------------------------------------------ '''
''' -------------------------- LOOK AT DATA -------------------------- '''
''' ------------------------------------------------------------------ '''

print('Viendo los datos:\n')

data_x = data_x_orig

print('num_private:')
print(data_x['num_private'].value_counts()[0:3])
print('recorded_by:')
print(data_x['recorded_by'].value_counts())
print(data_y.status_group.value_counts()/len(data_y))

data_y.status_group.value_counts().plot(kind='bar')
plt.xticks(rotation = 0)
plt.show()

print('Ejemplos con longitude = 0')
print(len(data_x.ix[data_x['longitude']==0,'longitude']))

print('Ejemplos con latitude = 0')
print(len(data_x.ix[data_x['latitude']==-0.00000002,'latitude']))

print('Ejemplos con construction_year = 0')
print(len(data_x.ix[data_x['construction_year']==0,'construction_year']))


corr = data_x.corr()
sns.heatmap (corr)
plt.xticks(rotation =45)
plt.show()

print("Valores perdidos:")
print(data_x.isnull().sum())

data_x.isnull().sum().plot.bar()
plt.show()

print('funder:\n')
print(data_x['funder'].value_counts()[0:6])
print('\ninstaller:\n')
print(data_x['installer'].value_counts()[0:6])
print('\npublic_meeting:\n')
print(data_x['public_meeting'].value_counts()[0:6])
print('\nscheme_management:\n')
print(data_x['scheme_management'].value_counts()[0:6])
print('\npermit:\n')
print(data_x['permit'].value_counts()[0:6])
print('\nsubvillage:\n')
print(data_x['subvillage'].value_counts()[0:6])
print('\nwpt_name:\n')
print(data_x['wpt_name'].value_counts()[0:6])

'''
data_x['funder'].value_counts()[0:10].plot.bar()
plt.show()
data_x['installer'].value_counts().plot.bar()
plt.show()
data_x['public_meeting'].value_counts().plot.bar()
plt.show()
data_x['scheme_management'].value_counts().plot.bar()
plt.show()
data_x['permit'].value_counts().plot.bar()
plt.show()
data_x['subvillage'].value_counts().plot.bar()
plt.show()
data_x['wpt_name'].value_counts().plot.bar()
plt.show()
'''

''' ------------------------------------------------------------------ '''
''' ------------------------- PREPROCESSING -------------------------- '''
''' ------------------------------------------------------------------ '''

print("\nPreprocesando datos...")

data_x=data_x_orig.append(data_x_tst)


''' ------------------ DROP COLUMNS ------------------ '''

print("  Borrando columnas...")
columns_to_drop = ['id', 'num_private', 'recorded_by', 'scheme_name']
data_x.drop(labels=columns_to_drop, axis=1, inplace = True)
data_y.drop(labels=['id'], axis=1,inplace = True)



''' ------------------ MISSING VALUES ------------------ '''

print("  Modificando valores nan...")
data_x['funder'] = data_x['funder'].fillna('Government Of Tanzania')
data_x['installer'] = data_x['installer'].fillna('DWE')
data_x['public_meeting'] = data_x['public_meeting'].fillna(True)
data_x['scheme_management'] = data_x['scheme_management'].fillna('VWC')
data_x['permit'] = data_x['permit'].fillna(True)
data_x['subvillage'] = data_x['subvillage'].fillna('Unknown')
data_x['wpt_name'] = data_x['wpt_name'].fillna('none')

data_x.ix[data_x['latitude']>-0.1,'latitude']=None
data_x.ix[data_x['longitude']==0,'longitude']=None
data_x["longitude"] = data_x.groupby("region_code").transform(lambda x: x.fillna(x.median())).longitude
data_x["latitude"] = data_x.groupby("region_code").transform(lambda x: x.fillna(x.median())).latitude

data_x.construction_year=pd.to_numeric(data_x.construction_year)
data_x.loc[data_x.construction_year <= 0, data_x.columns=='construction_year'] = 1950

# mean() tarda mucho, pero mejora un poco los resultados con respecto a median()
#data_x=data_x.fillna(data_x.mean())
#data_x = data_x.fillna(data_x.median())

''' ------------------ RARE VALUES ------------------ '''

print("  Etiquetando casos raros...")
columns_other = [x for x in data_x.columns if x not in ['latitude','longitude','gps_height','age','population','construction_year','month_recorder']]

for col in columns_other:
    value_counts = data_x[col].value_counts()
    lessthen = value_counts[value_counts < 20]
    listnow = data_x.installer.isin(list(lessthen.keys()))
    data_x.loc[listnow,col] = 'Others'


''' ------------------ CARTESIAN ------------------ '''

print("  Preprocesando coordenadas y distancias...")
data_x['dist'] = data_x.apply(lambda row: distancia(row['longitude'], row['latitude'], 0, 0), axis=1)
data_x['cartesian_x'] = data_x.apply(lambda row: cartesian_x(row['longitude'], row['latitude']), axis=1)
data_x['cartesian_y'] = data_x.apply(lambda row: cartesian_y(row['longitude'], row['latitude']), axis=1)
data_x.drop(labels=['longitude', 'latitude'], axis=1, inplace = True)

''' ------------------ DATES ------------------ '''

print("  Preprocesando fechas...")
data_x = date_parser(data_x)



data_x.population = data_x.population.apply(lambda x: np.log10(x+1))

print("  Convirtiendo categóricas a numéricas...")
data_x = data_x.astype(str).apply(LabelEncoder().fit_transform)

data_x_tst = data_x[len(data_x_orig):]
data_x = data_x[:len(data_x_orig)]

X = data_x.values
y = np.ravel(data_y.values)
#y = le.fit(y).transform(y)
X_tst = data_x_tst.values

print("Datos preprocesados con éxito.\n")


''' -------------------- CROSS VALIDATION -------------------- '''

'''
print("Validación cruzada:\n")

print('\nKNN\n')
knn = KNeighborsClassifier(n_neighbors=5)
cross_validation(clf=knn, X = X, y = y, cv = None, min_max_scaler = True)

print('\nXGB\n')
clf = xgb.XGBClassifier(n_estimators = 200)
cross_validation(clf, X, y)

print('\nLGB\n')
clf = lgb.LGBMClassifier(objective='binary', n_estimators=200, num_leaves=31)
cross_validation(clf, X, y)

print('\nRandomForest\n')
clf = RandomForestClassifier(n_estimators=125, max_depth = 20, random_state = 10)
cross_validation(clf, X, y)

print('\nExtraTreesClassifier\n')
clf = ExtraTreesClassifier(n_estimators = 125, max_depth = 20)
cross_validation(clf, X, y)
'''

''' -------------------- SUBMISSION 1 -------------------- '''
'''
clf = xgb.XGBClassifier(n_estimators = 200)
clf = clf.fit(X,y)

y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('data/water_pump_submissionformat.csv')
df_submission['status_group'] = y_pred_tst
df_submission.to_csv("submission1.csv", index=False)
'''
''' ---------------------------------------------------- '''

''' -------------------- SUBMISSION 2 -------------------- '''
'''
clf = RandomForestClassifier(n_estimators = 125)
clf = clf.fit(X,y)

y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('data/water_pump_submissionformat.csv')
df_submission['status_group'] = y_pred_tst
df_submission.to_csv("submission2.csv", index=False)
'''
''' ---------------------------------------------------- '''

''' -------------------- SUBMISSION 3 -------------------- '''
'''
clf = RandomForestClassifier()
clf = clf.fit(X,y)

y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('data/water_pump_submissionformat.csv')
df_submission['status_group'] = y_pred_tst
df_submission.to_csv("submission3.csv", index=False)
'''
''' ---------------------------------------------------- '''


''' -------------------- SUBMISSION 6 -------------------- '''
'''
# Eliminated features:
# 'num_private', 'recorded_by', 'region', 'scheme_name', 'scheme_management'

clf = RandomForestClassifier(max_features = 'sqrt', n_estimators = 500, random_state=10)
clf = clf.fit(X,y)

y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('data/water_pump_submissionformat.csv')
df_submission['status_group'] = y_pred_tst
df_submission.to_csv("submission6.csv", index=False)
'''
''' ---------------------------------------------------- '''


''' -------------------- SUBMISSION 8 -------------------- '''
'''
print("Submission 8")

clf = RandomForestClassifier(max_features = 'sqrt', n_estimators = 200, max_depth = 20)

clf = clf.fit(X,y)

y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('data/water_pump_submissionformat.csv')
df_submission['status_group'] = y_pred_tst
df_submission.to_csv("submission9.csv", index=False)
'''
''' ---------------------------------------------------- '''

''' -------------------- SUBMISSION 11 -------------------- '''
'''
print("Submission 11")

clf = RandomForestClassifier(n_estimators=200, max_depth = 20, random_state = 10)

clf = clf.fit(X,y)

y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('data/water_pump_submissionformat.csv')
df_submission['status_group'] = y_pred_tst
df_submission.to_csv("submission11.csv", index=False)
'''

''' -------------------- SUBMISSION 12 -------------------- '''
'''
print("Submission 12")

clf = RandomForestClassifier(n_estimators=125, max_depth = 20)

clf = clf.fit(X,y)

y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('data/water_pump_submissionformat.csv')
df_submission['status_group'] = y_pred_tst
df_submission.to_csv("submission12.csv", index=False)
'''

''' -------------------- SUBMISSION 13 -------------------- '''
'''
print("Submission 13")

fit_rf = RandomForestClassifier(max_features = 'sqrt', max_depth=20)
estimators = range(25,201,25)
param_dist = {'n_estimators': estimators}

clf= GridSearchCV(fit_rf, cv = 5, scoring = 'accuracy', param_grid=param_dist, n_jobs = 3)

clf = clf.fit(X,y)

y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('data/water_pump_submissionformat.csv')
df_submission['status_group'] = y_pred_tst
df_submission.to_csv("submission13.csv", index=False)
'''

''' -------------------- SUBMISSION 15 -------------------- '''
'''
print("Submission 15")

clf = RandomForestClassifier(n_estimators=125, max_depth = 22)

clf = clf.fit(X,y)

y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('data/water_pump_submissionformat.csv')
df_submission['status_group'] = y_pred_tst
df_submission.to_csv("submission15.csv", index=False)
'''
''' -------------------- SUBMISSION 16 -------------------- '''
'''
print("Submission 16")

clf = RandomForestClassifier(n_estimators=500)

clf = clf.fit(X,y)

y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('data/water_pump_submissionformat.csv')
df_submission['status_group'] = y_pred_tst
df_submission.to_csv("submission16.csv", index=False)

# Nota: este experimento empeora los resultados, posible sobreentrenamiento
'''

''' -------------------- SUBMISSION 17 -------------------- '''
'''
print("Submission 17")

clf = RandomForestClassifier(n_estimators=120, max_depth = 20)

clf = clf.fit(X,y)

y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('data/water_pump_submissionformat.csv')
df_submission['status_group'] = y_pred_tst
df_submission.to_csv("submission17.csv", index=False)

'''

''' -------------------- SUBMISSION 18 -------------------- '''
'''
# fillnan() with more repeated
print("Submission 18")

clf = RandomForestClassifier(n_estimators=160, max_depth = 20)

clf = clf.fit(X,y)

y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('data/water_pump_submissionformat.csv')
df_submission['status_group'] = y_pred_tst
df_submission.to_csv("submission18.csv", index=False)
'''

''' -------------------- SUBMISSION 19 -------------------- '''
'''
# fillnan() with more repeated
print("Submission 19")

clf = RandomForestClassifier(n_estimators=150, max_depth = 20)

clf = clf.fit(X,y)

y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('data/water_pump_submissionformat.csv')
df_submission['status_group'] = y_pred_tst
df_submission.to_csv("submission19.csv", index=False)
'''

''' -------------------- SUBMISSION 22 -------------------- '''
'''
print("Submission 22")

fit_rf = RandomForestClassifier(max_features = 'sqrt', max_depth=20)
estimators = range(25,201,25)
param_dist = {'n_estimators': estimators}

clf= GridSearchCV(fit_rf, cv = 5, scoring = 'accuracy', param_grid=param_dist, n_jobs = 3)

clf = clf.fit(X,y)

y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('data/water_pump_submissionformat.csv')
df_submission['status_group'] = y_pred_tst
df_submission.to_csv("submission22.csv", index=False)

best_param = clf.best_params_['n_estimators']
print ("Mejor valor para n_estimators: ", best_param)
'''
''' -------------------- SUBMISSION 23 -------------------- '''
'''
print("Submission 23")

fit_rf = RandomForestClassifier(max_features = 'sqrt', max_depth=25)
estimators = range(100,1101,25)
param_dist = {'n_estimators': estimators}

clf= GridSearchCV(fit_rf, cv = 5, scoring = 'accuracy', param_grid=param_dist, n_jobs = 3)

clf = clf.fit(X,y)

y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('data/water_pump_submissionformat.csv')
df_submission['status_group'] = y_pred_tst
df_submission.to_csv("submission23.csv", index=False)

best_param = clf.best_params_['n_estimators']
print ("Mejor valor para n_estimators: ", best_param)
'''


''' -------------------- SUBMISSION 24 -------------------- '''
'''
print("Submission 24")

clf = RandomForestClassifier(n_estimators=100, max_depth = 20)

clf = clf.fit(X,y)

y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('data/water_pump_submissionformat.csv')
df_submission['status_group'] = y_pred_tst
df_submission.to_csv("submission24.csv", index=False)

'''
''' -------------------- SUBMISSION 25 -------------------- '''
'''
print("Submission 25")

clf = RandomForestClassifier(n_estimators=150, max_depth = 20)

clf = clf.fit(X,y)

y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('data/water_pump_submissionformat.csv')
df_submission['status_group'] = y_pred_tst
df_submission.to_csv("submission25.csv", index=False)
'''


''' ------------------- FINAL SUBMISSION ------------------ '''

''' -------------------- SUBMISSION 26 -------------------- '''

print("Submission 26")

clf = RandomForestClassifier(n_estimators = 125, max_depth = 20)

clf = clf.fit(X,y)

y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('data/water_pump_submissionformat.csv')
df_submission['status_group'] = y_pred_tst
df_submission.to_csv("submission26.csv", index=False)