# -*- coding: utf-8 -*-

"""
Autor:
    Francisco Solano López Rodríguez
Fecha:
    Noviembre/2018
Contenido:
    Practica 2 Clustering
    Inteligencia de Negocio
    Grado en Ingeniería Informática
    Universidad de Granada
"""

import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import KMeans, AgglomerativeClustering,estimate_bandwidth
from sklearn.cluster import Birch,SpectralClustering,MeanShift,DBSCAN, MiniBatchKMeans
from sklearn import metrics
from sklearn import preprocessing
from math import floor
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram,ward

seed = 12345

################### FUNCIONES ###########################

def getPrediction(algorithm, X):
    t = time.time()
    cluster_predict = algorithm.fit_predict(X) 
    tiempo = time.time() - t

    return cluster_predict, tiempo

# Función para obtener las medias de cada cluster
def getMeans(dataFrame):
    return dataFrame.groupby("cluster").mean()

# Función para obtener las desviaciones de cada cluster
def getStd(dataFrame):
    return dataFrame.groupby("cluster").std()

# Función para pintar Scatter Matrix 
def DrawScatterMatrix(data, name=None, display=True, save=False):
    sns.set()
    variables = list(data)
    variables.remove('cluster')
    sns_plot = sns.pairplot(data, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25},
                            diag_kind="hist") 
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03)

    if name != None:        
        plt.title("scatter_"+name)

    # Mostrar imagen por pantalla
    if display:
        plt.show()

    # Guardar imagen en memoria
    if save:        
        if name == None:
            name = "_unknown_"
        image_name = "scatter/scatter_" + name + ".png"
        plt.savefig(image_name)
        plt.clf()
        print("Imagen guardada: ", image_name)

# Función para pintar heatmap
def DrawHeatmap(data, name = None, display=True, save = False):
    data_normal = data.apply(norm_to_zero_one)
    meanDF = getMeans(dataFrame = data_normal)
    hm = sns.heatmap(data=meanDF, linewidths=.1, cmap="Blues", annot=True, xticklabels='auto')
    plt.xticks(rotation=0)
    plt.title("heatmap_"+name)

    if name != None:        
        plt.title("heatmap_"+name)

    # Mostrar imagen por pantalla
    if display:
        plt.show()

    # Guardar imagen en memoria
    if save:
        if name == None:
            name = "_unknown_"
        image_name = "heatmap/heatmap_" + name + ".png"
        plt.savefig(image_name)
        plt.clf()
        print("Imagen guardada: ", image_name)

# Función para pintar dendograma
def DrawDendrogram(data, name = None, display=True, save = False):
    data_normal = preprocessing.normalize(data,norm='l2')
    linkage_array = ward(X_normal)

    dendrogram(linkage_array,leaf_rotation=90., leaf_font_size=5.)
    
    if name != None:        
        plt.title("dendograma_" + name)

    # Mostrar imagen por pantalla
    if display:
        plt.show()

    # Guardar imagen en memoria
    if save:
        if name == None:
            name = "_unknown_"
        image_name = "dendrogram/dendrogram_" + name + ".png"
        plt.savefig(image_name)
        plt.clf()
        print("Imagen guardada: ", image_name)

def dataFrameResultados(algoritmos, num_cluster, metrics_CH, metrics_SC, tiempos):
    df_algo = pd.DataFrame(algoritmos, columns=['Algoritmo'])
    df_nc = pd.DataFrame(num_cluster, columns=['Num. Clusters'])
    df_CH = pd.DataFrame(metrics_CH, columns=['CH'])
    df_SC = pd.DataFrame(metrics_SC, columns=['SH'])
    df_t = pd.DataFrame(tiempos, columns=['Tiempo'])

    resultados = pd.concat([df_algo, df_nc, df_CH, df_SC, df_t], axis=1)

    return resultados

def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())


def executeClustering(algorithms, X, caso):

    f = open("caso_" + str(caso) + ".txt", 'w')

    X_normal = X.apply(norm_to_zero_one)

    names = []
    num_cluster = []
    metrics_CH = []
    metrics_SC = []
    tiempos = []

    print("\nCaso de estudio ", caso, ", tamaño: ", len(X))
    f.write("\nCaso de estudio " + str(caso) + ", tamaño: " + str(len(X)))

    for algorithm, name_algorithm in algorithms:

        print("\n----------------------------------------\n")
        print("Ejecutando algoritmo: ", name_algorithm, "\n")
        f.write("\n--------------------------------------\n")
        f.write("Ejecutando algoritmo: " + name_algorithm + "\n")        
        # Ejecución algoritmo clustering
        cluster_predict, tiempo = getPrediction(algorithm, X_normal)

        # Pasar las predicciones a dataFrame
        clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])

        print("Tamaño de cada cluster:")
        f.write("\nTamaño de cada cluster:\n")
        size=clusters['cluster'].value_counts()

        for num,i in size.iteritems():
           print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
           f.write('%s: %5d (%5.2f%%)\n' % (num,i,100*i/len(clusters)))
        print()

        # Obtener los resultados de las métricas
        metric_CH = metrics.calinski_harabaz_score(X_normal, cluster_predict)
        metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', 
                                         sample_size=floor(0.2*len(X)), random_state=seed)

        # Guardamos el nombre del algoritmo, número de cluster, 
        # los tiempos y las métricas para la posterior comparacion 
        names.append(name_algorithm)   
        num_cluster.append(len(set(cluster_predict)))
        metrics_CH.append(metric_CH)
        metrics_SC.append(metric_SC)
        tiempos.append(tiempo)

        # Se añade la asignación de clusters como columna a X
        X_cluster = pd.concat([X, clusters], axis=1)
        X_normal_cluster = pd.concat([X_normal, clusters], axis=1)

        name = "caso_" + str(caso) + "_" + name_algorithm  

        # Pintamos el scatter matrix
        DrawScatterMatrix(data = X_cluster, name = name, display = False, save = True)

        # Pintamos el heatmap
        DrawHeatmap(data = X_cluster, name = name, display = False, save = True)

        # DataFrame con la media de cada característica en cada cluster
        meanDF = getMeans(dataFrame = X_cluster)
        print()
        print(meanDF)
        f.write(meanDF.to_string())

        # Si el algoritmo es AgglomerativeClustering pintamos el dendograma
        if name_algorithm == 'AC':
            DrawDendrogram(data = X_cluster, name = name, display = False, save = True)


    resultados = dataFrameResultados(names, num_cluster, metrics_CH, metrics_SC, tiempos)

    print("\n**************************************\n")
    print(resultados.to_string())
    print("\n**************************************\n")

    f.write("\n**************************************\n")
    f.write(resultados.to_string())
    f.write("\n**************************************\n")

    f.close()


#########################################################

# Lectura datos

print("Leyendo el conjunto de datos...")
censo = pd.read_csv('censo_granada.csv')
censo = censo.replace(np.NaN,0) 
print("Lectura completada.")


###### CASOS DE ESTUDIO ######

#-------- CASO 1 --------

casado = 2
hombre = 1
mujer = 6

subset = censo.loc[(censo['EDAD']>=20) & (censo['EDAD']<=50) & (censo['SEXO']==mujer)]
usadas = ['EDAD', 'NPFAM', 'HM5', 'H0515']
X = subset[usadas]
X_normal = preprocessing.normalize(X, norm='l2')

#-------- CASO 2 --------

subset_2 = censo.loc[(censo['EDAD']>=20) & (censo['EDAD']<=50) & (censo['SEXO']==hombre)]
usadas_2 = ['EDAD', 'NPFAM', 'HM5', 'H0515']
X_2 = subset_2[usadas_2]
X_normal_2 = X_2.apply(norm_to_zero_one)

#-------- CASO 3 --------

subset_3 = censo.loc[(censo['EDAD']>=20) & (censo['EDAD']<=50) & (censo['SEXO']==mujer)]
usadas_3 = ['EDAD', 'NPFAM', 'NHIJOS', 'ESREAL']
X_3 = subset_3[usadas_3]
X_normal_3 = X_3.apply(norm_to_zero_one)

###############################

# Obtener la correlación entre las variables
'''
correlation = X.corr()
sns.heatmap(correlation, square = True)
plt.show()
'''

#################### Algoritmos #####################

random_seed = 123

k_means = KMeans(init='k-means++', n_clusters=5, n_init=5, random_state=random_seed)

agglo=AgglomerativeClustering(n_clusters=5,linkage="ward")

meanshift = MeanShift(bin_seeding=True)

miniBatchKMeans = MiniBatchKMeans(init='k-means++',n_clusters=4, n_init=5, max_no_improvement=10, verbose=0, random_state=random_seed)

dbscan = DBSCAN(eps=0.2)

dbscan2 = DBSCAN(eps=0.1)

algorithms = [(k_means, "KMeans"),
              (agglo, "AC"),
              (meanshift, "MeanShift"), 
              (miniBatchKMeans, "MiniBatchKM"),
              (dbscan, "DBSCAN")]

algorithms2 = [(k_means, "KMeans"),
              (agglo, "AC"),
              (meanshift, "MeanShift"), 
              (miniBatchKMeans, "MiniBatchKM"),
              (dbscan2, "DBSCAN2")]


# Kmeans con diferentes números de cluster

algorithm_kmeans = []

for i in range(5,9):
    kmeans_i = KMeans(init='k-means++', n_clusters=i, n_init=5)
    algorithm_kmeans.append((kmeans_i, "KMeans_" + str(i)))

# AgglomerativeClustering con diferentes números de cluster

algorithm_AC = []

for i in range(5,9):
    agglo_i = AgglomerativeClustering(n_clusters=i,linkage="ward")
    algorithm_AC.append((agglo_i, "AC_" + str(i)))

# MiniBatchKmeans con diferentes números de cluster

algorithm_miniBatch = []

for i in range(5,9):
    miniBatch_i = MiniBatchKMeans(init='k-means++',n_clusters=i, n_init=5, max_no_improvement=10, verbose=0, random_state=random_seed)
    algorithm_miniBatch.append((miniBatch_i, "MiniBatchKM_" + str(i)))

#-----------------------------------------------------#

# EJECUCIÓN CASO 1
executeClustering(algorithms, X, 1)
executeClustering(algorithm_kmeans, X, 1.1)
executeClustering(algorithm_AC, X, 1.2)

# EJECUCIÓN CASO 2
executeClustering(algorithms, X_2, 2)
executeClustering(algorithm_kmeans, X_2, 2.1)
executeClustering(algorithm_miniBatch, X_2, 2.2)

# EJECUCIÓN CASO 3
executeClustering(algorithms2, X_3, 3)
executeClustering(algorithm_kmeans, X_3, 3.1)
executeClustering(algorithm_miniBatch, X_3, 3.2)


