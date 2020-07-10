### CLUSTERING Y FUNCIONES PARA EVALUAR LOS CLUSTERS
from sklearn import preprocessing
from statistics import mode
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.models.similarities import perfiles_similarity



perfiles_sqr = pd.read_parquet('data/processed/perfiles_sqr_filtrado.parquet')
perfiles_sqr.reset_index(drop=True, inplace=True)

###############################################################################################
################                  PREPARACION DATOS CLUSTERING                 ################
###############################################################################################

# Nos quedamos con las variables que nos interesan para el clustering
df_perfiles = perfiles_sqr[["presiones","posicion","altura","peso","sexo","sqr"]]
summary_perfiles_df_total = df_perfiles.groupby("presiones").agg({'sqr': [min, max, 'mean', 'std','size']})
len(summary_perfiles_df_total) # 185 presiones diferentes
summary_perfiles_df_total = summary_perfiles_df_total[summary_perfiles_df_total['sqr']['size']>5]


# Datos para cluster
df_tocluster = df_perfiles[["posicion","altura","peso","sexo"]]


###############################################################################################
################                        MATRIZ DISTANCIAS                      ################
###############################################################################################
# Utilizamos la que se obtiene con perfiles_similarity
weights=[0.25,0.25,0.25,0.25]
dis_matrix = perfiles_similarity(df_tocluster,weights)


###############################################################################################
################                     AGGLOMERATIVE CLUSTERING                  ################
###############################################################################################

dendrogram = sch.dendrogram(sch.linkage(dis_matrix, method='average'))
plt.show() # 4 clusters

model = AgglomerativeClustering(n_clusters=4, affinity='precomputed',linkage='average').fit(dis_matrix)
labels = model.labels_

# Guardamos las etiquetas en el df
df_perfiles["labels"] = labels

# Guardado de datos intermedios
df_perfiles.to_parquet(f'data/interim/perfiles_clusters.parquet')



