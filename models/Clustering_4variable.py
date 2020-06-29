### CLUSTERING
from sklearn import preprocessing
from statistics import mode
import gower
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Cargamos las dos tablas de datos
perfiles = pd.read_parquet('data/raw/flex_perfiles_usuario.parquet')
sesiones = pd.read_parquet('data/raw/flex_sesiones.parquet')

# Filtro en perfiles valores erroneos peso y altura. Corrección formatos
perfiles.loc[perfiles["posicion"]=="manual","posicion"] = "Manual"

for col in ["altura", "peso"]:
    perfiles[col] = perfiles[col].str.replace(',', '.')

perfiles[["altura", "peso"]] = perfiles[["altura", "peso"]].apply(pd.to_numeric)

perfiles_filtrado = perfiles[(perfiles["activo"]==1) &(perfiles["peso"]<150) &
                             (perfiles["peso"]!=0 )& (perfiles["altura"]>100) & (perfiles["altura"]<220)]
perfiles_filtrado.reset_index(drop=True, inplace=True)


###############################################################################################
################                  PREPARACION DATOS CLUSTERING                 ################
###############################################################################################

# Nos quedamos con las variables que nos interesan para el clustering
df_5var = perfiles_filtrado[["presiones","posicion","altura","peso","sexo"]]
df_5var = df_5var[df_5var["sexo"]!="Manual"] # quitamos manual en sexo y, consecuentemente, en posicion
df_4var = df_5var[["posicion","altura","peso","sexo"]]

# Damos valores numéricos
#df_4var.loc[df_4var["posicion"] == "Lateral","posicion"] = 1
#df_4var.loc[(df_4var["posicion"] == "Supine"),"posicion"] = 0
#df_4var.loc[df_4var["sexo"] == "Female","sexo"] = 1
#df_4var.loc[(df_4var["sexo"] == "Male"),"sexo"] = 0

# Estandarizamos las variables continuas
df_4var[["altura","peso"]] = preprocessing.scale(df_4var[["altura","peso"]],with_mean=True, with_std=True)


###############################################################################################
################                        MATRIZ DISTANCIAS                      ################
###############################################################################################
# Como tenemos variables mixtas utilizamos, por ahora, la distancia de Gower
gd = gower.gower_matrix(df_4var, cat_features = [True,False,False,True])


###############################################################################################
################                     AGGLOMERATIVE CLUSTERING                  ################
###############################################################################################

dendrogram = sch.dendrogram(sch.linkage(gd, method='ward'))
plt.show() # 4 clusters

model = AgglomerativeClustering(n_clusters=4, affinity='precomputed',linkage='average').fit(gd)
labels = model.labels_

# Guardamos las etiquetas en el df
df_5var["labels"] = labels


###############################################################################################
################                         ESTUDIO CLUSTERS                      ################
###############################################################################################

# Tamano clusters
df_5var.groupby('labels').size()

# Valores de las 4 variables en clusters
df_5var.groupby('labels').agg(
    {
        'altura': [min, max, 'mean', 'std'],
        'peso': [min, max, 'mean', 'std'],
        'sexo': [mode],
        'posicion': [mode, 'size']
    }
)



### Heatmap para las presiones

# Df para los 4 clusters
cols = ['Pos1', 'Pos2', 'Pos3', 'Pos4', 'Pos5', 'Pos6',
        'Pos7', 'Pos8', 'Pos9', 'Pos10', 'Pos11', 'Pos12']
rows_heat = ['Pres0', 'Pres1', 'Pres2', 'Pres3', 'Pres4', 'Pres5']
list_result = []

for h in range(len(np.unique(labels))):
    df_pres = df_5var.loc[df_5var['labels'] == h,'presiones']
    rows = range(len(df_pres))
    df_pres_distr = pd.DataFrame(columns=cols, index=rows)

    # Separamos las presiones
    for j in range(len(df_pres)):
        #print(j)
        pres_ejem = df_pres.iloc[j]
        pres_ejem_split = [pres_ejem[i:i + 1] for i in range(0, len(pres_ejem), 1)]
        df_pres_distr.iloc[j, :] = pres_ejem_split

    # Rellenamos para el heatmap
    df_press_heat = pd.DataFrame(columns=cols, index=rows_heat)
    k = 0
    for pos in cols:
        #print(pos)
        to_fill = list(df_pres_distr.groupby(pos)[pos].size())
        if len(to_fill) != 6:
            to_fill = [sum(df_pres_distr[pos] == '0'),
                       sum(df_pres_distr[pos] == '1'),
                       sum(df_pres_distr[pos] == '2'),
                       sum(df_pres_distr[pos] == '3'),
                       sum(df_pres_distr[pos] == '4'),
                       sum(df_pres_distr[pos] == '5')]
        df_press_heat.iloc[:, k] = to_fill
        k += 1

    list_result.append(df_press_heat)

df_press_heat0 = list_result[0]
df_press_heat1 = list_result[1]
df_press_heat2 = list_result[2]
df_press_heat3 = list_result[3]


ax = sns.heatmap(df_press_heat0, linewidth=0.5)
plt.show()

ax = sns.heatmap(df_press_heat1, linewidth=0.5)
plt.show()

ax = sns.heatmap(df_press_heat2, linewidth=0.5)
plt.show()

ax = sns.heatmap(df_press_heat3, linewidth=0.5)
plt.show()