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
import sys
sys.path.append(os.path.abspath('../src/models/'))
from similarities import perfiles_similarity, pressures_similarity

# # Cargamos las dos tablas de datos
# perfiles = pd.read_parquet('data/raw/flex_perfiles_usuario.parquet')
# sesiones = pd.read_parquet('data/raw/flex_sesiones.parquet')
#
# # Filtro en perfiles valores erroneos peso y altura. Corrección formatos
# perfiles.loc[perfiles["posicion"]=="manual","posicion"] = "Manual"
#
# for col in ["altura", "peso"]:
#     perfiles[col] = perfiles[col].str.replace(',', '.')
#
# perfiles[["altura", "peso"]] = perfiles[["altura", "peso"]].apply(pd.to_numeric)
#
# perfiles_filtrado = perfiles[(perfiles["peso"]<150) &
#                              (perfiles["peso"]!=0 )& (perfiles["altura"]>100) & (perfiles["altura"]<220)]
# perfiles_filtrado = perfiles_filtrado[perfiles_filtrado["sexo"]!="Manual"] # quitamos manual en sexo y, consecuentemente, en posicion
# perfiles_filtrado.reset_index(drop=True, inplace=True)
#
# # IMC
# perfiles_filtrado['IMC'] = perfiles_filtrado['peso'] / (perfiles_filtrado['altura']/100)**2
# perfiles_filtrado['IMC_cat'] = pd.cut(perfiles_filtrado['IMC'], bins=[0, 18.5, 24.9, 29.9, 50],
#                                 include_lowest=True,labels=['Bajo peso', 'Normal', 'Sobrepeso', 'Obesidad'])
#


perfiles_sqr = pd.read_parquet('data/processed/perfiles_sqr_filtrado.parquet')
perfiles_sqr.reset_index(drop=True, inplace=True)

###############################################################################################
################                  PREPARACION DATOS CLUSTERING                 ################
###############################################################################################

# Nos quedamos con las variables que nos interesan para el clustering
df_perfiles = perfiles_sqr[["presiones","posicion","altura","peso","sexo","sqr"]]
# Datos para cluster
df_tocluster = df_perfiles[["posicion","altura","peso","sexo"]]


# Estandarizamos las variables continuas
#df_4var[["altura","peso"]] = preprocessing.scale(df_4var[["altura","peso"]],with_mean=True, with_std=True)


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


###############################################################################################
################                         ESTUDIO CLUSTERS                      ################
###############################################################################################

# Tamano clusters
df_perfiles.groupby('labels').size()

# Valores de las 4 variables en clusters
summary_groups = df_perfiles.groupby('labels').agg(
    {
        'altura': [min, max, 'mean', 'std'],
        'peso': [min, max, 'mean', 'std'],
        'sqr': [min, max, 'mean', 'std'],
        'sexo': [mode],
        'posicion': [mode, 'size']
    }
)

df_clusterh0 = df_perfiles.loc[df_perfiles['labels'] == 0, ]
df_clusterh1 = df_perfiles.loc[df_perfiles['labels'] == 1, ]
df_clusterh2 = df_perfiles.loc[df_perfiles['labels'] == 2, ]
df_clusterh3 = df_perfiles.loc[df_perfiles['labels'] == 3, ]

### Heatmap para las presiones

# Creamos 1 dataframe por cada cluster
# en cada uno pondremos por columnas las 12 posiciones
# y por filas los 6 tipos de presion
# calcularemos la frecuencia de aparacion de cada presion en cada posicion
# y pintaremos un mapa de calor para poder comparar las presiones

# Para ello hacemos una funcion previa

def presiones_df_heat(df):
    # Dado un df input, separar las presiones de la variable presiones de perfiles_usuario
    # df_pres: dataframe  de 12 columnas, una por cada posicion, 1 fila por id
    # df_completo_pres: join(df, df_pres)
    # df_pres_count: dataframe con 12 columnas (posiciones) y 6 filas (niveles de presion), conteo de valores
    # df_pres_prop: df_pres_count/total valores

    # Del df input nos quedamos con presiones
    df_pres = df['presiones']

    # Estructura para los df resultantes
    cols = ['PresPos1', 'PresPos2', 'PresPos3', 'PresPos4', 'PresPos5', 'PresPos6',
            'PresPos7', 'PresPos8', 'PresPos9', 'PresPos10', 'PresPos11', 'PresPos12']
    rows_heat = ['NivPres0', 'NivPres1', 'NivPres2', 'NivPres3', 'NivPres4', 'NivPres5']
    rows = range(len(df_pres))
    df_pres_split = pd.DataFrame(columns=cols, index=rows)
    df_pres_count = pd.DataFrame(columns=cols, index=rows_heat)

    # Separamos las presiones
    for j in range(len(df_pres)):
        pres_j = df_pres.iloc[j]
        pres_j_split = [pres_j[i:i + 1] for i in range(0, len(pres_j), 1)]
        df_pres_split.iloc[j, :] = pres_j_split

    # Para la distribucion
    k = 0
    for pos in cols:
        # print(pos)
        to_fill = list(df_pres_split.groupby(pos)[pos].size())
        if len(to_fill) != 6:
            to_fill = [sum(df_pres_split[pos] == '0'),
                       sum(df_pres_split[pos] == '1'),
                       sum(df_pres_split[pos] == '2'),
                       sum(df_pres_split[pos] == '3'),
                       sum(df_pres_split[pos] == '4'),
                       sum(df_pres_split[pos] == '5')]
        df_pres_count.iloc[:, k] = to_fill
        k += 1

    # Dividimos df_pres_count por el total de observaciones para tener la proporcion
    df_pres_prop = df_pres_count / len(df_pres)
    # Juntamos las presiones separadas al df original
    df_completo_pres = pd.concat([df, df_pres_split], axis=1, join='inner')

    return df_completo_pres, df_pres_count, df_pres_prop




# Usamos la funcion presiones_df_heat para obtener mapas de calor para los clusters

# Df para los 4 clusters
list_result = []
for h in range(len(np.unique(labels))):
    df_clusterh = df_perfiles.loc[df_perfiles['labels'] == h]
    df_pres_h, df_pres_count_h, df_pres_prop_h = presiones_df_heat(df_clusterh)
    list_result.append(df_pres_prop_h)


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



###############################################################################################
################                EVALUACION CLUSTERS EN PRESIONES               ################
###############################################################################################

def evalucion_grupos_presiones(df):
    similarities=[]
    for h in range(len(np.unique(df['labels']))):
        df_clusterh = df.loc[df_perfiles['labels'] == h, "presiones"]
        n_filas = df_clusterh.shape[0]
        similarities_h=[]
        for i in range(n_filas):
            for j in np.arange(i + 1,n_filas):
                similarities_h.append(pressures_similarity(df_clusterh.iloc[i], df_clusterh.iloc[j]))

        similarities.append(round(np.mean(similarities_h),3))

    similarities=dict(zip(np.unique(df['labels']), similarities))
    return similarities

summary_groups_presiones=evalucion_grupos_presiones(df_perfiles)





