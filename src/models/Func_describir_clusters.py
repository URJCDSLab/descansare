import pandas as pd
import numpy as np
from src.models.similarities import pressures_similarity

###############################################################################################
################                FUNCIONES DESCRIPTIVAS CLUSTERS                ################
###############################################################################################
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

# Similaridad media de las presiones en cada cluster
def evaluacion_grupos_similaridad_presiones(df):
    similarities=[]
    for h in range(len(np.unique(df['labels']))):
        df_clusterh = df.loc[df['labels'] == h, "presiones"]
        n_filas = df_clusterh.shape[0]
        similarities_h=[]
        for i in range(n_filas):
            for j in np.arange(i + 1,n_filas):
                similarities_h.append(pressures_similarity(df_clusterh.iloc[i], df_clusterh.iloc[j]))

        similarities.append(round(np.mean(similarities_h),3))

    similarities=dict(zip(np.unique(df['labels']), similarities))
    return similarities

# Estudio de las presiones de cada cluster
def evaluacion_grupos_presiones(df,porcentaje):
    presiones=[]
    for h in range(len(np.unique(df['labels']))):
        df_clusterh = df.loc[df['labels'] == h, ]
        n_filas = df_clusterh.shape[0]
        presiones_h=df_clusterh.groupby("presiones").agg({'sqr': [min, max, 'mean', 'std',"size"]})
        presiones_h=presiones_h[presiones_h['sqr']['size']>n_filas*porcentaje]
        presiones_h=presiones_h.sort_values(('sqr','size'),ascending=False)
        presiones.append(presiones_h)

    presiones=dict(zip(np.unique(df['labels']), presiones))
    return presiones


