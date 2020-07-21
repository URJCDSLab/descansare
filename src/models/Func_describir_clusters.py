import pandas as pd
import numpy as np
from src.models.similarities import pressures_similarity

###############################################################################################
################                FUNCIONES DESCRIPTIVAS CLUSTERS                ################
###############################################################################################

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


