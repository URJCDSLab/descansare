import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Cargamos las dos tablas de datos
perfiles = pd.read_parquet('data/raw/flex_perfiles_usuario.parquet')
sesiones = pd.read_parquet('data/raw/flex_sesiones.parquet')

sesiones_validas = sesiones.idPerfil.isin(set(perfiles.idPerfiles))

ultima_sesion = sesiones[sesiones_validas].sort_values(['idPerfil', 'fechaInicio'],
                                                       ascending=False).groupby('idPerfil').first()[['sqr']]

perfiles_info = perfiles.set_index('idPerfiles')

perfiles_sqr = perfiles_info.join(ultima_sesion, how='inner')

# Procesado de errores en las columnas numéricas
for col in ['altura', 'peso']:
    perfiles_sqr[col] = perfiles_sqr[col].str.replace(',', '.').astype('float')

# Filtro de peso, altura, sexo y sqr
perfiles_sqr_filtrado = perfiles_sqr[(perfiles_sqr["peso"] < 150)
                                      & (perfiles_sqr["peso"] != 0)
                                      & (perfiles_sqr["altura"] > 100)
                                      & (perfiles_sqr["altura"] < 220)
                                      & (perfiles_sqr['sexo'] != 'Manual')
                                      & (perfiles_sqr['sqr'] > 0)
                                      & (perfiles_sqr['sqr'] < 100)]

# Guardado de datos procesados
perfiles_sqr_filtrado.to_parquet(f'data/processed/perfiles_sqr_filtrado.parquet')

# Procesado de variables categóricas para el knn

le = LabelEncoder()

df_perfiles = perfiles_sqr_filtrado[["presiones",  "altura", "peso", "sqr"]].copy()

df_perfiles.loc[:, 'posicion'] = le.fit_transform(perfiles_sqr_filtrado['posicion'])
df_perfiles.loc[:, 'sexo'] = le.fit_transform(perfiles_sqr_filtrado['sexo'])

# Guardado de datos procesados para el knn
perfiles_sqr_filtrado.to_parquet(f'data/processed/perfiles_sqr_knn.parquet')



