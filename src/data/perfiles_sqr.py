import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Cargamos las dos tablas de datos
perfiles = pd.read_parquet('data/raw/flex_perfiles_usuario.parquet')
sesiones = pd.read_parquet('data/raw/flex_sesiones.parquet')

# Filtrado de sesiones válidas
# #Sesiones con perfil asociado en la tabla de perfiles
sesiones_validas = sesiones.idPerfil.isin(set(perfiles.idPerfiles))

# #Filtros de existencia fechas
mask_1 = sesiones_validas & (~sesiones.fechaEntrada.isna() & ~sesiones.fechaInicio.isna() & ~sesiones.fechaFin.isna())
# #Duración de sesión mayor de 2 horas
mask_2 = mask_1 & ((sesiones.fechaFin - sesiones.fechaInicio).apply(lambda x: x.total_seconds()) > 7200)

# #Últimos dos años
date_time_str = '01/01/19 00:00:00'
ini_date = datetime.strptime(date_time_str, '%d/%m/%y %H:%M:%S')
mask_3 = mask_2 & (sesiones.fechaInicio > ini_date)

sesiones_filtradas = sesiones[mask_3].copy()

# Transformar ids a entero
sesiones_filtradas['idPerfiles'] = sesiones_filtradas['idPerfil'].astype(int)

sesiones_filtradas['idImatt'] = sesiones_filtradas['idImatt'].astype(int)
sesiones_filtradas['idUsuario'] = sesiones_filtradas['idUsuario'].astype(int)

# Borrado de idUsuario
perfiles.drop(columns='idUsuario', inplace=True)
sesiones_filtradas.drop(columns=['idUsuario', 'idPerfil'], inplace=True)

# Agreagamos información de perfiles a las sesiones
perfiles_info = perfiles.set_index('idPerfiles')
perfiles_sqr = sesiones_filtradas.join(perfiles_info, on='idPerfiles', how='left')

# Filtramos que las presiones tengan el formato correcto
perfiles_sqr = perfiles_sqr[perfiles_sqr.presiones.apply(lambda x: len(x)) == 12]

# Refactorización de presiones
perfiles_sqr['presiones_old'] = perfiles_sqr['presiones']

perfiles_sqr['presiones'] = perfiles_sqr.presiones.apply(lambda x: x[0] + x[2] + x[4:8])

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

# #Instanciamos el labelencoder
le = LabelEncoder()

# #Copia del dataframe con las variables de interés
df_perfiles = perfiles_sqr_filtrado[["presiones", "altura", "peso", "sqr"]].copy()

# #Se crean las dos variables nuevas con la codificación adecuada
df_perfiles.loc[:, 'posicion'] = le.fit_transform(perfiles_sqr_filtrado['posicion'])
df_perfiles.loc[:, 'sexo'] = le.fit_transform(perfiles_sqr_filtrado['sexo'])

# Guardado de datos procesados para el knn
df_perfiles.to_parquet(f'data/processed/perfiles_sqr_knn.parquet')
