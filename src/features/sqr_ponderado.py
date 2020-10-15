import pandas as pd
import numpy as np


def sqr_ponderado(data, drop=True):
    data['notaUsuario'] = data['notaUsuario']*10

    freq_notas_perfiles = data[data['notaUsuario'].notnull()]['idPerfiles'].value_counts()

    freq_sesiones = data['idPerfiles'].value_counts()

    freq_sesiones.name = 'freq_sesiones'

    freq_notas_perfiles_validos = freq_notas_perfiles[freq_notas_perfiles > 2]

    freq_notas_perfiles_validos.name = 'freq_notas'

    df_notas = pd.concat([freq_sesiones, freq_notas_perfiles_validos], axis=1)

    df_notas = df_notas[df_notas.freq_notas.notnull()]

    df_notas['freq_rel'] = np.round(df_notas.freq_notas / df_notas.freq_sesiones,2)

    new_data = pd.merge(data, df_notas['freq_rel'], how='left', left_on='idPerfiles', right_index=True)

    new_data['sqr_old'] = new_data['sqr']

    new_data.loc[(new_data['freq_rel'].notnull()) & (new_data['notaUsuario'].notnull()), 'sqr'] = new_data['freq_rel']*new_data['notaUsuario'] + (1-new_data['freq_rel'])*new_data['sqr_old']

    if drop:
        new_data.drop(columns=('freq_rel', 'old_sqr'))

    return new_data





