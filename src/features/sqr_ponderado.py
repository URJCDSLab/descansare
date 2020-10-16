import pandas as pd
import numpy as np


def sqr_ponderado(data, drop_cols=True, min_freq=3):
    """This function computed the Weighted SQR, by using the users' sleep quality notes.

    Args:
        data (DataFrame): DataFrame of users sleep sessions, with at least the following columns:

        idPerfiles, notaUsuario and sqr

        drop_cols (bool, optional): Defaults to True. If set to False
        initial SQR (sqr_old) and notes relative frequency (freq_rel) are provided
        min_freq (int, optional): Defaults to 3. Minimum number of notes to take
        into account user opinion.


    Returns:
        DataFrame: Same structure of data, where the column sqr is the new weighted sqr.
    """

    df = data.copy()

    df['notaUsuario'] = df['notaUsuario']*10

    freq_notas_perfiles = df[df['notaUsuario'].notnull()]['idPerfiles'].value_counts()

    freq_sesiones = df['idPerfiles'].value_counts()

    freq_sesiones.name = 'freq_sesiones'

    freq_notas_perfiles_validos = freq_notas_perfiles[freq_notas_perfiles > min_freq-1]

    freq_notas_perfiles_validos.name = 'freq_notas'

    df_notas = pd.concat([freq_sesiones, freq_notas_perfiles_validos], axis=1)

    df_notas = df_notas[df_notas.freq_notas.notnull()]

    df_notas['freq_rel'] = np.round(df_notas.freq_notas / df_notas.freq_sesiones,2)

    new_df = pd.merge(df, df_notas['freq_rel'], how='left', left_on='idPerfiles', right_index=True)

    new_df['sqr_old'] = new_df['sqr']

    new_df.loc[(new_df['freq_rel'].notnull()) & (new_df['notaUsuario'].notnull()), 'sqr'] = new_df['freq_rel']*new_df['notaUsuario'] + (1-new_df['freq_rel'])*new_df['sqr_old']

    if drop_cols:
        new_df.drop(columns=['freq_rel', 'sqr_old'])

    return new_df





