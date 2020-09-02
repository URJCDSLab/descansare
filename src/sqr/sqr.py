from random import choices

import numpy as np
import pandas as pd

MOV_TYPES = ['LIGHT', 'MEDIUM_1', 'MEDIUM_2', 'STRONG']
UP_TYPE = 'UP'
DOWN_TYPE = 'DOWN'

def calculate_awake_time_score(data_df):
    aux = data_df[data_df['type'].isin([UP_TYPE, DOWN_TYPE])]
    aux = aux[~(
        (aux['type'] == DOWN_TYPE) & (aux['type'].shift() == DOWN_TYPE) |
        (aux['type'] == UP_TYPE) & (aux['type'].shift(-1) == UP_TYPE)
    )]
    aux['diff'] = aux.index - aux.index.array.shift()
    aux = aux[aux['type'] == DOWN_TYPE]
    # aplicamos multiplicadores a las diff segun la etapa del sueno
    # primera 1.5 horas
    aux.loc[(aux.index >= 0) & (aux.index < 5400), 'diff'] *= 1.5 / 60.0
    # 1.5 -> 2.5 horas
    aux.loc[(aux.index >= 5400) & (aux.index < 9000), 'diff'] *= 1.5 / 60.0
    # 2.5 -> 3.5 horas
    aux.loc[(aux.index >= 9000) & (aux.index < 12600), 'diff'] *= 2.0 / 60.0
    # 3.5 -> 4.5 horas
    aux.loc[(aux.index >= 12600) & (aux.index < 16200), 'diff'] *= 2.0 / 60.0
    # 4.5 -> 5.5 horas
    aux.loc[(aux.index >= 16200) & (aux.index < 19800), 'diff'] *= 1.5 / 60.0
    # 5.5 -> 6.5 horas
    aux.loc[(aux.index >= 19800) & (aux.index < 23400), 'diff'] *= 1.5 / 60.0
    # 6.5 -> 7.5 horas
    aux.loc[(aux.index >= 23400) & (aux.index < 27000), 'diff'] *= 1.0 / 60.0
    # 7.5 -> 8 horas
    aux.loc[(aux.index >= 27000) & (aux.index < 28800), 'diff'] *= 0.5 / 60.0
    # Calculamos el out of bed factor 
    out_of_bed_factor = aux['diff'].sum()
    # puntuacion_tiempo_despierto = ((-15*outOfBedFactor)/15)+15 (por 15 entre 15 ??)
    awake_time_score = max([0, ((-15 * out_of_bed_factor) / 15) + 15])
    
    return awake_time_score


def calculate_latency_time_score(start_time, sleep_start_time):
    diff = (sleep_start_time - start_time).seconds
    latency_time_score = max([0, ((-5 * diff / 60) / 60) + 5])
    
    return latency_time_score


def calculate_movements_score(data_df):
    # Calculo de las penalizaciones por movimiento
    # primera 1.5 horas
    data_df.loc[(data_df.index >= 0) & (data_df.index < 5400) & (data_df['type'] == 'MEDIUM_2'), 'sqi'] = 3
    data_df.loc[(data_df.index >= 0) & (data_df.index < 5400) & (data_df['type'] == 'STRONG'), 'sqi'] = 5
    # 1.5 -> 2.5 horas
    data_df.loc[(data_df.index >= 5400) & (data_df.index < 9000) & (data_df['type'] == 'MEDIUM_2'), 'sqi'] = 3
    data_df.loc[(data_df.index >= 5400) & (data_df.index < 9000) & (data_df['type'] == 'STRONG'), 'sqi'] = 4
    # 2.5 -> 3.5 horas
    data_df.loc[(data_df.index >= 9000) & (data_df.index < 12600) & (data_df['type'] == 'MEDIUM_2'), 'sqi'] = 2
    data_df.loc[(data_df.index >= 9000) & (data_df.index < 12600) & (data_df['type'] == 'STRONG'), 'sqi'] = 4
    # 3.5 -> 4.5 horas
    data_df.loc[(data_df.index >= 12600) & (data_df.index < 16200) & (data_df['type'] == 'MEDIUM_2'), 'sqi'] = 2
    data_df.loc[(data_df.index >= 12600) & (data_df.index < 16200) & (data_df['type'] == 'STRONG'), 'sqi'] = 3
    # 4.5 -> 5.5 horas
    data_df.loc[(data_df.index >= 16200) & (data_df.index < 19800) & (data_df['type'] == 'MEDIUM_2'), 'sqi'] = 1
    data_df.loc[(data_df.index >= 16200) & (data_df.index < 19800) & (data_df['type'] == 'STRONG'), 'sqi'] = 3
    # 5.5 -> 6.5 horas
    data_df.loc[(data_df.index >= 19800) & (data_df.index < 23400) & (data_df['type'] == 'MEDIUM_2'), 'sqi'] = 1
    data_df.loc[(data_df.index >= 19800) & (data_df.index < 23400) & (data_df['type'] == 'STRONG'), 'sqi'] = 2
    # 6.5 -> 7.5 horas
    data_df.loc[(data_df.index >= 23400) & (data_df.index < 27000) & (data_df['type'] == 'STRONG'), 'sqi'] = 2
    # 7.5 -> 8 horas
    data_df.loc[(data_df.index >= 27000) & (data_df.index < 28800) & (data_df['type'] == 'STRONG'), 'sqi'] = 1
    # Calculo HORAS dormidas (last_move en php)
    hours = data_df.index.max() / 3600
    # calculo sqi total
    sqi = data_df['sqi'].sum() / hours
    movements_score = max([0, ((-8 * sqi) / 3) + 80])
    
    return movements_score


def calculate_sleep_hours_factor(data_df):
    hours = data_df.index.max() / 3600
    sleep_hours_factor = ((10 * hours) + 20) / 100
    
    return sleep_hours_factor

def preprocess_data(sleep_start_time, movement_df):
    data_df = movement_df[['timestamp', 'type']].copy()
    data_df = data_df.set_index('timestamp')
    # filtramos despues de dormir
    # creamos una columna para apuntar el sqi de cada movimiento. Por defecto 0
    data_df['sqi'] = 0
    data_df.index = (data_df.index - sleep_start_time).total_seconds()
    data_df = data_df[data_df.index > 0]
    
    return data_df

def calculate_sqi(start_time, sleep_start_time, movement_df):
    ''' entre 0-100''' 
    data_df = preprocess_data(sleep_start_time, movement_df)
    awake_time_score = calculate_awake_time_score(data_df)
    latency_time_score = calculate_latency_time_score(start_time, sleep_start_time)
    movements_score = calculate_movements_score(data_df)
    sleep_hours_factor = calculate_sleep_hours_factor(data_df)
    final_sqi = awake_time_score + latency_time_score + (movements_score * sleep_hours_factor)
    
    return final_sqi