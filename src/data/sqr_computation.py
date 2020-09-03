# Import packages
import pandas as pd
from src.features.sqr import calculate_sqi


# Aux functions
def rename_movement_columns(df):
    old_names = ['fecha', 'presion0', 'presion1', 'presion2', 'presion3',
                 'presion4', 'presion5', 'presion6', 'presion7', 'presion8',
                 'presion9', 'presion10', 'presion11', 'presion12',
                 'tipoMovimiento']
    new_names = ['timestamp', 'pressure0', 'pressure1', 'pressure2',
                 'pressure3', 'pressure4', 'pressure5', 'pressure6',
                 'pressure7', 'pressure8', 'pressure9', 'pressure10',
                 'pressure11', 'pressure12', 'type']

    return df.rename(columns=dict(zip(old_names, new_names)))


def calculate_new_sqr(session):
    session_movements = movements[movements['idSesion'] == session['idSesiones']]
    return calculate_sqi(session['fechaEntrada'], session['fechaInicio'], session_movements)


# Read data
sessions = pd.read_parquet('data/raw/flex_sesiones.parquet')
movements = pd.read_parquet('data/raw/flex_movimientos.parquet')

# Rename cols
movements = rename_movement_columns(movements)
# Calculate new sqr
sessions['sqr'] = sessions.apply(calculate_new_sqr, axis=1)
# Save data
sessions.to_parquet('data/interim/sessions_new_sqr_flex.parquet')
