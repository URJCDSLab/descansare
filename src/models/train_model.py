import pickle
import numpy as np
import pandas as pd
from src.models.knn import Knn

# Read data
perfiles = pd.read_parquet('data/processed/perfiles_sqr_knn.parquet')
# Define knn parameters
knn = Knn(k=11, weights=np.ones(4)*0.25)
# Fit model
knn.fit(perfiles[['altura', 'peso', 'posicion', 'sexo']].values, perfiles["presiones"].values, perfiles["sqr"].values)
# Serialize model
pickle.dump(knn, open('models/knn.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(knn, open('dashboard/model/knn.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
