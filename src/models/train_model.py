import pickle
import numpy as np
import pandas as pd
from src.models.knn_p import Knn_p

perfiles = pd.read_parquet('../../data/processed/perfiles_sqr_knn.parquet')
# Define knn parameters
knn = Knn_p(k=11, weights=np.ones(4) * 0.25)
# Fit model
knn.fit(perfiles[['altura', 'peso', 'posicion', 'sexo']].values, perfiles["presiones"].values, perfiles["sqr"].values)
# Serialize model
file1 = open('../../models/knn.p', 'wb')
file2 = open('../../dashboard/model/models/knn.p', 'wb')
pickle.dump(knn, file1, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(knn, file2, protocol=pickle.HIGHEST_PROTOCOL)

