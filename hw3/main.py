import pandas as pd
import numpy as np

feats = 30

df = pd.read_csv('knn_train.csv', index_col=False, header=None)
df = df.as_matrix()
x = np.delete(df, 0, axis=1)
y = np.delete(df, range(1, df.shape[1]), axis=1)
