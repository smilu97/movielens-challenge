#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
from svdpp import Model

df_ratings = pd.read_csv('../data/ml-1m/ratings.dat', sep='::', header=None, engine='python')
arr_ratings = df_ratings.to_numpy()
x = arr_ratings[:,0:2]
x -= 1
y    = arr_ratings[:, 2]
timestamps = arr_ratings[:, 3]
n_items = 3952
n_users = 6040

def main(argv):
    epochs = 1000
    batch_size = 10000
    n = y.shape[0]
    steps = n // batch_size

    model = Model(
        n_items=n_items,
        n_users=n_users,
        n_factors=64,
        reg_all=0.001,
        mu=np.mean(y),
    )
    model.compile(optimizer='Adam', loss='mse')
    model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.1)


if __name__ == '__main__':
    main(sys.argv)
