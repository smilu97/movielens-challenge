#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
from svdpp import Model

df_ratings = pd.read_csv('../data/ml-1m/ratings.dat', sep='::', header=None, engine='python')
arr_ratings = df_ratings.to_numpy()
user_ids   = arr_ratings[:, 0] - 1
item_ids   = arr_ratings[:, 1] - 1
ratings    = arr_ratings[:, 2]
timestamps = arr_ratings[:, 3]
n_items = 3952
n_users = 6040

def main(argv):
    epochs = 1000

    model = Model(
        n_items=n_items,
        n_users=n_users,
        n_factors=8,
        reg_all=0.0,
        rej_pass=0.0,
        scale_walk=0.1,
        sz_walk=5,
        user_ids=user_ids,
        item_ids=item_ids,
        ratings=ratings,
    )

    print('[init] Loss: {}, mse: {}'.format(model.curr_loss, model.mse()))
    for i in range(epochs):
        model.fit(steps=10)
        loss = model.curr_loss
        mse = model.mse()
        print('[step {}] Loss: {}, mse: {}'.format(i + 1, loss, mse))

if __name__ == '__main__':
    main(sys.argv)
