import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Embedding, Reshape, Dot
from tensorflow.keras.regularizers import L2

class Model(tf.keras.Model):
    def __init__(
        self,
        n_items,
        n_users,
        n_factors,
        reg_all,
        mu
    ):
        super(Model, self).__init__()
        self.n_items = n_items
        self.n_users = n_users
        self.n_factors = n_factors
        self.reg_all = reg_all
        self.mu = mu

        self.embed_latent_item = Embedding(n_items, n_factors, input_length=1, embeddings_regularizer=L2(reg_all), embeddings_initializer='uniform')
        self.embed_latent_user = Embedding(n_users, n_factors, input_length=1, embeddings_regularizer=L2(reg_all), embeddings_initializer='uniform')
        self.embed_bias_item = Embedding(n_items, 1, input_length=1, embeddings_regularizer=L2(reg_all), embeddings_initializer='zeros')
        self.embed_bias_user = Embedding(n_users, 1, input_length=1, embeddings_regularizer=L2(reg_all), embeddings_initializer='zeros')
        self.dot = Dot(axes=(1, 1))

        self.reshape_uf = Reshape((n_factors,))
        self.reshape_if = Reshape((n_factors,))
        self.reshape_ub = Reshape(())
        self.reshape_ib = Reshape(())
    
    def call(self, inputs):
        user_ids, item_ids = tf.split(inputs, num_or_size_splits=2, axis=1)

        u = self.reshape_uf(self.embed_latent_user(user_ids))
        i = self.reshape_if(self.embed_latent_item(item_ids))
        f = tf.math.reduce_sum(u*i, axis=1)
        # f = self.dot([u, i])

        ub = self.reshape_ub(self.embed_bias_user(user_ids))
        ib = self.reshape_ib(self.embed_bias_item(item_ids))

        return f + ub + ib + self.mu
