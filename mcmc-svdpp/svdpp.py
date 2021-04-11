import numpy as np

class Model:
    def __init__(
        self,
        n_items,
        n_users,
        n_factors,
        reg_all,
        rej_pass,
        scale_walk,
        sz_walk,
        user_ids,
        item_ids,
        ratings
    ):
        self.n_items = n_items
        self.n_users = n_users
        self.n_factors = n_factors
        self.reg_all = reg_all
        self.scale_walk = scale_walk
        self.sz_walk = sz_walk
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        self.rej_pass = rej_pass

        self.params = np.zeros((n_items + n_users) * n_factors + n_items + n_users + 1)

        if self.sz_walk == -1:
            self.sz_walk = self.params.shape[0]

        div0 = n_items * n_factors
        div1 = div0 + n_users * n_factors
        div2 = div1 + n_items
        div3 = div2 + n_users
        div4 = div3 + 1

        self.p_items = self.params[0:div0].reshape((n_items, n_factors))
        self.p_users = self.params[div0:div1].reshape((n_users, n_factors))
        self.b_items = self.params[div1:div2]
        self.b_users = self.params[div2:div3]
        self.b       = self.params[div3:div4].reshape(())

        self.curr_loss = self.estimate_loss()
        self.params[-1] = np.mean(ratings)
    
    def mse(self):
        e = self.predict() - self.ratings
        return np.sqrt(np.mean(np.square(e)))

    def predict(self):
        u_factors = self.p_users[self.user_ids]
        i_factors = self.p_items[self.item_ids]
        s = np.sum(u_factors * i_factors, axis=1)
        s += self.b_users[self.user_ids]
        s += self.b_items[self.item_ids]
        s += self.b
        return s
    
    def estimate_loss(self):
        e = self.predict() - self.ratings
        e = np.mean(np.square(e))

        reg  = np.sum(np.square(self.b_items))
        reg += np.sum(np.square(self.b_users))
        reg += np.sum(np.square(self.p_items))
        reg += np.sum(np.square(self.p_users))
        reg *= self.reg_all

        return e + reg

    def _compare_loss(self, next_loss):
        if next_loss < self.curr_loss:
            return True
        else:
            u = np.random.rand() * self.rej_pass + 1
            f = next_loss / self.curr_loss
            return f < u
    
    def _walk_random(self, i_diff):
        v_diff = np.random.normal(scale=self.scale_walk, size=(self.sz_walk,))
        self.params[i_diff] += v_diff
        next_loss = self.estimate_loss()
        if self._compare_loss(next_loss):
            self.curr_loss = next_loss
        else:
            self.params[i_diff] -= v_diff

    def fit(self, steps=10):
        i_diff = np.random.choice(np.arange(self.params.shape[0]), size=(self.sz_walk,))
        for _ in range(steps):
            self._walk_random(i_diff)
