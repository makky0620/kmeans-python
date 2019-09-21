import numpy as np
import numpy.random as random
from collections import Counter


class KMeans:
    def __init__(self, n_cluster: int, max_iter: int = 300):
        super().__init__()
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.cluster_centers_ = None

    def fit(self, X):
        # initialization of centroid
        max_x, min_x = np.max(X[:, 0]), np.min(X[:, 0])
        max_y, min_y = np.max(X[:, 1]), np.min(X[:, 1])
        self.cluster_centers_ = np.c_[random.uniform(low=min_x, high=max_x, size=self.n_cluster),
                                      random.uniform(low=min_y, high=max_y, size=self.n_cluster)]

        N = X.shape[0]
        for _iter in range(self.max_iter):
            # step 1
            # それぞれの点がどのセントロイドに近いかでクラスタリング
            r = np.zeros(N)
            for i in range(N):
                r[i] = np.argmin(
                    [np.linalg.norm(X[i] - self.cluster_centers_[k]) for k in range(self.n_cluster)])

            # step 2
            # step 1でクラスタリングされた点の平均値でセントロイドを更新
            cnt = dict(Counter(r))
            N_k = [cnt[k] for k in range(self.n_cluster)]
            cluster_centers_prev = self.cluster_centers_.copy()
            self.cluster_centers_ = np.asanyarray(
                [np.sum(X[r == k], axis=0) / N_k[k] for k in range(self.n_cluster)])

            # セントロイドの更新前と更新後の差を見て、小さかったら終了
            diff = self.cluster_centers_ - cluster_centers_prev
            if np.abs(np.sum(diff)) < 0.0001:
                print('completed')
                return r
