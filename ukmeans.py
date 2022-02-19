import numpy as np


class UKMeans(object):
    """
    Unsupervised K-Means clustering algorithm

    reference: https://ieeexplore.ieee.org/document/9072123
    """

    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon
        self.t = 0  # iteration

        self.record = []  # record the history of the model

        # learning rates
        self.gamma = 1  # γ
        self.beta = 1  # β

        self.z = None  # cluster assignment, z_i_k is 1 if x_i is in cluster k
        self.n_centers = None  # number of centers (c)
        self.centroids = None  # centroids (a)
        self.alpha = None  # the probability of one data point belonged to the kth class
        self.labels = None

    def _compute_z(self, X: np.ndarray):
        gamma = np.exp(-self.n_centers/250)
        for i in range(X.shape[0]):
            a = [np.linalg.norm(X[i] - k)**2 - gamma*np.log(self.alpha[j])
                 for j, k in enumerate(self.centroids)]
            idx = np.argmin(a)
            self.z[i, :] = 0
            self.z[i, idx] = 1

    def _update_gamma(self):
        self.gamma = np.exp(-self.n_centers/250)

    def _update_alpha(self, X: np.ndarray, gamma: float):
        entropy = np.sum(self.alpha * np.log(self.alpha))
        new_alpha = np.zeros_like(self.alpha)
        for kth, alpha in enumerate(self.alpha):
            new_alpha[kth] = np.sum(self.z[:, kth])/X.shape[0] +\
                (self.beta/gamma)*alpha*(np.log(alpha)-entropy)
        self.alpha = new_alpha

    def _update_beta(self, X: np.ndarray, alpha_t: np.ndarray):
        eta = min(1, 1/self.t**(np.floor(X.shape[1]/2 - 1)))
        first_term = np.sum(
            np.exp(-eta*X.shape[0]*np.abs(self.alpha-alpha_t)))/self.n_centers

        sum_ln_alpha_t = np.sum(np.log(alpha_t))
        second_term = (1-np.max(np.sum(self.z, axis=0) /
                       X.shape[0]))/(-np.max(alpha_t*sum_ln_alpha_t))

        self.beta = min(first_term, second_term)

    def _update_c_alpha_z(self, X: np.ndarray) -> np.ndarray:
        self.n_centers -= np.sum(self.alpha <= 1/X.shape[0])
        # idx = ~((self.alpha < 1/X.shape[0]) | (self.alpha <= 0))
        idx = ~((self.alpha <= 1/X.shape[0]))
        self.alpha = self.alpha[idx]
        assert self.alpha.shape[0] == self.n_centers, 'alpha.shape[0] != n_centers'
        self.alpha /= np.sum(self.alpha)

        self.z = self.z[:, idx]
        assert self.z.shape[1] == self.n_centers, 'z.shape[1] != n_centers'
        with np.errstate(divide='ignore', invalid='ignore'):
            self.z /= np.sum(self.z, axis=0)
            np.nan_to_num(self.z, nan=0, copy=False)
        return idx

    def _update_a(self, X: np.ndarray):
        self.centroids = np.zeros((self.n_centers, X.shape[1]))
        for kth in range(self.n_centers):
            with np.errstate(divide='ignore', invalid='ignore'):
                self.centroids[kth] = np.sum(
                    self.z[:, kth:kth+1]*X, axis=0)/np.sum(self.z[:, kth])
        # np.nan_to_num(self.centroids, nan=np.sum(np.mean(X,axis=0)), copy=False)
        np.nan_to_num(self.centroids, nan=0, copy=False)
        # self.centroids /= np.sum(self.z, axis=0).reshape(-1,1)

    def fit(self, X: np.ndarray):
        """
        fit the model to the data
        """
        self.centroids = X.copy()
        self.n_centers = self.centroids.shape[0]
        self.alpha = np.array([1 / self.n_centers] * self.n_centers)
        gamma = np.exp(-self.n_centers/250)
        self.z = np.zeros((X.shape[0], self.n_centers))
        for i in range(X.shape[0]):
            a = [np.linalg.norm(X[i] - k)**2 - gamma*np.log(self.alpha[j])
                 for j, k in enumerate(self.centroids)]
            a[i] = np.inf
            idx = np.argmin(a)
            self.z[i, :] = 0
            self.z[i, idx] = 1
        self._update_alpha(X, gamma)
        self._update_c_alpha_z(X)
        self.record.append({
            'centroids': self.centroids,
            'alpha': self.alpha,
        })
        self._update_a(X)
        self.t = 1

        while True:
            alpha_t = self.alpha.copy()
            self._compute_z(X)
            self._update_gamma()
            self._update_alpha(X, self.gamma)
            self._update_beta(X, alpha_t)
            idx = self._update_c_alpha_z(X)
            # if len(self.alpha == 1):
            #     break

            if self.t >= 60:
                self.beta = 0
            a_t = self.centroids[idx, :].copy()
            self._update_a(X)
            self.record.append({
                'centroids': self.centroids,
                'alpha': self.alpha,
            })
            if np.max(np.linalg.norm(self.centroids - a_t, axis=1)) < self.epsilon:
                break
            self.t += 1

    def predict(self, X: np.ndarray):
        """
        predict the labels of the data
        """
        labels = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            idx = np.argmin(np.linalg.norm(X[i] - self.centroids, axis=1))
            labels[i] = idx
        return labels

    def accuracy_rate(self, X: np.ndarray, y_true: np.ndarray):
        if np.unique(y_true).shape[0] != self.n_centers:
            return 0
        y_true = y_true.astype(int)

        custom_c = np.zeros_like(self.centroids)
        for i in range(self.n_centers):
            idx = np.argmin(np.linalg.norm(X - self.centroids[i], axis=1))
            custom_c[y_true[idx]] = self.centroids[i]
        self.centroids = custom_c

        return np.mean(y_true == self.predict(X))
