import numpy as np
import numpy.random as random
from scipy import stats as st
import matplotlib.pyplot as plt

from model import KMeans

if __name__ == "__main__":
    # make data
    n = [200, 150, 150]
    mu_true = np.asanyarray([[0.2, 0.5], [1.2, 0.5], [2.0, 0.5]])
    n_dim = mu_true.shape[1]
    sigma_true = np.asanyarray([[[0.1, 0.085], [0.085, 0.1]],
                                [[0.1, -0.085], [-0.085, 0.1]],
                                [[0.1, 0.085], [0.085, 0.1]]])
    random.seed(50)
    org_data = None

    for i in range(len(mu_true)):
        if org_data is None:
            org_data = np.c_[st.multivariate_normal.rvs(
                mean=mu_true[i], cov=sigma_true[i], size=n[i]),
                             np.ones(n[i]) * i]
        else:
            org_data = np.r_[org_data, np.c_[st.multivariate_normal.rvs(
                mean=mu_true[i], cov=sigma_true[i], size=n[i]),
                                             np.ones(n[i]) * i]]

    # drop true cluster label
    data = org_data[:, 0:2].copy()

    # make model and learn
    model = KMeans(3)
    label = model.fit(data)

    # visualize
    color_dict = {0: 'r', 1: 'g', 2: 'b'}
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(data.shape[0]):
        ax.scatter(data[i, 0], data[i, 1], c=color_dict[label[i]])

    ax.set_title('second scatter plot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    fig.show()
