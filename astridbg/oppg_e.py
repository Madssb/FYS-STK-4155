import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from create_design_matrix import create_design_matrix


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


np.random.seed(2023)

n = 40
n_bootstraps = 100
maxdegree = 5


# Make data set.
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
z = FrankeFunction(x, y) + np.random.normal(0, 0.1, n)

error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)
z_test = np.expand_dims(z_test, axis=1)

for degree in range(maxdegree):
    X_test = create_design_matrix(x_test, y_test, degree)
    z_pred = np.empty((z_test.shape[0], n_bootstraps))
    for i in range(n_bootstraps):
        x_, y_, z_ = resample(x_train, y_train, z_train)
        X_ = create_design_matrix(x_, y_, degree)
        beta = np.linalg.pinv(X_.T @ X_) @ X_.T @ z_
        z_pred[:, i] = X_test @ beta
    
    
    polydegree[degree] = degree
    error[degree] = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
    bias[degree] = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
    variance[degree] = np.mean( np.var(z_pred, axis=1, keepdims=True) ) 


plt.plot(polydegree, error, label='Total mean squared error')
plt.plot(polydegree, bias, label='Bias')
plt.plot(polydegree, variance, label='Variance')
plt.xlabel("Polynomial degree // complexity")
plt.ylabel("Error")
#plt.yscale("log")
plt.legend()
plt.savefig("figures/bias_variance.png")
plt.show()