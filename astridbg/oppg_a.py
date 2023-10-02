import numpy as np
from create_design_matrix import create_design_matrix
from sklearn.model_selection import train_test_split

# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

z = FrankeFunction(x, y) + np.random.normal(0, 0.1, np.shape(x))
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)

maxdegree = 1
for degree in range(1, maxdegree+1):
    XY = create_design_matrix(x_train, y_train, degree)
    beta = np.linalg.pinv(XY.T @ XY) @ XY.T @ z_train

y_tilde = XY @ beta
print(np.shape(y_tilde))
x, y = np.meshgrid(x,y)


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Plot the surface.
surf = ax.plot_surface(x, y, XY @ beta, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()