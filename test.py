import numpy as np


a = np.arange(0,10)
print(a)
b = np.random.permutation(a)
print(b)

c = np.array_split(b,3)
for i in c:
    print(i)