import numpy as np

def create_design_matrix(x, y, degree):

    d = degree
    n = len(x)
    p = 1 + 2*d + (d-1)**2

    X = np.zeros((n, p))

    col_counter = 0
    X[:,col_counter] = 1
    
    for i in range(1, d+1):
        col_counter += 1
        X[:,col_counter] = x**i
        col_counter += 1 
        X[:,col_counter] = y**i

    for i in range(1, d):
        for j in range(1, d):
            col_counter += 1
            X[:,col_counter] = x**i * y**j   
    
    if col_counter != p-1:
        print ("Error in design matrix construction")

    return X

