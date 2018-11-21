import numpy as np

def Gauss(A):
    n = A.shape[1]
    U = np.copy(A)
    for j in range(n-1):
        for i in range(j+1,n):
            if U[j,j] ==0:
                continue
            b = -U[i,j]/U[j,j]
            U[i,:] = U[i,:] + U[j,:]*b
    return U

A = np.array([[1,2,3],[4,5,6],[7,8,9]])
print Gauss(A)
