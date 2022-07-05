import numpy as np
import sympy as sym
'''Should create a class for all these functions in Controllability and Observability'''
'''Returns the rank of a matrix'''
def check_rank(A):
    return(np.linalg.matrix_rank(A))

'''Returns full or not full for the rank of the matrix'''
def full_rank(A):
    r, c = A.shape
    k = np.linalg.matrix_rank(A)
    if c == k or r ==k:
        return("Full Rank")
    else:
        return("Not Full Rank")
    
'''Returns whether the controllailty and observability has full rank or not, controllable = full rank'''
def kalman(A = None, B = None, C = None):
    T = None
    O = None
    if B is not None:
        #kc = np.zeros((B.shape[0], B.shape[1]))
        kc = B
        for i in range(1,A.shape[1]):
            kc = np.append(kc, np.matmul(pow(A, i), B), axis = 0)
        T = full_rank(kc)
    if C is not None:
        ko = C
        for j in range(1, A.shape[0]):
            ko = np.append(ko, np.matmul(C, pow(A, i)))
        O = full_rank(ko)
        
    #returns controllability and observability
    return(T, O)

'''Returns the rank of each of the hautus matrix for each eigenvalue'''
def hautus(A = None, B = None, C = None):
    a = []
    if B is not None:
        val, vec = np.linalg.eig(A)
        for i in range(len(val)):
            m = np.append(val[i]*np.identity(A.shape[0]) - A, B, axis = 1)
            a.append(full_rank(m))
            del m
    b = []     
    if C is not None:
        for j in range(len(val)):
            m = np.append(val[i]*np.identity(A.shape[0]) - A, C, axis = 0)
            b.append(full_rank(m))
            del m
            
    return(a, b)

'''Must implement Gilbert criterion'''
def gilbert(A = None, B = None, C = None):
    return(0)
'''Calculates the importance measure of each eigenvalue'''
def impor_eig(A, B, C, n, T):
    #implement for n = none and T = none, to find them
    Bp = np.matmul(np.linalg.inv(T), B)
    Cp = np.matmul(C, T)
    val, vec = np.linalg.eig(A)
    importance = []
    eig = []
    for k in range(len(val)):
        m = []
        for i in range(Cp.shape[0]):
            for j in range(Bp.shape[1]):
                m.append(abs((n[0, j]/n[1, i])*(Cp[i, k]*Bp[k, j])/val[k]))
        importance.append(max(m))
        eig.append(val[k])
        del m
    return(eig, importance)