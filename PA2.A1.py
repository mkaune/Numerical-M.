

#Vektorisierten Laplace-Operator:

import numpy as np
from scipy import sparse
from scipy.sparse import dia_matrix

def Laplace(N,M):
    
    ones_N=np.ones(N)
    ones_M=np.ones(M)
    data_N, data_M=np.array([ones_N,ones_N*-2,ones_N]), np.array([ones_M,ones_M*-2,ones_M])
    form=[-1,0,1]
    
    #Diskretisierte Ableitung:
    D_N=dia_matrix((data_N, form), shape=(N,N)).toarray()
    D_M=dia_matrix((data_M, form), shape=(M,M)).toarray()
    print(D_N)
    
    #Identitat Matrix
    Id_N=sparse.eye(N).toarray()
    Id_M=sparse.eye(M).toarray()
    
    #Laplace formula 
    return sparse.kron(Id_M,D_N).toarray()+sparse.kron(D_M, Id_N).toarray()

print(Laplace(5,7))

def SeamlessCloning(f,g):
    #g rechteckiges Bild in ein grosseres bild f* eingefuegt 
    
    
def Graubilder(f,g:
    