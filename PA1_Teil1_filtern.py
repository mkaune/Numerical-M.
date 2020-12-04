import skimage
import numpy as np
import matplotlib.pyplot as plt

from skimage import data,io
import os

#Bilder:
C=io.imread("C.png")
B1=io.imread("B1.png")
B2=io.imread("B2.png")

def probe():
    median_tupels=[] #tupels mit np.median und -median()
    MW_tupels=[] #tupels mit np.mean und -Mittelwert()
    Matrizen=[]
    
    #Matrix Groessen:
    A=(np.random.randint(1,100, size=(100,2))) #Bsp:[[12, 71],[ 7, 90],[35, 64]]  
    for i in range(100):
        l=A[i] #groesse fuer matrix, Bsp: [12, 71] 
        U = np.random.RandomState(123).uniform(0, 1, l)#matrix shape l, values 0-1
        Matrizen.append(U)
    
    for matrix in Matrizen:
            median_tupels.append([np.median(matrix),-median(matrix)]) 
            MW_tupels.append([np.mean(matrix),-Mittelwert(matrix)]) #Vergleich
            #print(np.median(matrix),median(matrix))
            print(np.mean(matrix),Mittelwert(matrix))
    median_tupels=np.array(median_tupels) #list--> array
    MW_tupels=np.array(MW_tupels) #list-->array
    
    #max_abweichung:
    abweichung_median=np.max(np.abs(np.sum(median_tupels,axis=1))) 
    abweichung_MW=np.max(np.abs(np.sum(MW_tupels,axis=1))) 
    
    return (abweichung_median,abweichung_MW)

def median(A, filt=None):
    if filt==None:
        B=np.sort(A,axis=None)
        n=B.size
        if n%2==0: 
            return 0.5*(B[(n//2)]+B[(n//2)-1])
        else:
            return B[(n-1)//2]
    else:    
        B=np.sort(A,axis=None) #sorted and flattened
        #B=array([0, 1, 2, 3, 4, 5, 6, 7,8, 9])
        dim=np.shape(A)
        gewichte=gewicht(dim,filt=filt)
        s=gewichte.cumsum()
        
        index=np.argwhere(s>=0.5)[0]
        if s[index]==0.5:

            return 0.5*(B[index]+B[index+1])[0]
        else:
            return B[index][0]
        
def Mittelwert(A,filt=None):
    dim=np.shape(A)
    gewichte=gewicht(dim, filt=filt)
    #elementwise multiplication of matrix objects
    return A.flatten()@gewichte.flatten() #multiplication A,Gewichte als Vectors

def gewicht(dim, filt=None, var=3,numpy=True):
    if filt==None:
        #gleichgewicht
        n,m=dim
        return np.full(dim, (1/(n*m))) #matrix mit nur (1/(n*m) eintrage
    
    elif filt=="Gaus_filt":
        M=3*var  #M<=s
        s=(dim[0]-1)//2 #dim=2s+1
        u,w=s-M,s+M+1 #verschiebte Maske
        W=np.zeros(shape=dim) #shape:(2*s+1,2*s+1))
        
        
        """
        for k in range(-M,+M+1):#0,2M+1
            for l in range(-M,M+1): #0,2M+1
                W[k+s,l+s]=(-(k**2+l**2)/(2*var**2)) #anpassung indeces
        """ #aquivalent zu:
        
        if numpy:
            g= lambda k,l: -((k-M)**2+(l-M)**2)/(2*var**2)
            matrix=np.fromfunction(np.vectorize(g),(2*M+1,2*M+1),dtype=int)
            print(matrix)
            W[u:w,u:w]=np.exp(matrix)#Maske
            
        else:
            #pure python
            for k in range(-M,+M+1):#0,2M+1
                for l in range(-M,M+1): #0,2M+1
                    W[k+s,l+s]=(-(k**2+l**2)/(2*var**2))
            
        #verzichte for loops
           
        #normiere=np.sum(W) #=1
       # W=W/normiere
        return W
    
def Mittelwertfilter(B,s,fil=None,numpy=True):
    n,m =np.shape(B) #dimension
    K=repeat(B) #Fortsetzung B
    
    #naive:
    """
    for i in range(0,n):#0, n
        for j in range(0,m):#o,m
            a1=i-s+n
            a2=j-s+m
            a3=i+s+1+n
            a4=j+s+1+m
            
            A[i,j]=Mittelwert(K[a1:a3,a2:a4],fil)

    """ #Aquivalent zu:
    
    g= lambda i, j: Mittelwert(K[i-s+n:i+s+1+n,j-s+m:j+s+1+m],fil)
    matrix=np.fromfunction(np.vectorize(g),(n,m),dtype=int)
    
    return matrix

def Medianfilter(B,s,fil=None):
    n,m =np.shape(B) #dimension
    K=repeat(B) #Fortsetzung B
    
    #naive:
    """
    for i in range(0,n):#0, n
        for j in range(0,m):#o,m
            a1=i-s+n
            a2=j-s+m
            a3=i+s+1+n
            a4=j+s+1+m
            
            A[i-n,j-m]=Mittelwert(K[a1:a3,a2:a4],fil)

    """ #Aquivalent zu:
    
    g= lambda i, j: median(K[i-s+n:i+s+1+n,j-s+m:j+s+1+m],fil)
    matrix=np.fromfunction(np.vectorize(g),(n,m),dtype=int)
    
    return matrix

#fortsetzung A
def repeat(A): 
    K=np.concatenate((A,np.flip(A,0)),axis=0)
    K=np.concatenate(((np.flip(A,0)),K),axis=0)
    B=np.concatenate(((np.flip(K,1)),K),axis=1)
    K=np.concatenate((B,np.flip(K,1)),axis=1)

    return K

#print(repeat(B1).shape)
#print(Mittelwertfilter(B1,10),"Gaus_filt" )  
#print(Medianfilter(B1,10),"Gaus_filt" )  

l=np.array([[ 0,  2,  4,  6,  8],
       [10, 12, 14, 16, 18],
       [20, 22, 24, 26, 28],
       [30, 32, 34, 36, 38],
       [40, 42, 44, 46, 48]])

#print((gewicht((10,10),filt="Gaus_filt",numpy=False)))
Y=(gewicht((50,50),filt="Gaus_filt",numpy=True))
X=
#print(Medianfilter(l,5,"Gaus_filt") )

plt.figure(figsize=(5,5))

plt.hist(Y, bins=60, density=True, alpha=0.6, rwidth=0.5)

plt.xlim(0,0.02)

plt.figure(figsize=(5,5))
plt.plot()

import timeit
from statistics import mean

def np_vs_py():
    time_np=[] 
    time_py=[] 
    dlist=[1000,2000,5000,10000] #matrix size
    #Matrix Groessen:
    #A=(np.random.randint(1,100, size=(100,2))) #Bsp:[[12, 71],[ 7, 90],[35, 64]]  
    for i in dlist:
        l=[i,i] #groesse fuer matrix, Bsp: [50, 50] 
        U = np.random.RandomState(123).uniform(0, 1, l)#matrix shape l, values 0-1
        #Matrizen.append(U)
        delta = mean(timeit.repeat(lambda : gewicht(U,filt="Gausfilt"), number=1, repeat=3))
        time_np.append(delta)
        
        delta = mean(timeit.repeat(lambda : gewicht(U,filt="Gausfilt",numpy=False), number=1, repeat=3))
        time_py.append(delta)

    speed_up = np.divide(time_np,time_py)
    # Plot the results in a graph
    fig = plt.figure(figsize=(5, 3))
    plt.plot(dlist, time_py, '-o', label = 'py')
    plt.plot(dlist, time_np, '-x', label = 'np')
    plt.legend()
    #plt.xscale('log'); plt.yscale('log'); plt.xlabel('d'); plt.ylabel('time'); plt.grid(True)
    
    fig = plt.figure(2) # insert info in second graph
    plt.plot(dlist,speed_up,'-o')
    plt.title('Speed_up')
    plt.xlabel('dimension Matrix')
    plt.ylabel('speed-up')
        
        
        