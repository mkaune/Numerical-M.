#Programmieraufgabe 1: Bilder und Filter
import skimage
import numpy as np
import matplotlib.pyplot as plt

from skimage import data,io
import os

#Bilder übergeben:
B1=io.imread("B1.png")
B2=io.imread("B2.png")
C=io.imread("C.png")

# Aufgabe 1: Erzeugt 100 Zufallsmatrizen unterschiedlicher Größe und vergleicht die Ergebnisse mit den numpy-Funktionen
# mean und median für den gleichgewichteten Fall: Ausgegeben wird die maximale absolute Abweichung 
def Probe():
    median_tupels=[] #tupels mit np.median und -median()
    MW_tupels=[] #tupels mit np.mean und -Mittelwert()
    Matrizen=[]
    #Matrix Groessen:
    A=(np.random.randint(1,100, size=(100,2))) #Bsp:[[12, 71],[ 7, 90],[35, 64]]
    print(A)  
    for i in range(100):
        l=A[i] #groesse fuer matrix, Bsp: [12, 71] 
        U = np.random.RandomState(123).uniform(0, 1, l)#matrix shape l, Werte 0-1
        Matrizen.append(U)
    
    for matrix in Matrizen:
            median_tupels.append([np.median(matrix),-Median(matrix)]) 
            MW_tupels.append([np.mean(matrix),-Mittelwert(matrix)]) #Vergleich
            #print(np.median(matrix),median(matrix))
            #print(np.mean(matrix),Mittelwert(matrix))
    median_tupels=np.array(median_tupels) #list--> array
    MW_tupels=np.array(MW_tupels) #list-->array
    
    #max_abweichung:
    abweichung_median=np.max(np.abs(np.sum(median_tupels,axis=1))) 
    abweichung_MW=np.max(np.abs(np.sum(MW_tupels,axis=1))) 
    string = 'Abweichung des Medians={}'.format(abweichung_median)
    String = 'Abweichung des Mittelwertes={}'.format(abweichung_MW)
    
    return string, String
#Beispieloutput für maximale absolute Abweichung: 0.0 für Median, 1.6653345369377348e-15 für Mittelwert
#Berechnet für beliebige Matrix A mit einer Gewichtung den Median (über alle Matrixelemente)
def Median(A, filt=None, var=None):
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
        gewichte=gewicht(dim,filt=filt, var=var)
        s=gewichte.cumsum()
        
        index=np.argwhere(s>=0.5)[0]
        if s[index]==0.5:

            return 0.5*(B[index]+B[index+1])[0]
        else:
            return B[index][0]
#Berechnet für beliebige Matrix A mit einer Gewichtung (filt) den Mittelwert (über alle Matrixelemente) 
def Mittelwert(A,filt=None, var=None):
    dim=np.shape(A)
    gewichte=gewicht(dim, filt, var)
    #@ Elementweise Multiplikation von Matrizenobjekten
    return A.flatten()@gewichte.flatten() #multiplication A,Gewichte als Vectkoren
#Gewichte
def gewicht(dim, filt=None, var=None):                                                     
    if filt==None:                     # Keine Gewichtung übergeben 
        #gleichgewicht
        n,m=dim                        # Übergibt die Dimensionen von dim = [i, j] an n=i und m=j
        return np.full(dim, (1/(n*m))) # n x m-Matrix mit 1/(n*m) als Eintrag für jedes Matrixelement
    
    elif filt=="Gaus_filt":            # Gewichtung mittels Gaußverteilung
        M=np.floor(var)
        M=3*M #M<=s
        M=int(M)                       #rundet auf wie im ueubungspdf
        s=(dim[0]-1)//2                # dim=2s+1
        u,w=s-M,s+M+1 #verschiebte Maske
        W=np.zeros(shape=dim)          #shape:(2*s+1,2*s+1))
        
        """
        for k in range(-M,+M+1):#0,2M+1
            for l in range(-M,M+1): #0,2M+1
                W[k+s,l+s]=(-(k**2+l**2)/(2*var**2)) #anpassung indeces
        """ #aquivalent zu:
        
        g= lambda k,l: -((k-M)**2+(l-M)**2)/(2*var**2)
        matrix=np.fromfunction(np.vectorize(g),(2*M+1,2*M+1),dtype=int)
        W[u:w,u:w]=np.exp(matrix) #Maske
        
        #verzichte auf for loops
           
        normiere=np.sum(W) #=1
        W=W/normiere
        return W
#2.Aufgabe
def Mittelwertfilter(B,s,filt=None, var=None):  #var abgerundet  *3muss kleiner s sein                                   # B ist die Matrix mit den Grauwerten von 0 bis 255 für jedes Bildpixel
                                                                        # s definiert die Größer der Matrix um den Pixel der gefiltert wird, fil = Filter
    n,m =np.shape(B) #dimension                                         # Übergabe der Bildgröße an n und m                                       
    K=repeat(B) #Fortsetzung B                                          # K ist die Fortsetzunge von B (siehe repeat())
    
    #naive:
    """
    for i in range(0,n):#0, n                                           # Doppelschleife: durchläuft alle Pixel der Matrix B bzw. dessen Fortsezung K
        for j in range(0,m):#o,m                                    
            a1=i-s+n                                                    # Koordinaten für die Eckpunkter der Teil-Matrix B um das Pixel [i,j] in B über dessen Einträge der Wert für B[i,j] gefiltert/ gemittelt wird
            a2=j-s+m
            a3=i+s+1+n
            a4=j+s+1+m
            
            A[i-n,j-m]=Mittelwert(K[a1:a3,a2:a4],fil)

    """ #Aquivalent zu:
    
    g= lambda i, j: Mittelwert(K[i-s+n:i+s+1+n,j-s+m:j+s+1+m],filt, var)     # Mit lambda wird g als math. Funktion mit den Variablen i und j definiert. Die Abbildungsvorschrift ist hier der Mittelwert
                                                                       # Mittelwert über alle Werte von (i-s+n:i+s+1+n)x(j-s+n, j+s+1+m) mit Gewichten = fil
    matrix=np.fromfunction(np.vectorize(g),(n,m),dtype=int)            # Definier Matrix, die die Größe n x m genau wie das Bild hat und Wende die Mitterlwertfunktion
    plt.imshow(matrix, cmap='gray', vmin=0, vmax=255) 
    plt.show()                                                                   # auf jedes Pixel (i,j) an und schreibe die Werte als ganze Zahl in die Matrix
    #return matrix

#recheckiger gleichgewichteter Filter
#Mittelwertfilter(B1,1)
#Mittelwertfilter(B2,1)
#Mittelwertfilter(C,1)
#Gauss Filter mit geeigneten Gewichten
#Mittelwertfilter(B1,1, 'Gaus_filt', 0.333)
#Mittelwertfilter(B2,2, 'Gaus_filt', 0.6)
#Mittelwertfilter(C,1, 'Gaus_filt', 0.33)
#3.Aufgabe
# Funktioniert wie Mittelwertfilter nur mit der Medianfunktion
def Medianfilter(B,s,fil=None, var=None):  #var abgerundet *3 muss kleiner s sein 
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

     #Aquivalent zu:
    """
    g= lambda i, j: Median(K[i-s+n:i+s+1+n,j-s+m:j+s+1+m],fil, var)
    matrix=np.fromfunction(np.vectorize(g),(n,m),dtype=int)
    plt.imshow(matrix, cmap='gray', vmin=0, vmax=255) 
    plt.show()
    #return matrix

#rechteckiger gleichgewichteter Filter
#Medianfilter(B1,1)
#Medianfilter(B2,2)
#Medianfilter(C,1)
#Gauss Filter mit geeigneten Gewichten
#Medianfilter(B1,1, 'Gaus_filt', 0.1)
#Medianfilter(B2,1, 'Gaus_filt', 0.285715286)
#Medianfilter(C,1, 'Gaus_filt', 0.1)
#fortsetzung A
def repeat(A):                                                      # np.flip(A,0) --> dreht die Reihenfolge der Einträge von A um [1,2,3...] --> [...3, 2, 1]
    K=np.concatenate((A,np.flip(A,0)),axis=0)                       # K = concatenate(...) setzt die gedrehte Matrix A an die Matrix A entlang der Zeilen (axis = 0)                                        
    K=np.concatenate(((np.flip(A,0)),K),axis=0)                     # K = setzt die Matrix K an die gedrehte Matrix A = flip(A)-A-flip(A)
                                                                    # np.flip(K,1) --> Dreht die Reihenfolge der Einträge der Einträge von K entlang der Spalten: [a1, a2, ...]-[flip[A]1, flip[A]2,...] --> [..., a2, a1]-[...,flip[A]2, flip[A]1]
    B=np.concatenate(((np.flip(K,1)),K),axis=1)                     # B = concatenate(...) 
    K=np.concatenate((B,np.flip(K,1)),axis=1)                       # K = [[flip(A,0), flip(A,1), flip(A,0)], [flip(A,0 und 1), A, flip(A, 0 und 1)], [flip(A,0), flip(A,1), flip(A,0)]]

    return K                                                        # Ausgabe von K
#Beispielinput:
#A=[[1,2,3],
#[4,5,6],
#[7,8,9]]
#Beispieloutput:
#[[9, 8, 7, 7, 8, 9, 9, 8, 7],
#[6, 5, 4, 4, 5, 6, 6, 5, 4],
#[3, 2, 1, 1, 2, 3, 3, 2, 1],
#[3, 2, 1, 1, 2, 3, 3, 2, 1],
#[6, 5, 4, 4, 5, 6, 6, 5, 4],
#[9, 8, 7, 7, 8, 9, 9, 8, 7],
#[9, 8, 7, 7, 8, 9, 9, 8, 7],
#[6, 5, 4, 4, 5, 6, 6, 5, 4],
#[3, 2, 1, 1, 2, 3, 3, 2, 1]]

# Aufgabe 4 Bilateraler Gaußfilter



def gaussian(x, y = 0, var=0.5):#für Filter
    if y == 0:
        return np.exp(-np.power(x, 2.)/ (2*np.power(var, 2.)))
    else:
        return np.exp(-(np.power(x, 2.)+(np.power(y, 2.))/ (2*np.power(var, 2.))))

#Bilateralfilter

def Filter(B, s , varS, varR): 
    n,m =np.shape(B) 
    K=repeat(B) 

#Definiere F, die Matrize in die die gefilterten Werte geschrieben werden.
#Und die Hilfsmatrizen S für die Werte der Umgebungspixel der zu filternden Pixel (i,j)
#W ist die Hilfsmatritze für die Werte für die GewichtungNormierung für jedes Pixel (i, j)
    U = B
    S = []
    W = []
#Die Doppelschleife mit den Laufindizes i und j durchläuft alle Pixel im Bild
    for i in range(0,n):                                               
        for j in range(0,m): 
#Für jedes Bild-Pixel (i,j) werden die Werte der Pixel in der Umgebung (2s+1)x(2s+1) nach
#Pixelabstand und Grauwertdiffernz über die Gaußfunktion gewichtet und so der neue Grauwert gebildet. Der Wert wird danach in U[i,j] geschrieben
            for l in range(2*s+1):
                row1 = []
                row2 = []
                for k in range(2*s+1):
                    x = i-k
                    y = j-l
                    z = K[(n+i), (m+j)] - K[(n+i-s+k), (m+j-s+l)]
                    w1 = gaussian(x, y, varR)
                    w2 = gaussian(z, y,varS)
                    row1.append(w1*w2*K[n+i-s+k,m+j-s+l]) 
                    row2.append(w1*w2)
                S.append(row1)
                W.append(row2)
#Die gebildeten Werte für die Filterung in S und für die Normierung in W werden aufsummiert
            S = np.sort(S,axis=None)
            W = np.sort(W,axis=None)
            sumW = 0
            sumS = 0
            anz = (2*s+1)*(2*s+1)
            for q in range(anz):
                sumW = sumW + W[q]
                sumS = sumS + S[q]
#Normierung des gefilterten Grauwerts
            if sumW!=0:
                U[i][j] = sumS/sumW
            S=[]#
            W=[]#
#Ausgabe der gefilterten Bildmatrix
    #return U
    plt.imshow(U, cmap='gray', vmin=0, vmax=255) 
    plt.show()
#Filter(B1,1, 10, 50)
#Filter(B2,1, 25, 50)
#Filter(C,1, 3, 75)
#Filter(C,2, 3, 75)
