import numpy as np

def sigmoide(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
        return 1/(1+np.exp(-x))
        
x=np.array(([0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],))
            
y=np.array(([0],
             [1],
             [1],
             [0]))
             
np.random.seed(1)

sinopse0= 2*np.random.random((3,4))-1
sinopse1= 2*np.random.random((4,1))-1

for j in range (100000):
    
    k0=x
    k1=sigmoide(np.dot(k0, sinopse0)
    k2=sigmoide(np.dot(k1, sinopse1)
    
    k2_erro = y - k2
    
    if (j% 10000 ==0):
        print("Erro", j/10000 ":" , str(np.mean(np.abs(k2_erro)))
    
    k2_delta = k2_erro * sigmoide(k2,deriv=True)
    
    k1_erro=k2_delta.dot(sinopse1.t)
    
    
    k1_delta= k1_erro * sigmoide(k1,deriv=True)
    
    sinopse1 += k1.T.dot(k2_delta)
    
    sinopse0 += k0.T.dot(k1_delta)