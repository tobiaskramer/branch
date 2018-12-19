#http://scikit-image.org/docs/dev/auto_examples/transform/plot_piecewise_affine.html#sphx-glr-auto-examples-transform-plot-piecewise-affine-py

import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import data
from numpy import exp,pi,sqrt,cos,sin
from numpy.fft import *

def randpot1d_four(cf):
    n=cf.shape[0]
    p=(2*np.random.rand(n//2-1)-1)*pi
    phase=np.zeros(n)
    phase[1:n//2]=p
    phase[n//2+1:]=-np.fliplr([p])[0]
    phase[n//2]=0
    cff=cf*exp(1.j*phase)
    cff[0]=0
    
    u=sqrt(n)*ifft(cff)
    
    u=np.real(u)
    s=np.std(u)    
    return u/s

def randpot1d_getFFT(c):
    f=fft(ifftshift(c))
    return complex(1,0)*sqrt(np.abs(f))


def InverseKick(R,cf,K):
    global dp
    # R is a (M,2) array of phase space coordinates
    # cf is fourier transform of the correlation of the kick
    # K is the kick strength
    
    X=np.array(R[:,0])
    dp=K*randpot1d_four(cf)
    Rp=np.array(R).copy()
    ix=np.floor(X).astype('int')
    Rp[:,1]=Rp[:,1]-dp[ix]-(dp[np.mod(ix+1,dp.shape[0])]-dp[ix])*(X-ix)
    return(Rp)
    
def InverseDrift(R,row0,Tau):
    # R is a (M,2) array of phase space coordinates
    # row0 is the row in the image that corresponds to zero momentum
    # Tau os the time of the drift period
    Rp=np.array(R).copy()
    Rp[:,0]=Rp[:,0]+(Rp[:,1]-row0)*Tau
    return(Rp)
 
    
if __name__ == '__main__':    
    
    global dp
    
    maxiteration=10
    
    rimage = data.imread('init_image.png')
    rows, cols = rimage.shape[0], rimage.shape[1]
    
    if rows %2 != 0 :
        image=rimage[1:,:,:]
    if cols %2 != 0 :
        image=image[:,1:,:]    
    rows, cols = image.shape[0], image.shape[1]
    
    # The center row that corresponds to momentum 0
    row0=rows/2.0
  
    #Kicks will be Gaussian correlated with correlation length lc
    lc=0.05
    
    #Kick strength K0 and drift period Tau0
    K0=0.025
    Tau0=0.5
    
    
    
    #some preparations
    Lx=1.0
    Lp=1.0
    

    K=K0/np.float(Lp)*rows
    Tau=np.float(Tau0*Lx*rows)/(cols*Lp)


    #Create a Gaussian correlation 
    x=np.linspace(-np.float(Lx)/2,np.float(Lx)/2,num=rows,endpoint=True)
    c=np.exp(-x**2/lc**2)
    cf=randpot1d_getFFT(c)
    
    
    plt.close('all')

    
    for iteration in range(0, maxiteration):
        str="th"
        if iteration < 2:
            str=["st","nd"][iteration]
            
        print("doing iteration ",iteration+1," out of ",maxiteration)
        
        #Warp image with the kick part of the map, using the inverse mapping
        stage1 = warp(image, InverseKick,map_args={'cf': cf,'K': K},mode='wrap',order=1)
        
        #Warp image with the drift part of the map, using the inverse mapping
        stage2 = warp(stage1, InverseDrift,map_args={'row0': row0,'Tau': Tau},mode='wrap',order=1)
        
        f, (p1, p2, p3) = plt.subplots(1, 3,sharey=True,figsize=(24,8))
        #plt.tight_layout()
        p1.axis('equal')
        p1.axis('off')
        p2.axis('equal')
        p2.axis('off')
        p3.axis('equal')
        p3.axis('off')
        
        p1.imshow(image)
        if iteration==0:
            p1.set_title('initial condition')
        else:
            p1.set_title('after %d%s kick-drift cycle '%(iteration,str))
        p2.imshow(stage1)
        p2.set_title('after kick')
        p2.plot(rows/2+dp,color='c',linewidth=3)
        #p2.plot([0,cols],[rows/2,rows/2],":",linewidth=1)

        p3.imshow(stage2)
        p3.set_title('after %d%s kick-drift cycle '%(iteration+1,str))
        
        # rinse and repeat
        image=stage2
        #f.savefig("iteration-%d.png"%iteration, dpi=160, facecolor='w')
    plt.show()