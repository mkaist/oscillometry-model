import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.signal import hilbert 

FS = 500
t = np.arange(0,40,1/FS)
f = 1 #HR [bps]
w = 2*np.pi*f


def set_parameters(r=3, a=0.11, b=0.0453, DBP=80, PP=40, Va0=0.3, Vc0=300, Pc0=150):
    prms = {'r':r, 'a':a, 'b':b, 'DBP':DBP, 'PP':PP, 'Va0':Va0, 'Vc0':Vc0, 
            'Pc0':Pc0, 'Pext':Pc0 - r*t}
    return prms
        
def generate_waveform(prms):
    DBP, PP = prms['DBP'], prms['PP']   
    P_artery = DBP + 0.5*PP + \
               0.36*PP*(np.sin(w*t) +\
               0.5*np.sin(2*w*t) +\
               0.25*np.sin(3*w*t))
    
    dP_artery = 0.36*PP*w*(np.cos(w*t) +\
                np.cos(2*w*t) +\
                3/4*np.cos(3*w*t)) #w added as mult, derivative wrong in the paper
    return P_artery, dP_artery

def cuff_model(prms, Pa, dPa):
    r,a,b,Va0,Vc0,Pc0, Pext = prms['r'], prms['a'], prms['b'], prms['Va0'],\
                              prms['Vc0'], prms['Pc0'], prms['Pext']
    Pt = Pa - Pext    
    x = lambda stiffness : a*Va0*np.exp(stiffness*Pt)*(dPa + r)

    dVa = np.where(Pt < 0, x(a), x(-b))    
    Pcuff = integrate.cumtrapz(-r  + ((Pext+760))/Vc0*dVa, t, initial=0) + Pc0  
    return Pcuff

def osc_filter(arr):
    arr = arr - np.convolve(np.ones(FS), arr, 'same')/FS
    arr[:FS], arr[-FS:] = np.ones(FS)*arr[FS], np.ones(FS)*arr[-FS]
    return arr

def peak_finder(y, fs=FS):    
      
    L = int(np.ceil(fs*2)) 
    n = len(y)
    M = np.zeros([L, n], dtype=bool) 
  
    for k in range(1, L):    
        M[k - 1, k:n - k] = (
                (y[k:n-k] > y[0:n-2*k]) &\
                (y[k:n-k] > y[2*k:n])
        )    
    #Find scale with most maxima
    G = np.sum(M, axis=1)
    l_scale = np.argmax(G)

    # find peaks that persist on all scales up to l
    locs_logical = np.min(M[0:l_scale, :], axis=0)
    return np.flatnonzero(locs_logical)
  
def get_envelope(Pext, Pcuff_f):      
    osc_hilb = np.abs(hilbert(Pcuff_f))
    locs_hilb = peak_finder(osc_hilb)
    ex, ey = Pext[locs_hilb], osc_hilb[locs_hilb]   
    return ex, ey 
  

if __name__ == '__main__':
    
    plt.close('all')      
    
    #run example
    prms = set_parameters()
    Pa, dPa = generate_waveform(prms)
    Pcuff = cuff_model(prms, Pa, dPa)
    Pcuff_f = osc_filter(Pcuff)      
    ex, ey = get_envelope(prms['Pext'], Pcuff_f)
    
    #ocillometry
    plt.figure()
    ax1 = plt.subplot(211)
    plt.plot(t, Pa, 'r')
    plt.plot(t, Pcuff, 'b')
    plt.ylabel('P [mmHg]')
    plt.setp(ax1.get_xticklabels(), visible=False)        
    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(t, Pcuff_f, 'b')
    plt.xlabel('Time (s)')
    plt.ylabel('P [mmHg]')
    plt.savefig('oscillogram')   
        
    #envelope
    plt.figure()
    plt.plot(ex,ey)
    plt.xlabel('P_ext [mmHg]')
    plt.ylabel('P [mmHg]')
    plt.savefig('envelope')   
    



