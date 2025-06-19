#!/usr/bin/env python
# coding: utf-8

# In[1]:


import multiprocessing as mp
import numpy as np

#### Notebook to show how to use the parser
from use_gsm import *

import numpy as np
import math
import pandas as pd 
import matplotlib.pyplot as plt
from IPython.display import clear_output
from cosmoprimo import *
from cosmoprimo.fiducial import Planck2018FullFlatLCDM, AbacusSummit, DESI, TabulatedDESI

import scipy.integrate as integrate
import scipy.special as special
from scipy.interpolate import CubicSpline
import os.path
import random

#### We define the size of the grid  ####   

ocdm_min=0.08    ; ocdm_max=0.16
ob_min=0.019     ; ob_max=0.024
h_min=0.55       ; h_max=0.91
logAs_min=2      ; logAs_max=4


sigma_eff_min=-20  ; sigma_eff_max=100
b1_min=0         ; b1_max=2
b2_min=-5        ; b2_max=10
c1_min=-50       ; c1_max=50
bs_min=-3        ; bs_max=3

param_names=['$\omega_{cdm}$',   '$\omega_{b}$',     'h',  'ln 10¹⁰ As'   , '$\sigma^2_{eff}$', 'b1', 'b2']#, 'c1', 'bs']
param_min= ocdm_min, ob_min, h_min, logAs_min, sigma_eff_min, b1_min, b2_min#, c1_min, bs_min
param_max= ocdm_max, ob_max, h_max, logAs_max, sigma_eff_max, b1_max, b2_max#, c1_max, bs_max


########### range of the Correlation Function ############
smin=22.5; smax=127.5 ; Ns=22
s_fid=np.linspace(smin, smax, Ns)


########## redshift of the Correlation funcion ######     
redshift       = 0.55
#proc           = 12234

def create_korobov_samples(order, dim, base=17797):
    """
    Create Korobov lattice samples.

    Args:
        order (int):
            The order of the Korobov latice. Defines the number of
            samples.
        dim (int):
            The number of dimensions in the output.
        base (int):
            The number based used to calculate the distribution of values.

    Returns (numpy.ndarray):
        Korobov lattice with ``shape == (dim, order)``
    """
    values = np.empty(dim)
    values[0] = 1
    for idx in range(1, dim):
        values[idx] = base * values[idx - 1] % (order + 1)

    grid = np.mgrid[:dim, : order + 1]
    out = values[grid[0]] * (grid[1] + 1) / (order + 1.0) % 1.0
    return out[:, :order]


######## we define the number of points to be used #############
npoints=50000           # number of points for the NN
npoints_test=5000          # number of points for the test
npoints_valset=5000      # number of points for the validation set



######## we save the distributions #############
def grid(points, seed=int(17797), name='parametros.csv'):
  distribution_korobov=np.zeros((len(param_max), points))
  distribution=create_korobov_samples(points, len(param_max), base=seed)
  for ii in range(len(param_max)):
    distribution_korobov[ii,:]=distribution[ii,:]*(param_max[ii]-param_min[ii])+param_min[ii]

  matrix = pd.DataFrame(data=distribution_korobov.T, columns=param_names)
  matrix.to_csv(name, index=False)
  return distribution_korobov.T


korobov_dist= grid(npoints, int(17797), './dataNN/korobov_'+str(npoints)+'.csv')
korobov_test_dist= grid(npoints_test, int(17797), './dataNN/korobov_test_'+str(npoints_test)+'.csv')
korobov_valset_dist= grid(npoints_valset, int(1234), './dataNN/korobov_valset_'+str(npoints_valset)+'.csv')


# In[2]:


print('s_fid='+str(s_fid))


# # We need this function in order to validate the method

# In[3]:


def funct(params):
    #We define the parameter we are varying
    omch2=params[0]
    ombh2=params[1]
    h=params[2]
    log_As=(params[3])
    sigma2eft=params[4]
    b1 =params[5]
    b2 =params[6]
    c1eft=0
    bs=0


    As=((math. e)**(log_As))/(10**(10))
    H0=h*100  
    cosmo_cloned = Cosmology(A_s=As,h=h,omega_b=ombh2, omega_cdm=omch2, n_s=0.96, N_eff=3.046, m_ncdm=0.0 )

    fo1 = Fourier(cosmo_cloned, engine='camb')
    pk1 = fo1.pk_interpolator()
    Om=cosmo_cloned['Omega_m']

    Kmin = 0.00001 ;  Kmax = 20
    k = np.logspace(np.log10(Kmin), np.log10(Kmax), num = 611)
        
    path_gsm='./' 
    np.savetxt(path_gsm+'Input/ps_%d.txt'%proc, np.vstack((k, pk1(k, z=redshift))).T )   
    
            
    res_rsd = run_gsm(proc ,pk_name='ps_%d.txt'%proc , Om=Om,zout=redshift, sigma2eft=sigma2eft,c1eft=c1eft,bs=bs, b1=b1, b2=b2 , smin=smin, smax=smax , Ns=Ns, remove=True )         
    return res_rsd['mono'], res_rsd['quad'],res_rsd['hexa']


# # We define the funtion needed to calculate the grid for korobov_test

# In[4]:


def funct_korobov_test_dist(ii):
    #We define the parameter we are varying
    omch2=korobov_test_dist[ii,0]
    ombh2=korobov_test_dist[ii,1]
    h=korobov_test_dist[ii,2]
    log_As=korobov_test_dist[ii,3]
    sigma2eft=korobov_test_dist[ii,4]
    b1 =korobov_test_dist[ii,5]
    b2 =korobov_test_dist[ii,6]
    c1eft=0
    bs=0


    numero_entero = random.randint(1, 10000000000)
    proc          = int(numero_entero)


    As=((math. e)**(log_As))/(10**(10))
    H0=h*100  
    cosmo_cloned = Cosmology(A_s=As,h=h,omega_b=ombh2, omega_cdm=omch2, n_s=0.96, N_eff=3.046,  m_ncdm=0.0 )

    fo1 = Fourier(cosmo_cloned, engine='camb')
    pk1 = fo1.pk_interpolator()
    Om=cosmo_cloned['Omega_m']

    Kmin = 0.00001 ;  Kmax = 20
    k = np.logspace(np.log10(Kmin), np.log10(Kmax), num = 611)
        
    path_gsm='./' 
    np.savetxt(path_gsm+'Input/ps_%d.txt'%proc, np.vstack((k, pk1(k, z=redshift))).T )              
    res_rsd = run_gsm(proc ,pk_name='ps_%d.txt'%proc , Om=Om,zout=redshift, sigma2eft=sigma2eft,c1eft=c1eft,bs=bs, b1=b1, b2=b2 , smin=smin, smax=smax , Ns=Ns, remove=True )         
    parameters_grid= (korobov_test_dist[ii,0],korobov_test_dist[ii,1],korobov_test_dist[ii,2],korobov_test_dist[ii,3],korobov_test_dist[ii,4],korobov_test_dist[ii,5],korobov_test_dist[ii,6])

    result = np.concatenate((res_rsd['mono'], res_rsd['quad'], res_rsd['hexa']))

    return result



# # We define the funtion needed to calculate the grid for korobov_valset

# In[5]:


def funct_korobov_valset_dist(ii):
    #We define the parameter we are varying
    omch2=korobov_valset_dist[ii,0]
    ombh2=korobov_valset_dist[ii,1]
    h=korobov_valset_dist[ii,2]
    log_As=korobov_valset_dist[ii,3]
    sigma2eft=korobov_valset_dist[ii,4]
    b1 =korobov_valset_dist[ii,5]
    b2 =korobov_valset_dist[ii,6]
    c1eft=0
    bs=0


    numero_entero = random.randint(1, 10000000000)
    proc          = int(numero_entero)

    As=((math. e)**(log_As))/(10**(10))
    H0=h*100  
    cosmo_cloned = Cosmology(A_s=As,h=h,omega_b=ombh2, omega_cdm=omch2, n_s=0.96, N_eff=3.046, m_ncdm=0.0 )

    fo1 = Fourier(cosmo_cloned, engine='camb')
    pk1 = fo1.pk_interpolator()
    Om=cosmo_cloned['Omega_m']

    Kmin = 0.00001 ;  Kmax = 20
    k = np.logspace(np.log10(Kmin), np.log10(Kmax), num = 611)
        
    path_gsm='./' 
    np.savetxt(path_gsm+'Input/ps_%d.txt'%proc, np.vstack((k, pk1(k, z=redshift))).T )              
    res_rsd = run_gsm(proc ,pk_name='ps_%d.txt'%proc , Om=Om,zout=redshift, sigma2eft=sigma2eft,c1eft=c1eft,bs=bs, b1=b1, b2=b2 , smin=smin, smax=smax , Ns=Ns, remove=True )         
    parameters_grid= (korobov_valset_dist[ii,0],korobov_valset_dist[ii,1],korobov_valset_dist[ii,2],korobov_valset_dist[ii,3],korobov_valset_dist[ii,4],korobov_valset_dist[ii,5],korobov_valset_dist[ii,6])

    result = np.concatenate((res_rsd['mono'], res_rsd['quad'], res_rsd['hexa']))

    return result


# # We define the funtion needed to calculate the grid for korobov_training

def funct_korobov_train_dist(ii):
    #We define the parameter we are varying
    omch2=korobov_dist[ii,0]
    ombh2=korobov_dist[ii,1]
    h=korobov_dist[ii,2]
    log_As=korobov_dist[ii,3]
    sigma2eft=korobov_dist[ii,4]
    b1 =korobov_dist[ii,5]
    b2 =korobov_dist[ii,6]
    c1eft=0
    bs=0


    numero_entero = random.randint(1, 10000000000)
    proc          = int(numero_entero)


    As=((math. e)**(log_As))/(10**(10))
    H0=h*100  
    cosmo_cloned = Cosmology(A_s=As,h=h,omega_b=ombh2, omega_cdm=omch2, n_s=0.96, N_eff=3.046, m_ncdm=0.0 )

    fo1 = Fourier(cosmo_cloned, engine='camb')
    pk1 = fo1.pk_interpolator()
    Om=cosmo_cloned['Omega_m']

    Kmin = 0.00001 ;  Kmax = 20
    k = np.logspace(np.log10(Kmin), np.log10(Kmax), num = 611)
        
    path_gsm='./' 
    np.savetxt(path_gsm+'Input/ps_%d.txt'%proc, np.vstack((k, pk1(k, z=redshift))).T )   
    
    
    res_rsd = run_gsm(proc ,pk_name='ps_%d.txt'%proc , Om=Om,zout=redshift, sigma2eft=sigma2eft,c1eft=c1eft,bs=bs, b1=b1, b2=b2 , smin=smin, smax=smax , Ns=Ns, remove=True )      

    
    
    parameters_grid= (korobov_dist[ii,0],korobov_dist[ii,1],korobov_dist[ii,2],korobov_dist[ii,3],korobov_dist[ii,4],korobov_dist[ii,5],korobov_dist[ii,6])

    result = np.concatenate((res_rsd['mono'], res_rsd['quad'], res_rsd['hexa']))

    return result


# # We calculate the functions

# # Korovov test

# In[7]:



m=npoints_test
n=Ns
# Crea una matriz vacía para guardar los resultados
resultados_test = np.zeros((m, (n*3)))

if __name__ == '__main__':
    # Crea una piscina de procesos con el número de núcleos disponibles
    num_procesos = mp.cpu_count()
    pool = mp.Pool(processes=num_procesos)

    # Evalúa la función para cada valor en el arreglo x utilizando la piscina de procesos
    resultados_test[:, :] = np.array(pool.map(funct_korobov_test_dist, range(m))).reshape(m, (n*3))
    # Cierra la piscina de procesos
    pool.close()

    # Espera a que todos los procesos terminen
    pool.join()
    
############ we save the data ###########
monopole_test_header = pd.DataFrame(data=resultados_test[:,:Ns], columns=s_fid)
quadrupole_test_header = pd.DataFrame(data=resultados_test[:,Ns:2*Ns], columns=s_fid)
hexadecapole_test_header = pd.DataFrame(data=resultados_test[:,2*Ns:3*Ns], columns=s_fid)

monopole_test_header.to_csv('./dataNN/monopole_test_z'+str(redshift)+'points'+str(npoints_test)+'.csv', index=False  )
quadrupole_test_header.to_csv('./dataNN/quadrupole_test_z'+str(redshift)+'points'+str(npoints_test)+'.csv', index=False )
hexadecapole_test_header.to_csv('./dataNN/hexadecapole_test_z'+str(redshift)+'points'+str(npoints_test)+'.csv', index=False )


# In[8]:
# In[9]:


m=npoints_valset
n=Ns
# Crea una matriz vacía para guardar los resultados
resultados_valset = np.zeros((m, (n*3)))

if __name__ == '__main__':
    # Crea una piscina de procesos con el número de núcleos disponibles
    num_procesos = mp.cpu_count()
    pool = mp.Pool(processes=num_procesos)

    # Evalúa la función para cada valor en el arreglo x utilizando la piscina de procesos
    resultados_valset[:, :] = np.array(pool.map(funct_korobov_valset_dist, range(m))).reshape(m, (n*3))
    # Cierra la piscina de procesos
    pool.close()

    # Espera a que todos los procesos terminen
    pool.join()
    
############ we save the data ###########


monopole_valset_header = pd.DataFrame(data=resultados_valset[:,:Ns], columns=s_fid)
quadrupole_valset_header = pd.DataFrame(data=resultados_valset[:,Ns:2*Ns], columns=s_fid)
hexadecapole_valset_header = pd.DataFrame(data=resultados_valset[:,2*Ns:3*Ns], columns=s_fid)
    

monopole_valset_header.to_csv('./dataNN/monopole_valset_z'+str(redshift)+'points'+str(npoints_valset)+'.csv', index=False )
quadrupole_valset_header.to_csv('./dataNN/quadrupole_valset_z'+str(redshift)+'points'+str(npoints_valset)+'.csv', index=False)
hexadecapole_valset_header.to_csv('./dataNN/hexadecapole_valset_z'+str(redshift)+'points'+str(npoints_valset)+'.csv', index=False)


# # korobov_training

# In[ ]:


m=npoints
n=Ns
# Crea una matriz vacía para guardar los resultados
resultados_training = np.zeros((m, (n*3)))

if __name__ == '__main__':
    # Crea una piscina de procesos con el número de núcleos disponibles
    num_procesos = mp.cpu_count()
    pool = mp.Pool(processes=num_procesos)

    # Evalúa la función para cada valor en el arreglo x utilizando la piscina de procesos
    resultados_training[:, :] = np.array(pool.map(funct_korobov_train_dist, range(m))).reshape(m, (n*3))
    # Cierra la piscina de procesos
    pool.close()

    # Espera a que todos los procesos terminen
    pool.join()

    
    ###########   we save te data
monopole_header = pd.DataFrame(data=resultados_training[:,:Ns], columns=s_fid)
quadrupole_header = pd.DataFrame(data=resultados_training[:,Ns:2*Ns], columns=s_fid)
hexadecapole_header = pd.DataFrame(data=resultados_training[:,2*Ns:3*Ns], columns=s_fid)


monopole_header.to_csv('./dataNN/monopole_z'+str(redshift)+'points'+str(npoints)+'.csv', index=False)
quadrupole_header.to_csv('./dataNN/quadrupole_z'+str(redshift)+'points'+str(npoints)+'.csv', index=False)
hexadecapole_header.to_csv('./dataNN/hexadecapole_z'+str(redshift)+'points'+str(npoints)+'.csv', index=False)





