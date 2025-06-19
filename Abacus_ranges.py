import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as N
import pandas as pd
from scipy.linalg import svd
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
import json
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
seed=12345
import emcee
import sys
from IPython.display import display, Math
import math
import random
from schwimmbad import MPIPool
import os
from scipy import interpolate
from cosmoprimo import *
from cosmoprimo.fiducial import Planck2018FullFlatLCDM, AbacusSummit, DESI, TabulatedDESI

###############

#global/homes/s/sadiram/KP5/NN_MCMC/NN_evaluation/z0.5/v1_mono_quad
weights_mono = json.load(open('/global/cfs/projectdirs/desi/users/sadiram/KP5_proyect/Abacus_LRG/FullModeling/Emulator/z0.8ns/Weigth_mono_sinh_new_z0.8.json'))

W_mono=weights_mono['W']
b_mono=weights_mono['b']
alpha_mono=weights_mono['alpha']
beta_mono=weights_mono['beta']
input_mean_mono=N.array(weights_mono['input_mean'])
input_std_mono=N.array(weights_mono['input_std'])
mono_std_mono=N.array(weights_mono['mono_std'])
param_mean_mono= N.array(weights_mono['param_mean'])  
param_std_mono=N.array(weights_mono['param_std' ])  

def activation__mono(param,alpha_mono,beta_mono):
        return tf.multiply(tf.add(beta_mono, tf.multiply(tf.sigmoid(tf.multiply(alpha_mono, param)), tf.subtract(1.0, beta_mono))), param)


def NN_emulator_mono(param):
     param = (param - param_mean_mono) / param_std_mono
     for ii in range(len(W_mono) - 1):
           param = (param @ W_mono[ii])+ b_mono[ii]
           param = activation__mono(param,alpha_mono[ii],beta_mono[ii])
     param = (param @ W_mono[-1]) + b_mono[-1]
     param=(param* input_std_mono)+input_mean_mono
     param=N.sinh(param)* mono_std_mono
     return param
   
    
weights_quad = json.load(open('/global/cfs/projectdirs/desi/users/sadiram/KP5_proyect/Abacus_LRG/FullModeling/Emulator/z0.8ns/Weigth_quad_sinh_new_z0.8.json'))

W_quad=weights_quad['W']
b_quad=weights_quad['b']
alpha_quad=weights_quad['alpha']
beta_quad=weights_quad['beta']
input_mean_quad=N.array(weights_quad['input_mean'])
input_std_quad=N.array(weights_quad['input_std'])
mono_std_quad=N.array(weights_quad['mono_std'])
param_mean_quad= N.array(weights_quad['param_mean'])  
param_std_quad=N.array(weights_quad['param_std' ]) 


def activation__quad(param,alpha_mono,beta_mono):
        return tf.multiply(tf.add(beta_mono, tf.multiply(tf.sigmoid(tf.multiply(alpha_mono, param)), tf.subtract(1.0, beta_mono))), param)


def NN_emulator_quad(param):
     param = (param - param_mean_quad) / param_std_quad
     for ii in range(len(W_quad) - 1):
           param = (param @ W_quad[ii])+ b_quad[ii]
           param = activation__quad(param,alpha_quad[ii],beta_quad[ii])
     param = (param @ W_quad[-1]) + b_quad[-1]
     param=(param* input_std_quad)+input_mean_quad
     param=N.sinh(param)* mono_std_quad
     return param

######################hexa
#global/homes/s/sadiram/KP5/NN_MCMC/NN_evaluation/z0.5/v1_mono_quad
weights_hexa = json.load(open('/global/cfs/projectdirs/desi/users/sadiram/KP5_proyect/Abacus_LRG/FullModeling/Emulator/z0.8ns/Weigth_hexa_sinh_new_z0.8.json'))

W_hexa=weights_hexa['W']
b_hexa=weights_hexa['b']
alpha_hexa=weights_hexa['alpha']
beta_hexa=weights_hexa['beta']
input_mean_hexa=N.array(weights_hexa['input_mean'])
input_std_hexa=N.array(weights_hexa['input_std'])
mono_std_hexa=N.array(weights_hexa['mono_std'])
param_mean_hexa= N.array(weights_hexa['param_mean'])  
param_std_hexa=N.array(weights_hexa['param_std' ])  

def activation__hexa(param,alpha_mono,beta_mono):
        return tf.multiply(tf.add(beta_mono, tf.multiply(tf.sigmoid(tf.multiply(alpha_mono, param)), tf.subtract(1.0, beta_mono))), param)


def NN_emulator_hexa(param):
     param = (param - param_mean_hexa) / param_std_hexa
     for ii in range(len(W_hexa) - 1):
           param = (param @ W_hexa[ii])+ b_hexa[ii]
           param = activation__hexa(param,alpha_hexa[ii],beta_hexa[ii])
     param = (param @ W_hexa[-1]) + b_hexa[-1]
     param=(param* input_std_hexa)+input_mean_hexa
     param=N.sinh(param)* mono_std_hexa
     return param
    
#######################################################################################################
# Obtener los valores de los argumentos de la línea de comandos
variable1 = float(sys.argv[1])  # El primer argumento es variable1
variable2 = float(sys.argv[2]) 
rescaled = int(sys.argv[3]) 

s_min=variable1; s_max=variable2
step=4
Ns=int(((s_max-s_min)/step)+1)
s_fid=N.linspace(s_min, s_max, Ns)
print(s_fid)

S_data    = N.loadtxt('/global/homes/s/sadiram/KP5/Abacus_xi_box/s.txt', unpack = True) 
Xi0_data  = N.loadtxt('/global/homes/s/sadiram/KP5/Abacus_xi_box/Xi_0.txt', unpack = True)
Xi2_data  = N.loadtxt('/global/homes/s/sadiram/KP5/Abacus_xi_box/Xi_2.txt', unpack = True)
Xi4_data  = N.loadtxt('/global/homes/s/sadiram/KP5/Abacus_xi_box/Xi_4.txt', unpack = True)


def multipoles(s_min, s_max , S_data, Xi0_data, Xi2_data, Xi4_data):

    nr = len(Xi0_data)    #number of realizations
    srange = N.where((s_min <= S_data) & (S_data <= s_max))
    s_av = S_data[srange]
    
    Xi0_av = N.zeros(len(s_av)) 
    for ii in range (0, nr):
        Xi0_av += (Xi0_data[ii][srange])/nr
        
    Xi2_av = N.zeros(len(s_av)) 
    for ii in range (0, nr):
        Xi2_av += (Xi2_data[ii][srange])/nr
        
    Xi4_av = N.zeros(len(s_av)) 
    for ii in range (0, nr):
        Xi4_av += (Xi4_data[ii][srange])/nr
    
    return (s_av, Xi0_av, Xi2_av, Xi4_av) 

s_av, Xi0_av, Xi2_av, Xi4_av = multipoles(s_min, s_max , S_data, Xi0_data, Xi2_data, Xi4_data)

nr_sub=len(s_av)
data = N.zeros(2 * nr_sub)
for ii in range(0, nr_sub):
    data[ii] = Xi0_av[ii]
    data[ii + nr_sub] = Xi2_av[ii]


s_cov = N.loadtxt('/global/homes/s/sadiram/KP5/Cov_Abacus_xi_box/s.txt', usecols=(0), unpack=True)
Pk0_cov = N.loadtxt('/global/homes/s/sadiram/KP5/Cov_Abacus_xi_box/Xi_0.txt', unpack=True)
Pk2_cov = N.loadtxt('/global/homes/s/sadiram/KP5/Cov_Abacus_xi_box/Xi_2.txt', unpack=True)
Pk4_cov = N.loadtxt('/global/homes/s/sadiram/KP5/Cov_Abacus_xi_box/Xi_4.txt', unpack=True)


def covariance(s_min, s_max, s, Xi0, Xi2, Xi4):

    Nm = len(Xi0)
    
    mu_l0 = N.zeros(len(s))
    for ii in range (0, Nm):
        mu_l0 += (Xi0[ii][:])/Nm

    mu_l2 = N.zeros(len(s))
    for ii in range (0, Nm):
        mu_l2 += (Xi2[ii][:])/Nm
        
    mu_l4 = N.zeros(len(s))
    for ii in range (0, Nm):
        mu_l4 += (Xi4[ii][:])/Nm
        
    srange = N.where((s_min <= s) & (s <= s_max))
    srange = srange[0]
    sr = s[srange]
    dim = len(sr)
    #print(dim)
    
    Xi0 = Xi0[:,srange];  Xi2 = Xi2[:,srange];  Xi4 = Xi4[:,srange];
    mu_l0 = mu_l0[srange]; mu_l2 = mu_l2[srange];  mu_l4 = mu_l4[srange];
    
    cov = N.zeros((3*dim,3*dim))
    for i in range(0, dim):
        for j in range(0, dim):
            #differences
            diffi0 = Xi0[:, i] - mu_l0[i]
            diffj0 = Xi0[:, j] - mu_l0[j]
            diffi2 = Xi2[:, i] - mu_l2[i]
            diffj2 = Xi2[:, j] - mu_l2[j]
            diffi4 = Xi4[:, i] - mu_l4[i]
            diffj4 = Xi4[:, j] - mu_l4[j]
            #diagonals terms
            cov[i,j] = sum(diffi0*diffj0)
            cov[i+dim,j+dim] = sum(diffi2*diffj2)
            cov[i+2*dim,j+2*dim] = sum(diffi4*diffj4)
            #off diagonal terms
            cov[i,j+dim] = sum(diffi0*diffj2)
            cov[i,j+2*dim] = sum(diffi0*diffj4)
            cov[i+dim,j+2*dim] = sum(diffi2*diffj4)
            #symmetry
            cov[j+dim,i] = cov[i,j+dim]
            cov[j+2*dim,i+dim] = cov[i+dim,j+2*dim]
            cov[j+2*dim,i] = cov[i,j+2*dim]
    cov = cov/(Nm-1.0)
    
    return cov

#rescaled=25
cov_arr= covariance( s_min, s_max , s_cov, Pk0_cov, Pk2_cov, Pk4_cov)/rescaled

sise=int((2*len(cov_arr[:,0])/3))
cov_arr_no_hexa=N.zeros((sise,sise))

for ii in range (sise):
    for jj in range (sise):
        cov_arr_no_hexa[ii,jj]= cov_arr[ii,jj]
        
cov_inv = N.linalg.inv(cov_arr_no_hexa)

################## priors ####################
    
omch2_min = 0.08 # lower range of prior
omch2_max = 0.16  # upper range of prior                                                                                                                                       

ombh2_min = 0.019 # lower range of prior
ombh2_max = 0.024  # upper range of prior

sigmaeft_min = -20. # lower range of prior
sigmaeft_max = 100.  # upper range of prior

h_min = 0.55 # lower range of prior                         
h_max = 0.91  # upper range of prior                                                                                                                                               
        
logAs_min = 2 # lower range of prior             
logAs_max = 4  # upper range of prior 


ns_min = 0.5 # lower range of prior             
ns_max = 1.5  # upper range of prior 


b1_min = 0. # lower range of prior                                                                                                         
b1_max = 2.  # upper range of prior                                                                                                                                       
b2_min = -5. # lower range of prior                                                                                                       
b2_max = 10.  # upper range of prior 

c1_min=-100       ; c1_max=100
bs_min=-5        ; bs_max=5


param_names=['$\omega_{cdm}$',   '$\omega_{b}$',     'h',  'ln 10¹⁰ As', 'ns','$\sigma^2_{eff}$', 'b1', 'b2', 'c1', 'bs']
par_min= omch2_min, ombh2_min, h_min, logAs_min, ns_min, sigmaeft_min, b1_min, b2_min, c1_min, bs_min
par_max= omch2_max, ombh2_max, h_max, logAs_max, ns_max, sigmaeft_max, b1_max, b2_max, c1_max, bs_max



###############3range emulator###################
smin_em=22; smax_em=142 ; Ns_em=31
rbin_emulator=N.linspace(smin_em, smax_em, Ns_em)

inrange_em=N.where((rbin_emulator>=s_min)&(rbin_emulator<=s_max))

z=0.8

cosmo_fid = AbacusSummit('000')
DH_fid =1/cosmo_fid.efunc(z) #km/s/Mpc need to multiply by c but it cancels when dividing by the prime
DA_fid =cosmo_fid.comoving_angular_distance(z) #Mpc

def spllin(x_sp, y, x):
    spform = interpolate.splrep(x,y,s=0)
    y_out = interpolate.splev(x_sp,spform,der=0)
    return y_out


def loglike(params):
    
    omch2=params[0]
    ombh2=params[1]
    h=params[2]
    log_As=(params[3])       #math.log(10**10* As)=params[2]
    As=((math. e)**(log_As))/(10**(10))
    ns=params[4]
    
    sigma2eft=params[5]
    b1 =params[6]
    b2 =params[7]
    c1eft=params[8]
    bs=params[9]

    
    if par_min[0]<=omch2<=par_max[0] and par_min[1]<=ombh2<par_max[1] and par_min[2]<=h<=par_max[2] and  par_min[3]<=log_As<=par_max[3]  and par_min[4]<=ns <=par_max[4]   and par_min[5]<=sigma2eft<=par_max[5] and par_min[6]<=b1<=par_max[6]and par_min[7]<=b2<=par_max[7]and par_min[8]<=c1eft<=par_max[8] and par_min[9]<=bs<=par_max[9]:
       
        
        mono_em= NN_emulator_mono(params)[0][inrange_em]
    
        quad_em= NN_emulator_quad(params)[0][inrange_em]
        
        hexa_em= NN_emulator_hexa(params)[0][inrange_em]
        
        z=0.8
        
        cosmo = AbacusSummit('000').clone(A_s=As,h=h,omega_b=ombh2, omega_cdm=omch2, n_s=ns)
        DH=1/cosmo.efunc(z) #km/s/Mpc need to multiply by c but it cancels when dividing by the prime
        DA=cosmo.comoving_angular_distance(z)#Mpc
        
        
        q_perp=DA/DA_fid
        q_parall=DH/DH_fid
        
        
        alpha_parralel=q_parall
        alpha_perp=q_perp
        ### Begin AP test#################################################################3
        alpha=((alpha_parralel)**(1/3))*((alpha_perp)**(2/3))
        epsilon=((alpha_parralel/alpha_perp)**(1/3))-1
        #########
        opep=1+epsilon
        ### defining Fid cooordinate
        x_w, weight = N.loadtxt("/global/homes/s/sadiram/KP5/gauss.txt", unpack=True)
        nmuo2 = int(N.size(x_w)/2)
        mu_fid2 = x_w[0:nmuo2]**2.
    
        weighto2 = weight[0:nmuo2]**2                                                                                                                             
        r_fid2_ = s_fid**2.
        L2_fid = 0.5*(3.*mu_fid2-1.)
        L4_fid = 0.125*((35.*mu_fid2**2.)-(30*mu_fid2)+3.)
                                                                         

        mu_obs2 = 1./(1. + (1./mu_fid2-1)/opep**6.)
        L2_obs = 0.5*(3.*mu_obs2-1.)
        L4_obs = 0.125*(35.*mu_obs2**2. - 30.*mu_obs2 + 3.)
        r_sub=s_fid
        nr=len(r_sub)
        
        r_obs   = N.zeros((nr,nmuo2))
        xi0_obs = N.zeros((nr,nmuo2))
        xi2_obs = N.zeros((nr,nmuo2))
        xi4_obs = N.zeros((nr,nmuo2))
        for i in range(0,nmuo2):
            r_obs[:,i] = r_sub*alpha*N.sqrt(opep**4.*mu_fid2[i] + (1.-mu_fid2[i])/opep**2.)
        r_obs2 = r_obs**2.
                                                                        
        for i in range(0,nmuo2):
            xi0_obs[:,i] = spllin(r_obs[:,i],r_fid2_*mono_em,s_fid) / r_obs2[:,i]
            xi2_obs[:,i] = spllin(r_obs[:,i],r_fid2_*quad_em,s_fid) / r_obs2[:,i] * L2_obs[i]
            xi4_obs[:,i] = spllin(r_obs[:,i],r_fid2_*hexa_em,s_fid) / r_obs2[:,i] * L4_obs[i]


        xi_obs = xi0_obs + xi2_obs + xi4_obs
        xi_obs0 = xi_obs*0.0
        xi_obs2 = xi_obs*0.0
        xi_obs4 = xi_obs*0.0
        for i in range(0,nmuo2):
            xi_obs0[:,i] = xi_obs[:,i]*weight[i]
            xi_obs2[:,i] = xi_obs[:,i]*L2_fid[i]*weight[i]
            xi_obs4[:,i] = xi_obs[:,i]*L4_fid[i]*weight[i]

    
        xi0 = N.sum(xi_obs0,1)
        xi2 = 5.*N.sum(xi_obs2,1)
        xi4 = 9.*N.sum(xi_obs4,1)
        
        nr_sub=len(s_fid)
        model = N.zeros(2*nr_sub)
        for i in range(0,nr_sub): 
            model[i]        = xi0[i]
            model[i+nr_sub] = xi2[i]
     
        
        residual = data - model    
        chi_2 = N.dot(N.dot(residual, cov_inv), residual)

        ###### we define this conditional for the cases when the chi2 is undefined #####
        if math.isnan(chi_2)==True:
            loglike=-N.inf
        else:
            loglike=-0.5*chi_2


        ### We define a Gaussian obh2 prior ### 
        
        lp1 = 0.
        mu_ombh2 = 0.02237 # lower range of prior                                                                                                                                         
        sigma_ombh2 = 0.00037  # upper range of prior                                                                                                                                       
        lp1 -= 0.5*((ombh2 - mu_ombh2)/sigma_ombh2)**2
        
        ### We define a Gaussian c1 prior ### 
        
        lp2 = 0.
        mu_c1eft = 0 # lower range of prior                                                                                                                                         
        sigma_c1eft = 30  # upper range of prior                                                                                                                                       
        lp2 -= 0.5*((c1eft - mu_c1eft)/sigma_c1eft)**2

                                                                                                                                               # return chi_2
    else:

        loglike=-N.inf
        lp1=0
        lp2=0
    
    return loglike+lp1+lp2

#####################################################
# Initialize the sampler

ndim = 10 # Number of parameters/dimensions (e.g. m and c)
nwalkers = ndim*4# Number of walkers to use. It should be at least twice the number of dimensions.
nsteps = 10000#700  
nchains = 1

def rand(start, end):
    return random.random() * (end - start) + start

start = N.zeros((ndim ,nwalkers) )

for jj in range (nwalkers):
    start[:,jj]=(rand(par_min[0], par_max[0]),rand(par_min[1], par_max[1]),rand(par_min[2], par_max[2]),rand(par_min[3], par_max[3]),rand(par_min[4], par_max[4]),rand(par_min[5], par_max[5]),rand(par_min[6], par_max[6]),rand(par_min[7], par_max[7]),rand(par_min[8], par_max[8]),rand(par_min[9], par_max[9]))


path_file= '/global/cfs/projectdirs/desi/users/sadiram/KP5_proyect/Abacus_LRG/FullModeling/Chains/Reescaled_'+str(rescaled)+'/AbacusExtended_ns_0.8_AP_NN_smax'+str(s_max)+'_smin'+str(s_min)+'_'+str(rescaled)+'.h5'

start_sampler=start.T
start.shape
backend = emcee.backends.HDFBackend(path_file)
if os.path.isfile(path_file):
    
    start_sampler=None
else:
    start_sampler=start.T
    
    
#Set up convergence
max_n = nsteps

# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = N.empty(max_n)

# This will be useful to testing convergence
old_tau = N.inf



with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
        
    sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike, backend=backend, 
                                    pool=pool)
    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(start_sampler, iterations=max_n, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue
            
            
        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = N.mean(tau)
        index += 1
        
        # Check convergence
        converged = N.all(tau * 100 < sampler.iteration)
        converged &= N.all(N.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau