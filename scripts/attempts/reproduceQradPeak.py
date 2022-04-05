#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 15:48:58 2021

@author: bfildier
"""

from datetime import datetime as dt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import xarray as xr
import pandas as pd
import pytz 
from datetime import timedelta, timezone
import sys,os,glob
import argparse
import pickle
from matplotlib import cm
from math import log, pi


##-- directories and modules

workdir = os.path.dirname(os.path.realpath(__file__))
repodir = os.path.dirname(workdir)
moduledir = os.path.join(repodir,'functions')
subdirname = 'proxyQradPeakFromMoistureProfile'
resultdir = os.path.join(repodir,'results',subdirname)
radresultdir = os.path.join(repodir,'results','radiative_features')
figdir = os.path.join(repodir,'figures',subdirname)
inputdir = '/Users/bfildier/Dropbox/Data/EUREC4A/sondes_radiative_profiles/'

# Load own module
projectname = 'EUREC4A_organization'
# while os.path.basename(repodir) != projectname:
#     repodir = os.path.dirname(repodir)
thismodule = sys.modules[__name__]

## Own modules

sys.path.insert(0,moduledir)
print("Own modules available:", [os.path.splitext(os.path.basename(x))[0]
                                 for x in glob.glob(os.path.join(moduledir,'*.py'))])

from radiativefeatures import *
# from thermodynamics import *
# from conditionalstats import *
from matrixoperators import *

##--- local functions

def defineSimDirectories():
        
    # create output directory if not there
    os.makedirs(os.path.join(figdir),exist_ok=True)


def planck(nu,temp):
    """Planck function
    
    Input:
      - nu wavenumber in m-1
      - temp in K
    
    Output in J.s-1.sr-1.m-2.cm
    """
    
    
    h = 6.626e-34 # J.s
    kB = 1.381e-23 # J.K-1
    c = 2.998e8 # m.s-1
    
    planck = 2*h*nu**3*c**2/(np.exp(h*c*nu/kB/temp)-1) # J.s-1.sr-1.m-2.cm !
    
    return planck # recode to have vector inputs and ouput matrix


def computeW(qv,pres,dim_z=1):
    """compute integrated water path above each level
    
    Returns:
      - matrix W(z,z_min)"""
    
    W_above = np.full(qv.shape,np.nan)
    Nz = pres.shape[0]

    for i_z in range(Nz-2):
        W_above[:,i_z] = mo.pressureIntegral(arr=qv[:,i_z:],pres=pres[i_z:],p_levmin=pres[i_z],p_levmax=pres[-1],z_axis=dim_z)

    return W_above


if __name__ == "__main__":
    
    
    # paths
    defineSimDirectories()
    
    # operators
    mo = MatrixOperators()
    
    # Profiles
    radprf = xr.open_dataset(os.path.join(inputdir,'rad_profiles_CF.nc'))
    # choose profiles for that day that start at bottom
    data_all = radprf.where(radprf.z_min<=50,drop=True)
    
    
    ## TO DO
    
    import random 
    def generate_colors(n): 
      rgb_values = [] 
      hex_values = [] 
      r = int(random.random() * 256) 
      g = int(random.random() * 256) 
      b = int(random.random() * 256) 
      step = 256 / n 
      for _ in range(n): 
        r += step 
        g += step 
        b += step 
        r = int(r) % 256 
        g = int(g) % 256 
        b = int(b) % 256 
        r_hex = hex(r)[2:] 
        g_hex = hex(g)[2:] 
        b_hex = hex(b)[2:] 
        hex_values.append('#' + r_hex + g_hex + b_hex) 
        rgb_values.append((r,g,b)) 
      return rgb_values, hex_values 
  
    #--> load features for each day
    days =          '20200122','20200124','20200126','20200128','20200131','20200202','20200205','20200207','20200209','20200211','20200213'
    cols,_ = generate_colors(len(days))
    rad_features_all = {}
    
    for day in days:

        #-- Radiative features
        features_filename = 'rad_features.pickle'
        print('loading %s'%features_filename)
        # load
        features_path = os.path.join(radresultdir,day,features_filename)
        f = pickle.load(open(features_path,'rb'))
        # store
        rad_features_all[day] = f 
    
    dim_s = 0
    dim_z = 1
    
    # box of analysis
    lat_box = 11,16
    lon_box = -60,-52
    
    qv = data_all.specific_humidity.values
    pres = data_all.pressure/100 # hPa
    pres_mean = np.nanmean(pres,axis=dim_s) # hPa
    
    ##-- constants
    
    g = 9.81 # m.s-2
    c_p = 1000 # J/K/kg
    m_to_cm = 1e2
    cm_to_um = 1e4
    
    ##-- pick one \nu^\star for all profiles, assuming W^\star = 3mm
    
    print('- compute reference emitting wavenumber:')
    nu_rot = 150 # cm-1
    l_rot = 56 # cm-1
    kappa_rot = 127 # m2.kg-1
    W_star = 3  # mm, or kg.m-2
    kappa_star = 1/W_star # m2.kg-1
    nu_star = nu_rot + l_rot*log(kappa_rot/kappa_star) # cm-1
    print('nu_star =',nu_star,'cm-1')

    nu_star_inv_m = nu_star*m_to_cm # m-1
    
    ##-- \beta coefficient (=dln(tau) / dln(p), Jeevanjee Fueglistaler 2019) in p space

    W_above = {}
    beta = {}
    
    for day in days:
        
        qv_day = data_all.sel(launch_time=day).specific_humidity.values
    
        W_above[day] = computeW(qv_day,pres_mean)
    
        ln_wpz = np.log(W_above[day])
        ln_p = np.log(pres)
        ln_p_mean = np.nanmean(ln_p,axis=0)
    
        beta[day],_ = mo.derivative(ln_wpz,ln_p_mean,axis=dim_z)

    ##-- compute proxy peak height and estimate magnitude
    keep = {}
    beta_peak_all = {}
    p_peak_all = {}
    planck_peak_constant_all = {}
    planck_peak_all = {}
    proxyQrad = {}
    proxyQrad_constant_B = {}
    Qrad = {}

    for day in days:
        
        f_day = rad_features_all[day]
        data_day = data_all.sel(launch_time=day)

        # number of profiles
        N_p = beta[day].shape[0]
        
        # additional filters on sondes
        qrad_peak = f_day.qrad_lw_peak
        keep_large = np.absolute(qrad_peak) > 5 # K/day
        keep_box = np.logical_and(np.logical_and(data_day.longitude[:,0] > lon_box[0],
                                              data_day.longitude[:,0] < lon_box[1]),
                               np.logical_and(data_day.latitude[:,0] > lat_box[0],
                                              data_day.latitude[:,0] < lat_box[1]))
        keep[day] = np.logical_and(keep_large,keep_box)

        # beta term
        beta_peak_all[day] = np.full((N_p,),np.nan)
        for i_p,i_z in zip(range(N_p),f_day.i_lw_peak):
        
            beta_peak_all[day][i_p] = beta[day][i_p,i_z]

        # peak altitude
        p_peak_all[day] = np.full((N_p,),np.nan)
        for i_p,i_z in zip(range(N_p),f_day.i_lw_peak):
        
            p_peak_all[day][i_p] = data_day.pressure.values[i_p,i_z]

        # planck term
        planck_peak_constant_all[day] = np.full((N_p,),np.nan)
        planck_peak_all[day] = np.full((N_p,),np.nan)
        for i_p,i_z in zip(range(N_p),f_day.i_lw_peak):
        
            # constant
            planck_peak_constant_all[day][i_p] = 0.45
            # T and nu dependent
            temp_peak = data_day.temperature.values[i_p,i_z]
            planck_peak_all[day][i_p] = pi*planck(nu_star_inv_m,temp_peak)*1e2


        # proxy Qrad
        proxyQrad[day] = -g/c_p*(beta_peak_all[day]/p_peak_all[day])*planck_peak_all[day]
        proxyQrad_constant_B[day] = -g/c_p*(beta_peak_all[day]/p_peak_all[day])*planck_peak_constant_all[day]
        Qrad[day] = f_day.lw_peaks.qrad_lw_peak.values

#     ##-- linear regression


# print('- using scipy.linregress')

# slope,intercept,rvalue,pvalue,stderr = linregress(xdata,ydata)
# slope_constB,intercept_constB,rvalue_constB,pvalue_constB,stderr_constB = linregress(xdata_constant_B,ydata)


    ##-- show correlation with true Qrad

fig,ax = plt.subplots(figsize=(7,5))

for day,col in zip(days,cols):
    
    # select = keep[day]
    
    delta_nu = 77 # cm-1
    
    xdata = proxyQrad[day]*86400*delta_nu
    xdata_constant_B = proxyQrad_constant_B[day]*86400*delta_nu
    ydata = Qrad[day]
    
    # ax.scatter(xdata,ydata,s=15,c='k',alpha=0.3,label=r'variable $B_{\nu^*}(T^*)$')
    ax.scatter(xdata_constant_B,ydata,s=15,c='r',alpha=0.3,label=r'constant $B_{\nu^*}(T^*)$')
    
ax.plot([-20,0],[-20,0],'k')

ax.set_xlabel(r'$-\frac{g}{c_p}\, \pi B_{\nu}\,\frac{1}{e}$ ')
ax.set_ylabel(r'$Q_{rad,peak}$ (K/day)')
# ax.legend()
    

## Height

fig,ax = plt.subplots(figsize=(7,5))

for day,col in zip(days,cols):
    
    # select = keep[day]
    
    delta_nu = 77 # cm-1
    
    # xdata = [day]*86400*delta_nu
    # xdata_constant_B = proxyQrad_constant_B[day]*86400*delta_nu
    ydata = p_peak_all[day]
    
    # ax.scatter(xdata,ydata,s=15,c='k',alpha=0.3,label=r'variable $B_{\nu^*}(T^*)$')
    ax.scatter(xdata_constant_B,ydata,s=15,c='r',alpha=0.3,label=r'constant $B_{\nu^*}(T^*)$')
    
ax.plot([-20,0],[-20,0],'k')

ax.set_xlabel(r'$-\frac{g}{c_p}\, \pi B_{\nu}\,\frac{1}{e}$ ')
ax.set_ylabel(r'$Q_{rad,peak}$ (K/day)')
# ax.legend()
