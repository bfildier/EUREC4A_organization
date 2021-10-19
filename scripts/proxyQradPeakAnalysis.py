#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:50:10 2021

@author: bfildier
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import xarray as xr
import pandas as pd
import pytz
from datetime import datetime as dt
from datetime import timedelta, timezone
import sys,os,glob
import argparse
import pickle
from matplotlib import cm
# import matplotlib.image as mpimg
from math import ceil,e,pi
from scipy.constants import Stefan_Boltzmann
from scipy.ndimage.filters import convolve1d

# ## Graphical parameters
# plt.style.use(os.path.join(matplotlib.get_configdir(),'stylelib/presentation.mplstyle'))

##-- directories and modules

workdir = os.path.dirname(os.path.realpath(__file__))
repodir = os.path.dirname(workdir)
moduledir = os.path.join(repodir,'functions')
subdirname = 'radiative_features'
resultdir = os.path.join(repodir,'results',subdirname)
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
from conditionalstats import *
from matrixoperators import *

##--- local functions

def defineSimDirectories(day):
        
    # create output directory if not there
    os.makedirs(os.path.join(figdir,day),exist_ok=True)

def planck(nu,temp):
    """Planck's function
    
    Arguments:
        - nu: wavenumber in m-1
        - temp: temperature in K
        
    Output in J.s-1.sr-1.m-2.m
    """
    
    h = 6.626e-34 # J.s
    kB = 1.381e-23 # J.K-1
    c = 2.998e8 # m.s-1
    
    planck = 2*h*nu**3*c**2/(np.exp(h*c*nu/kB/temp)-1) # J.s-1.sr-1.m-2.cm !
    
    return planck

if __name__ == "__main__":
    
    # arguments
    parser = argparse.ArgumentParser(description="Compute and store features from radiative profile data")
    parser.add_argument('--day', default='20200126',help="day of analysis")
    parser.add_argument('--overwrite',type=bool,nargs='?',default=False)
    
    args = parser.parse_args()
    day = args.day
    
    # day = '20200126'
    date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
    
    # varids
    ref_varid = 'PW'
    cond_varids = 'QRAD','QRADSW','QRADLW','QV'
    
    # paths
    defineSimDirectories(day)
    
    mo = MatrixOperators()
    
    ###--- Load data ---###
    
    # Profiles
    radprf = xr.open_dataset(os.path.join(inputdir,'rad_profiles_CF.nc'))
    # choose profiles for that day that start at bottom
    data_all = radprf.where(radprf.z_min<=50,drop=True)
    data_day = data_all.sel(launch_time=day)
    
    # Radiative features
    features_filename = 'rad_features.pickle'
    print('loading %s'%features_filename)
    features_path = os.path.join(resultdir,day,features_filename)
    f = pickle.load(open(features_path,'rb'))
    
    # Reference distribution
    ref_filename = 'dist_%s.pickle'%(ref_varid)
    print('load reference %s distribution'%ref_varid)
    ref_dist_path = os.path.join(resultdir,day,ref_filename)
    ref_dist = pickle.load(open(ref_dist_path,'rb'))
    # save in current environment
    setattr(thismodule,'dist_%s'%ref_varid,ref_dist)
    
    ref_var_min = ref_dist.bins[0]
    ref_var_max = ref_dist.bins[-1]
    
    # Conditional distributions
    for cond_varid in cond_varids:
        
        cond_filename = 'cdist_%s_on_%s.pickle'%(cond_varid,ref_varid)
        print('loading %s'%cond_filename)
        cond_dist_path = os.path.join(resultdir,day,cond_filename)
        cond_dist = pickle.load(open(cond_dist_path,'rb'))
        # save in current environment
        setattr(thismodule,'cdist_%s_on_%s'%(cond_varid,ref_varid),cond_dist)

    ###--- Compute terms following Jeevanjee Fueglistaler 2019)
    
    pres = data_day.pressure/100 # hPa
    pres_mean = np.nanmean(pres,axis=0)
    z = data_day.alt/1e3 # km

    # beta coefficient (=dln(tau) / dln(p) in p space    
    ln_wpz = np.log(f.wp_z)
    ln_p = np.log(pres)
    ln_p_mean = np.nanmean(ln_p,axis=0)
    beta,_ = mo.derivative(ln_wpz,ln_p_mean,axis=1)
    
    # transmittivity coefficient -beta/p profiles in p space
    dTrans_dp_tau1 = -beta/pres
    
    def estimatedFluxDivergence(W,temp,pres,beta,emissivity=0.9,extinction_coef=0.2):
        
        planck = pi*emissivity*Stefan_Boltzmann*(temp**4)
        tau = extinction_coef*W
        
        return -planck*beta/pres*tau*np.exp(-tau)
    
    #- 1/q^2 dqdp
    
    qv = data_day.specific_humidity.values
    n_smooth = int(f.dz_smooth/np.diff(data_day.alt.values)[0])*3 # 450m
    # smooth humidity profile
    qv_smooth = convolve1d(qv,np.ones(n_smooth)/n_smooth,axis=1,mode='constant')
    # vertical derivative
    dqvdp,_ = mo.derivative(qv_smooth,pres_mean*100,axis=1)
    # 1/q^2 dqdp factor
    inv_qv2_dqvdp = 1/qv**2 * dqvdp
    # remove negative values
    inv_qv2_dqvdp[inv_qv2_dqvdp<=0] = np.nan
    # smooth again
    inv_qv2_dqvdp_smooth = convolve1d(inv_qv2_dqvdp,np.ones(n_smooth)/n_smooth,axis=1,mode='constant')
    



#%% Figures


    ##-- QV vs. p
    
    fig,axs = plt.subplots(ncols=2,nrows=1,figsize=(14,5))

    Ns = data_day.dims['launch_time']
    
    z = data_day.alt/1e3
    
    # colors
    var_col = f.pw
    norm = matplotlib.colors.Normalize(vmin=var_col.min(), vmax=var_col.max())
    cmap = plt.cm.nipy_spectral
    cols = cmap(norm(var_col)) 
    
    #- QV vs p
    
    ax = axs[0]
    
    for i_s in range(Ns):
        ax.plot(qv[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.3)
        
    ax.invert_yaxis()
    # ax.set_xlim((279,301))
    ax.set_xlabel('T (K)')
    ax.set_ylabel(r'$q_v$ (kg/kg)')
    
    #- colorbar
    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                 ax=ax,shrink=0.95,pad=0.09)
    cb.set_label('PW (mm)')
    
    #- QV smoothed vs p
    
    ax = axs[1]
    
    for i_s in range(Ns):
        ax.plot(qv_smooth[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
        
    ax.invert_yaxis()
    # ax.set_xlim((-0.1,6.1))
    # ax.invert_xaxis()
    ax.set_xlabel('smoothed $q_v$ (km)')
    ax.set_ylabel(r'$q_v$ smoothed (mm)')
    
    #- colorbar
    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                 ax=ax,shrink=0.95,pad=0.09)
    cb.set_label('PW (mm)')
    
    plt.savefig(os.path.join(figdir,day,'qv_and_qv_smoothed_profiles_colorPW_%s.pdf'%date.strftime("%Y%m%d")),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,day,'qv_and_qv_smoothed_profiles_colorPW_%s.png'%date.strftime("%Y%m%d")),bbox_inches='tight')
    

#%%

    ##-- WP_above_z vs. z
    
    fig,axs = plt.subplots(ncols=2,nrows=1,figsize=(14,5))

    Ns = data_day.dims['launch_time']
    
    z = data_day.alt/1e3
    
    # colors
    var_col = f.pw
    norm = matplotlib.colors.Normalize(vmin=var_col.min(), vmax=var_col.max())
    cmap = plt.cm.nipy_spectral
    cols = cmap(norm(var_col)) 
    
    #- WP as a function of T
    
    ax = axs[0]
    
    for i_s in range(Ns):
        ax.plot(data_day.temperature[i_s],f.wp_z[i_s],c=cols[i_s],linewidth=0.5,alpha=0.3)
        
    ax.invert_yaxis()
    ax.set_xlim((279,301))
    ax.set_xlabel('T (K)')
    ax.set_ylabel(r'$\int_z^{TOA} q_v dp/g$ (mm)')
    
    #- colorbar
    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                 ax=ax,shrink=0.95,pad=0.09)
    cb.set_label('PW (mm)')
    
    #- WP as a function of z
    
    ax = axs[1]
    
    for i_s in range(Ns):
        ax.plot(z,f.wp_z[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
        
    ax.invert_yaxis()
    ax.set_xlim((-0.1,6.1))
    ax.invert_xaxis()
    ax.set_xlabel('z (km)')
    ax.set_ylabel(r'$\int_z^{TOA} q_v dp/g$ (mm)')
    
    #- colorbar
    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                 ax=ax,shrink=0.95,pad=0.09)
    cb.set_label('PW (mm)')
    
    plt.savefig(os.path.join(figdir,day,'WPz_vs_T_and_z_colorPW_%s.pdf'%date.strftime("%Y%m%d")),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,day,'WPz_vs_T_and_z_colorPW_%s.png'%date.strftime("%Y%m%d")),bbox_inches='tight')
    
    
    ##-- WP_above_z vs. z with a log y-axis
    
    fig,axs = plt.subplots(ncols=2,nrows=1,figsize=(14,5))

    Ns = data_day.dims['launch_time']
    
    z = data_day.alt/1e3
    
    # colors
    var_col = f.pw
    norm = matplotlib.colors.Normalize(vmin=var_col.min(), vmax=var_col.max())
    cmap = plt.cm.nipy_spectral
    cols = cmap(norm(var_col)) 
    
    #- WP as a function of T
    
    ax = axs[0]
    
    for i_s in range(Ns):
        ax.plot(data_day.temperature[i_s],f.wp_z[i_s],c=cols[i_s],linewidth=0.5,alpha=0.3)
        
    ax.set_yscale('log')
    ax.set_ylim((0.5,100))
    ax.invert_yaxis()
    ax.set_xlim((269,301))
    ax.set_xlabel('T (K)')
    ax.set_ylabel(r'$\int_z^{TOA} q_v dp/g$ (mm)')
    
    #- colorbar
    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                 ax=ax,shrink=0.95,pad=0.09)
    cb.set_label('PW (mm)')
    
    #- WP as a function of z
    
    ax = axs[1]
    
    for i_s in range(Ns):
        ax.plot(z,f.wp_z[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
        
    ax.set_yscale('log')
    ax.set_ylim((0.5,100))
    ax.invert_yaxis()
    ax.set_xlim((-0.1,6.1))
    ax.invert_xaxis()
    ax.set_xlabel('z (km)')
    ax.set_ylabel(r'$\int_z^{TOA} q_v dp/g$ (mm)')
    
    #- colorbar
    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                 ax=ax,shrink=0.95,pad=0.09)
    cb.set_label('PW (mm)')
    
    plt.savefig(os.path.join(figdir,day,'WP_above_z_logy_vs_T_and_z_colorPW_%s.pdf'%(date.strftime("%Y%m%d"))),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,day,'WP_above_z_logy_vs_T_and_z_colorPW_%s.png'%date.strftime("%Y%m%d")),bbox_inches='tight')
    
    
    ##-- WP profile in T z space
    
    fig,axs = plt.subplots(ncols=2,nrows=1,figsize=(14,5))

    Ns = data_day.dims['launch_time']
    
    pres = data_day.pressure/100
    temp = data_day.temperature
    
    # colors
    var_col = f.pw
    norm = matplotlib.colors.Normalize(vmin=var_col.min(), vmax=var_col.max())
    cmap = plt.cm.nipy_spectral
    cols = cmap(norm(var_col)) 
    
    #- WP profile in T space
    
    ax = axs[0]
    
    for i_s in range(Ns):
        ax.plot(f.wp_z[i_s],temp[i_s],c=cols[i_s],linewidth=0.5,alpha=0.3)
    
    # ax.set_yscale('log')
    ax.set_xlim((0.5,100))
    ax.set_xlabel(r'$\int_z^{TOA} q_v dp/g$ (mm)')
    ax.set_ylim((269,301))
    ax.invert_yaxis()
    ax.set_ylabel('T (K)')

    
    #- colorbar
    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                 ax=ax,shrink=0.95,pad=0.09)
    cb.set_label('PW (mm)')
    
    #- WP profile in p space (log-log)
    
    ax = axs[1]
    
    for i_s in range(Ns):
        ax.plot(f.wp_z[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim((0.5,100))
    ax.set_ylim((590,1010))
    ax.invert_yaxis()
    # ax.invert_xaxis()
    ax.set_ylabel('p (hPa)')
    ax.set_xlabel(r'$\int_z^{TOA} q_v dp/g$ (mm)')
    
    #- colorbar
    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                 ax=ax,shrink=0.95,pad=0.09)
    cb.set_label('PW (mm)')
    
    plt.savefig(os.path.join(figdir,day,'WPz_profile_in_T_z_space_colorPW_%s.pdf'%(date.strftime("%Y%m%d"))),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,day,'WPz_profile_in_T_z_space_colorPW_%s.png'%date.strftime("%Y%m%d")),bbox_inches='tight')
    
    
    ##-- beta coefficient (=dln(tau) / dln(p), Jeevanjee Fueglistaler 2019) in p space
    
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(7,5))

    Ns = data_day.dims['launch_time']
    
    pres = data_day.pressure/100
    
    # colors
    var_col = f.pw
    norm = matplotlib.colors.Normalize(vmin=var_col.min(), vmax=var_col.max())
    cmap = plt.cm.nipy_spectral
    cols = cmap(norm(var_col))
    
    #- beta coefficient (=dln(tau) / dln(p), Jeevanjee Fueglistaler 2019) and beta/p profiles in p space
    
    ln_wpz = np.log(f.wp_z)
    ln_p = np.log(pres)
    ln_p_mean = np.nanmean(ln_p,axis=0)
    
    beta,_ = mo.derivative(ln_wpz,ln_p_mean,axis=1)
    
    
    for i_s in range(Ns):
        ax.plot(beta[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.3)
        
    # ax.set_yscale('log')
    ax.set_xlim((-0.5,50))
    ax.set_xlabel(r'$\beta$')
    ax.set_ylim((590,1010))
    ax.invert_yaxis()
    ax.set_ylabel('p (hPa)')

    #- colorbar
    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                 ax=ax,shrink=0.95,pad=0.09)
    cb.set_label('PW (mm)')
    
    
    plt.savefig(os.path.join(figdir,day,'beta_profile_in_p_space_colorPW_%s.pdf'%(date.strftime("%Y%m%d"))),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,day,'beta_profile_in_p_space_colorPW_%s.png'%date.strftime("%Y%m%d")),bbox_inches='tight')
    
    
    ##-- Show qv, wp_z, beta, -1/e beta/p, -pi sigma T^4 1/e beta/p, Qrad
    
    fig,axs = plt.subplots(ncols=2,nrows=3,figsize=(11,13))
    
    Ns = data_day.dims['launch_time']
    
    pres = data_day.pressure/100
    
    # colors
    var_col = f.pw
    norm = matplotlib.colors.Normalize(vmin=var_col.min(), vmax=var_col.max())
    cmap = plt.cm.nipy_spectral
    cols = cmap(norm(var_col))
    
    #---- qv
    
    ax = axs[0,0]
    
    for i_s in range(Ns):
        ax.plot(data_day.specific_humidity[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
    
    # ax.set_xlim((0.5,100))
    ax.set_xlabel(r'$q_v$ (kg/kg)')
    
    #---- water path above level
    
    ax = axs[0,1]
    
    for i_s in range(Ns):
        ax.plot(f.wp_z[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
    
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlim((0.5,100))
    ax.set_xlabel(r'$W \equiv \int_z^{TOA} q_v \frac{dp}{g}$ (mm)',labelpad=-1)
    
    #---- beta
    
    ax = axs[1,0]
    
    for i_s in range(Ns):
        ax.plot(beta[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.3)
    
    # ax.set_yscale('log')
    ax.set_xlim((-0.5,50))
    ax.set_xlabel(r'$\beta \equiv \frac{\partial \ln W}{\partial \ln p}$')
    
    #---- -beta/p
    
    ax = axs[1,1]
    
    for i_s in range(Ns):
        ax.plot(dTrans_dp_tau1[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.3)
    
    # ax.set_yscale('log')
    ax.set_xlim((-0.051,0.001))
    ax.set_xlabel(r'$-\frac{\beta}{p}$')
    
    #---- pi sigma T^4 beta/p
    
    ax = axs[2,0]
    
    # compute
    emissivity = 0.6
    extinction_coef = 0.3
    estimated_QRAD = estimatedFluxDivergence(f.wp_z,temp,pres,beta,emissivity,extinction_coef)

    for i_s in range(Ns):
        ax.plot(estimated_QRAD[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.3)

    ax.text(x=0.06,y=0.06,s=r'$\varepsilon = %1.2f$, $\kappa = %1.2f$'%(emissivity,extinction_coef),transform=ax.transAxes)

    # ax.set_yscale('log')
    ax.set_xlim((-15.1,0.6))
    ax.set_xlabel(r'$Q_{rad}^{estimate} \equiv \left(\pi\varepsilon\sigma T^4\right)*\left(-\frac{\beta}{p}*\kappa W e^{-\kappa W}\right)$')
    
    #---- Qrad
    
    ax = axs[2,1]
    
    for i_s in range(Ns):
        ax.plot(data_day.q_rad_lw[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.3)
    
    # ax.set_yscale('log')
    ax.set_xlim((-15.1,0.6))
    ax.set_xlabel(r'$Q_{rad}^{LW}$')
    
    
    for ax in axs.flatten():
    
        #- all y axes
        ax.set_ylim((490,1010))
        ax.invert_yaxis()
        ax.set_ylabel('p (hPa)')
    
        #- all colorbars
        cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                     ax=ax,shrink=0.95,pad=0.09)
        cb.set_label('PW (mm)')
        
        
    plt.savefig(os.path.join(figdir,day,'proxy_QRADLW_profile_all_steps_colorPW_%s.pdf'%(date.strftime("%Y%m%d"))),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,day,'proxy_QRADLW_profile_all_steps_colorPW_%s.png'%date.strftime("%Y%m%d")),bbox_inches='tight')
    
    
    
    ##-- Show T(p) for all daily profiles
    
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(6,5))
    
    Ns = data_day.dims['launch_time']
    
    pres = data_day.pressure/100
    
    # colors
    var_col = f.pw
    norm = matplotlib.colors.Normalize(vmin=var_col.min(), vmax=var_col.max())
    cmap = plt.cm.nipy_spectral
    cols = cmap(norm(var_col))
    
    #---- T
    
    for i_s in range(Ns):
        ax.plot(data_day.temperature[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
    
    ax.set_xscale('log')
    ax.set_xlim((270,302))
    ax.set_xlabel(r'$T$ (K)')
    
    #- all y axes
    ax.set_yscale('log')
    ax.set_ylim((490,1010))
    ax.invert_yaxis()
    ax.set_ylabel('p (hPa)')

    #- all colorbars
    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                 ax=ax,shrink=0.95,pad=0.09)
    cb.set_label('PW (mm)')
        
    plt.savefig(os.path.join(figdir,day,'T_profile_colorPW_%s.pdf'%(date.strftime("%Y%m%d"))),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,day,'T_profile_colorPW_%s.png'%date.strftime("%Y%m%d")),bbox_inches='tight')
    
    
    ##-- Show 1/qv**2 dqvdp for all daily profiles
    
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(6,5))
    
    Ns = data_day.dims['launch_time']
    
    pres = data_day.pressure/100
    
    # colors
    var_col = f.pw
    norm = matplotlib.colors.Normalize(vmin=var_col.min(), vmax=var_col.max())
    cmap = plt.cm.nipy_spectral
    cols = cmap(norm(var_col))
    
    #---- 1/qv**2 dqvdp
    
    for i_s in range(Ns):
        ax.plot(inv_qv2_dqvdp_smooth[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
    
    # kappa/g vertical line
    k_g = extinction_coef/9.81
    ax.axvline(k_g,c='k',linewidth=0.5)
    
    ax.set_xscale('log')
    # ax.set_xlim((270,302))
    # ax.set_xlabel(r'$T$ (K)')
    
    #- all y axes
    # ax.set_yscale('log')
    ax.set_ylim((490,1010))
    ax.invert_yaxis()
    ax.set_ylabel('p (hPa)')

    #- all colorbars
    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                 ax=ax,shrink=0.95,pad=0.09)
    cb.set_label('PW (mm)')
        
    plt.savefig(os.path.join(figdir,day,'inv_qv2_dqvdp_colorPW_%s.pdf'%(date.strftime("%Y%m%d"))),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,day,'inv_qv2_dqvdp_colorPW_%s.png'%date.strftime("%Y%m%d")),bbox_inches='tight')
    

#%% Paper figure - summary figure for analytic derivation

    def updateBounds(ax,x_left,x_right,y_bot,y_top):
        """Save boundaries for legend"""
        x,y,w,h = ax.get_position().bounds
        x_left = np.nanmin(np.array([x,x_left]))
        x_right = np.nanmax(np.array([x+w,x_right]))
        y_bot = np.nanmin(np.array([y,y_bot]))
        y_top = np.nanmax(np.array([y+h,y_top]))
        return x_left,x_right,y_bot,y_top

    x_left = np.nan
    x_right = np.nan
    y_bot = np.nan
    y_top = np.nan
    
    #-- start figure
    
    fig,axs = plt.subplots(ncols=3,nrows=3,figsize=(10,12))
    
    Ns = data_day.dims['launch_time']
    
    pres = data_day.pressure/100
    
    # colors
    var_col = f.pw
    norm = matplotlib.colors.Normalize(vmin=var_col.min(), vmax=var_col.max())
    cmap = plt.cm.nipy_spectral
    cols = cmap(norm(var_col))
    
    #---- T
    ax = axs[0,0]
    
    for i_s in range(Ns):
        ax.plot(data_day.temperature[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
    
    ax.set_xlim((270,302))
    ax.set_xlabel(r'$T$ (K)')
    
    x_left,x_right,y_bot,y_top = updateBounds(ax,x_left,x_right,y_bot,y_top)
    
    #---- qv
    ax = axs[0,1]
    for i_s in range(Ns):
        ax.plot(data_day.specific_humidity[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)

    # ax.set_xlim((270,302))
    ax.set_xlabel(r'$q_v$ (kg/kg)')
    
    x_left,x_right,y_bot,y_top = updateBounds(ax,x_left,x_right,y_bot,y_top)
    
    #---- W
    ax = axs[0,2]
    for i_s in range(Ns):
        ax.plot(f.wp_z[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
    
    ax.set_xscale('log')
    ax.set_xlim((0.5,100))
    ax.set_xlabel(r'$W(p) \equiv \int_z^{TOA} q_v \frac{dp}{g}$ (mm) $\propto \tau$',labelpad=-1)
    
    x_left,x_right,y_bot,y_top = updateBounds(ax,x_left,x_right,y_bot,y_top)
    
    #---- B_nu(T(p))
    ax = axs[1,0]
    
    nu_star = 482.80 # cm-1
    nu_star_m_m1 = nu_star*1e2 # m-1
    B = planck(nu_star_m_m1,data_day.temperature)*1e2 # cm-1
    
    for i_s in range(Ns):
        ax.plot(pi*B[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
    
    ax.set_xlim((0.33,0.47))
    ax.set_xlabel(r'$\pi B_{\tilde{\nu}^\star}(T)$ (W.m$^{-2}$.cm)')
    
    x_left,x_right,y_bot,y_top = updateBounds(ax,x_left,x_right,y_bot,y_top)
    
    #---- phi_nu_star = tau*e(-tau)
    ax = axs[1,1]
    
    W_star = 3 # mm
    kappa_star = 1/W_star # m2/kg, or mm-1
    tau_star = kappa_star*f.wp_z 
    phi = tau_star*np.exp(-tau_star)
    
    for i_s in range(Ns):
        ax.plot(phi[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
    
    # ax.set_yscale('log')
    # ax.set_xlim((-0.051,0.001))
    ax.set_xlabel(r'$\tau^\star e^{-\tau^\star}$')
    
    x_left,x_right,y_bot,y_top = updateBounds(ax,x_left,x_right,y_bot,y_top)
    
    #---- integral
    ax = axs[1,2]

    #-- first, compute profiles    
    nu_array_inv_m = np.linspace(40000,70000) # m-1
    dnu_array = np.diff(nu_array_inv_m) # m-1
    N_nu = len(nu_array_inv_m)
    N_s = B.shape[0]
    N_z = B.shape[1]
    # cm_to_um = 1e4
    
    # stores int_nu(pi*B*phi) in dimensions (N_s,N_z)
    integral = np.full((N_s,N_z),np.nan)
    
    for i_s in range(Ns):
        
        # stores pi*B*phi*d_nu in (N_nu,N_s) at i_s, before integration
        integrand_nu = np.full((N_nu-1,N_z),np.nan)
        
        for i_nu in range(N_nu-1):
            
            nu_inv_m = nu_array_inv_m[i_nu]
            # Planck
            B_nu = planck(nu_inv_m,data_day.temperature[i_s]) # W.sr-1.m-2.m
            # phi
            W_s = f.wp_z[i_s] # mm
            k_rot = 127 # m2/kg
            nu_rot = 150 # cm-1
            l_rot = 56 # cm-1
            kappa_nu = k_rot*np.exp(-(nu_inv_m/1e2-nu_rot)/l_rot) # m2/kg, or mm-1
            tau_nu = kappa_nu * W_s
            phi_nu = tau_nu * np.exp(-tau_nu)
            # product
            integrand_nu[i_nu] = pi*B_nu*phi_nu*dnu_array[i_nu]
            # print(integrand_nu[i_nu])
        
        integral[i_s] = np.nansum(integrand_nu,axis=0)
    
    x_left,x_right,y_bot,y_top = updateBounds(ax,x_left,x_right,y_bot,y_top)
    
    #-- show
    for i_s in range(Ns):
        ax.plot(integral[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.3)
    
    ax.set_xlabel(r'$\int \pi B_{\nu}(T)\phi(\tau_\nu) d\nu$ (W.m$^2$)')
    
    x_left,x_right,y_bot,y_top = updateBounds(ax,x_left,x_right,y_bot,y_top)
    
    #---- beta/p
    ax = axs[2,0]
    display_factor = 1e4
    
    for i_s in range(Ns):
        ax.plot(display_factor*(beta/pres/100)[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.3)
    
    # ax.set_yscale('log')
    ax.set_xlim((-0.1,5.1))
    ax.set_xlabel(r'$\frac{\beta}{p}$ ($\times 10^{4}$ Pa$^{-1}$)')
    
    x_left,x_right,y_bot,y_top = updateBounds(ax,x_left,x_right,y_bot,y_top)
    
    #---- full H estimate
    ax = axs[2,1]
    
    g = 9.81 # m/s
    c_p = 1000 # J/kg
    day_to_seconds = 86400
    H_est = -g/c_p*(beta/pres/100)*integral*day_to_seconds
    
    for i_s in range(Ns):
        ax.plot(H_est[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.3)
    
    ax.set_xlim((-15.1,0.6))
    ax.set_xlabel(r'$-\frac{g}{c_p} \frac{\beta}{p} \int \pi B_{\nu}(T)\phi(\tau_\nu) d\nu$ (K/day)')
    
    x_left,x_right,y_bot,y_top = updateBounds(ax,x_left,x_right,y_bot,y_top)
    
    #---- actual H
    ax = axs[2,2]
    
    for i_s in range(Ns):
        ax.plot(data_day.q_rad_lw[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.3)
    
    # ax.set_yscale('log')
    ax.set_xlim((-15.1,0.6))
    ax.set_xlabel(r'$Q_{rad}^{LW} (K/day)$')
    
    x_left,x_right,y_bot,y_top = updateBounds(ax,x_left,x_right,y_bot,y_top)
    
    #- all y axes
    for ax in axs.flatten():
        ax.set_ylim((490,1010))
        ax.invert_yaxis()
    for ax in axs[:,0]:
        ax.set_ylabel('p (hPa)')
        
    fs = 14
    #- label hypotheses
    axs[1,0].text(0.5, 0.05, r'H$_1$', horizontalalignment='center',
         verticalalignment='center', transform=axs[1,0].transAxes,fontsize=fs)
    axs[1,1].text(0.5, 0.05, r'H$_2$', horizontalalignment='center',
         verticalalignment='center', transform=axs[1,1].transAxes,fontsize=fs)
    axs[1,2].text(0.5, 0.05, r'H$_1$ & H$_2$', horizontalalignment='right',
         verticalalignment='center', transform=axs[1,2].transAxes,fontsize=fs)
    axs[2,0].text(0.5, 0.05, r'H$_3$', horizontalalignment='center',
         verticalalignment='center', transform=axs[2,0].transAxes,fontsize=fs)
    #- label appproximation and target
    axs[2,1].text(0.5, 0.05, r'(approximation)', horizontalalignment='center',
         verticalalignment='center', transform=axs[2,1].transAxes,fontsize=fs-2)
    axs[2,2].text(0.5, 0.05, r'(target)', horizontalalignment='center',
         verticalalignment='center', transform=axs[2,2].transAxes,fontsize=fs-2)
    
    #- Color bar    
    dy = (y_top-y_bot)/80
    cax = plt.axes([x_left,y_bot-8*dy,x_right-x_left,dy])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax, orientation='horizontal')
    cbar.set_label('PW (mm)',fontsize=fs)
        
    plt.savefig(os.path.join(figdir,day,'paper_method_summary_%s.pdf'%(date.strftime("%Y%m%d"))),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,day,'paper_method_summary_%s.png'%date.strftime("%Y%m%d")),bbox_inches='tight')
    

#%% Paper figure - testing the scaling for all days

    