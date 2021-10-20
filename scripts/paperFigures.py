#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fildier et al. (2021)

@author: bfildier
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import xarray as xr
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta, timezone
import pytz
import sys,os,glob
import argparse
import pickle
from matplotlib import cm
# import matplotlib.image as mpimg
from math import ceil
from scipy.stats import linregress
from scipy.stats import gaussian_kde

##-- directories and modules

workdir = os.path.dirname(os.path.realpath(__file__))
repodir = os.path.dirname(workdir)
moduledir = os.path.join(repodir,'functions')
resultdir = os.path.join(repodir,'results','radiative_features')
figdir = os.path.join(repodir,'figures','paper')
inputdir = '/Users/bfildier/Dropbox/Data/EUREC4A/sondes_radiative_profiles/'
scriptsubdir = 'Fildier2021'

# Load own module
projectname = 'EUREC4A_organization'
thismodule = sys.modules[__name__]

## Own modules
sys.path.insert(0,moduledir)
print("Own modules available:", [os.path.splitext(os.path.basename(x))[0]
                                 for x in glob.glob(os.path.join(moduledir,'*.py'))])

from radiativefeatures import *
from radiativescaling import *
# from thermodynamics import *
from conditionalstats import *
from matrixoperators import *

##--- local functions

def defineSimDirectories():
    """Create specific subdirectories"""
        
    # create output directory if not there
    os.makedirs(os.path.join(figdir),exist_ok=True)


if __name__ == "__main__":
    
    # arguments
    parser = argparse.ArgumentParser(description="Draw paper figures from all precomputed data")
    parser.add_argument('--overwrite',type=bool,nargs='?',default=False)

    # output directory
    defineSimDirectories()
    
    
    #-- day-by-day metadata
    
    days =          '20200122','20200124','20200126','20200128','20200131','20200202','20200205','20200207','20200209','20200211','20200213'
    name_pattern =  'Fish',    'Fish',    'Fish',    'Gravel',  'Fish',    'Flower',  'Gravel',  'Flower',    'Sugar',   'Sugar',   'Fish'
    confidence_pattern = 'High','Medium', 'Medium',     'Low',     'Low',     'High',    'High',    'High',  'Medium',  'Medium',  'High'
    col_pattern = {'':'silver',
                   'Fish':'navy',
                   'Gravel':'orange',
                   'Sugar':'seagreen',
                   'Flower':'firebrick'}
    
    dim_t,dim_z = 0,1
    
    # box of analysis
    lat_box = 11,16
    lon_box = -60,-52
    
    # varids
    ref_varid = 'PW'
    cond_varids = 'QRAD','QRADSW','QRADLW','QV','UNORM','T','P'

    
#%%    ###--- Load data ---###
    
    # Profiles
    radprf = xr.open_dataset(os.path.join(inputdir,'rad_profiles_CF.nc'))
    # choose profiles for that day that start at bottom
    data_all = radprf.where(radprf.z_min<=50,drop=True)
    
    z = data_all.alt.values/1e3 # km
    pres = np.nanmean(data_all.pressure.data,axis=dim_t)/100 # hPa
    
    rad_features_all = {}
    rad_scaling_all = {}
    ref_dist_all = {}
    cond_dist_all = {}
    
    # initalize
    for cond_varid in cond_varids:
         cond_dist_all[cond_varid] = {}
    
    for day in days:

        #-- Radiative features
        features_filename = 'rad_features.pickle'
        print('loading %s'%features_filename)
        # load
        features_path = os.path.join(resultdir,day,features_filename)
        f = pickle.load(open(features_path,'rb'))
        # store
        rad_features_all[day] = f
        
        #-- Radiative scaling
        rad_scaling_filename = 'rad_scaling.pickle'
        print('loading %s'%rad_scaling_filename)
        rad_scaling_path = os.path.join(resultdir,day,rad_scaling_filename)
        rs = pickle.load(open(rad_scaling_path,'rb'))
        # store
        rad_scaling_all[day] = rs
        
        #-- Reference PW distribution
        ref_filename = 'dist_%s.pickle'%(ref_varid)
        print('load reference %s distribution'%ref_varid)
        ref_dist_path = os.path.join(resultdir,day,ref_filename)
        ref_dist = pickle.load(open(ref_dist_path,'rb'))
        # save in current environment
        ref_dist_all[day] = ref_dist
        
        #-- Conditional distributions
        for cond_varid in cond_varids:
            
            # load
            cond_filename = 'cdist_%s_on_%s.pickle'%(cond_varid,ref_varid)
            print('loading %s'%cond_filename)
            cond_dist_path = os.path.join(resultdir,day,cond_filename)
            cond_dist = pickle.load(open(cond_dist_path,'rb'))
            # save in current environment
            cond_dist_all[cond_varid][day] = cond_dist
        
    # Bounds
    ref_var_min = ref_dist_all['20200122'].bins[0]
    ref_var_max = ref_dist_all['20200122'].bins[-1]
    
    
    
#%% (a) summary figure for analytic derivation

    day = '20200126'
    date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
    data_day = data_all.sel(launch_time=day)
    f = rad_features_all[day]

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
    
    pres = data_day.pressure.data/100 # hPa
    
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
        # ax.plot(f.wp_z[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
        ax.plot(rad_scaling_all[day].rad_features.wp_z[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
    
    ax.set_xscale('log')
    ax.set_xlim((0.5,100))
    ax.set_xlabel(r'$W(p) \equiv \int_z^{TOA} q_v \frac{dp}{g}$ (mm) $\propto \tau$',labelpad=-1)
    
    x_left,x_right,y_bot,y_top = updateBounds(ax,x_left,x_right,y_bot,y_top)

    #---- B_nu(T(p))
    ax = axs[1,0]

    nu_star = 482.80 # cm-1
    nu_star_m_m1 = nu_star*1e2 # m-1
    B = rad_scaling_all[day].planck(nu_star_m_m1,data_day.temperature)*1e2 # cm-1
    
    for i_s in range(Ns):
        ax.plot(pi*B[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
    
    ax.set_xlim((0.33,0.47))
    ax.set_xlabel(r'$\pi B_{\tilde{\nu}^\star}(T)$ (W.m$^{-2}$.cm)')
    
    x_left,x_right,y_bot,y_top = updateBounds(ax,x_left,x_right,y_bot,y_top)
    
    #---- phi_nu_star = tau*e(-tau)
    ax = axs[1,1]
    
    W_star = 3 # mm
    # kappa_star = 1/W_star # m2/kg, or mm-1
    kappa_star = 0.3
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
            B_nu = rad_scaling_all[day].planck(nu_inv_m,data_day.temperature[i_s]) # W.sr-1.m-2.m
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
        ax.plot(display_factor*(rad_scaling_all[day].beta/pres/100)[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.3)
    
    # ax.set_yscale('log')
    ax.set_xlim((-0.1,5.1))
    ax.set_xlabel(r'$\frac{\beta}{p}$ ($\times 10^{4}$ Pa$^{-1}$)')
    
    x_left,x_right,y_bot,y_top = updateBounds(ax,x_left,x_right,y_bot,y_top)
    
    #---- full H estimate
    ax = axs[2,1]
    
    g = 9.81 # m/s
    c_p = 1000 # J/kg
    day_to_seconds = 86400
    H_est = -g/c_p*(rad_scaling_all[day].beta/pres/100)*integral*day_to_seconds
    
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
        
    plt.savefig(os.path.join(figdir,'paper_method_summary_%s.pdf'%(date.strftime("%Y%m%d"))),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,'paper_method_summary_%s.png'%date.strftime("%Y%m%d")),bbox_inches='tight')
    


#%% (b) comparing radiative peak height and magnitude with scaling

    fig,axs = plt.subplots(ncols=2,nrows=2,figsize=(10,10))
    
    #-- peak height, using the approximation for the full profile, showing all profiles
    ax = axs[0,0]
    
    for day in days:
    # for day in '20200126',:
        
        # proxy peak heights
        pres_beta_peak = rad_scaling_all[day].rad_features.beta_peaks.pres_beta_peak
        pres_beta_over_p_peak = rad_scaling_all[day].rad_features.beta_over_p_peaks.pres_beta_over_p_peak
        pres_scaling_profile_peak = rad_scaling_all[day].rad_features.scaling_profile_peaks.pres_scaling_profile_peak
        pres_proxy_peak = pres_scaling_profile_peak
        # pres_proxy_peak = pres_beta_peak
        # qrad peak height
        pres_qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak
        
        s = np.absolute(rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak)
        
        # 1:1 line
        ax.plot([910,360],[910,360],'k-.',linewidth=0.5,alpha=0.5)
        # peaks
        ax.scatter(pres_qrad_peak,pres_proxy_peak,s=s,alpha=0.4)

        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xlabel('$Q_{rad}$ peak height (hPa)')
        ax.set_ylabel('Estimated peak height (hPa)')
        ax.set_title('Approximating peak height\nusing full profile approximation')
        
        # ax.set_title(r'Approximating peak height\n as $p^\star = \arg\max_p \left(-\frac{g}{c_p}\frac{\beta}{p}\int B_\nu \phi_\nu\right)$')
    #-- peak magnitude, using the approximation for the full profile, showing all profiles
    ax = axs[0,1]
    
    H_peak_all = {}
    qrad_peak_all = {}
    
    for day in days:
    # for day in '20200126',:
        
        #- approximated peak
        # H_peak = rad_scaling_all[day].scaling_magnitude_lw_peak*1e8
        Ns = rad_scaling_all[day].rad_features.pw.size
        H_peak = np.full(Ns,np.nan)
        for i_s in range(Ns):
            i_z = rad_scaling_all[day].rad_features.scaling_profile_peaks.i_scaling_profile_peak[i_s]
            H_peak[i_s] = rad_scaling_all[day].scaling_profile[i_s,i_z]
        H_peak_all[day] = H_peak
        
        #- true peak
        qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak
        qrad_peak_all[day] = qrad_peak
        
        s = 0.005*np.absolute(rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak)
        
        # plot
        ax.scatter(qrad_peak,H_peak,s=s,alpha=0.5)
        
        ax.set_xlabel('$Q_{rad}$ peak magnitude (K/day)')
        ax.set_ylabel('Estimated peak magnitude (K/day)')
        ax.set_title('Approximating peak magnitude\nusing full profile approximation')
        
    # linear fit
    x = np.hstack([H_peak_all[day] for day in days])
    y = np.hstack([qrad_peak_all[day] for day in days])
    slope, intercept, r, p, se = linregress(x, y)

    xmin,xmax = np.nanmin(x),np.nanmax(x)
    xrange = xmax-xmin
    x_fit = np.linspace(xmin-xrange/20,xmax+xrange/20)
    y_fit = intercept + slope*x_fit
    
    # show
    ax.plot(y_fit,x_fit,'k:')
    # write numbers
    ax.text(0.5,0.1,'$Q_{rad} = 1.45 Q_{rad}^{est} -0.4$\n r=%1.2f'%r,transform=ax.transAxes)
    
    
    #-- peak height, using the simpler approximation for peak heights, showing binned profiles each day
    ax = axs[1,0]
    
    for day in days:
    # for day in '20200126',:
        
        # proxy peak heights
        pres_beta_peak = rad_scaling_all[day].rad_features.beta_peaks.pres_beta_peak
        pres_beta_over_p_peak = rad_scaling_all[day].rad_features.beta_over_p_peaks.pres_beta_over_p_peak
        pres_scaling_profile_peak = rad_scaling_all[day].rad_features.scaling_profile_peaks.pres_scaling_profile_peak
        # pres_proxy_peak = pres_beta_over_p_peak
        pres_proxy_peak = pres_beta_peak
        # qrad peak height
        pres_qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak
        
        s = np.absolute(rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak)
        
        # 1:1 line
        ax.plot([910,360],[910,360],'k-.',linewidth=0.5,alpha=0.5)
        # peaks
        ax.scatter(pres_qrad_peak,pres_proxy_peak,s=s,alpha=0.4)

        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xlabel('$Q_{rad}$ peak height (hPa)')
        ax.set_ylabel('Estimated peak height (hPa)')
        ax.set_title(r'as peak of $\beta$')
    
    #-- peak magnitude, using the simler approximation for peak magnitudes
    ax = axs[1,1]
    
    H_peak_all = {}
    qrad_peak_all = {}
    
    for day in days:
    # for day in '20200126',:
        
        #- approximated peak
        # H_peak = rad_scaling_all[day].scaling_magnitude_lw_peak*1e8
        Ns = rad_scaling_all[day].rad_features.pw.size
        H_peak = np.full(Ns,np.nan)
        for i_s in range(Ns):
            # i_z = rad_scaling_all[day].rad_features.scaling_profile_peaks.i_scaling_profile_peak[i_s]
            H_peak[i_s] = rad_scaling_all[day].scaling_magnitude_lw_peak[i_s]
        H_peak_all[day] = H_peak
        
        #- true peak
        qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak
        qrad_peak_all[day] = qrad_peak
        
        s = 0.005*np.absolute(rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak)
        
        # plot
        ax.scatter(qrad_peak,H_peak,s=s,alpha=0.5)
        
        ax.set_xlabel('$Q_{rad}$ peak magnitude (K/day)')
        ax.set_ylabel('Estimated peak magnitude (K/day)')
        ax.set_title('Using magnitude approximation')
    
    
    plt.savefig(os.path.join(figdir,'paper_approx_peak_with_scalings.pdf'),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,'paper_approx_peak_with_scalings.png'),bbox_inches='tight')
    
    
#%% (b') comparing radiative peak height and magnitude with scaling -- only final scalings

fig,axs = plt.subplots(ncols=2,figsize=(9,4))
plt.subplots_adjust(wspace=0.3)


#-- peak height, using the simpler approximation for peak heights, showing binned profiles each day
ax = axs[0]

for day in days:
# for day in '20200126',:
    
    # proxy peak heights
    pres_beta_peak = rad_scaling_all[day].rad_features.beta_peaks.pres_beta_peak
    pres_beta_over_p_peak = rad_scaling_all[day].rad_features.beta_over_p_peaks.pres_beta_over_p_peak
    pres_scaling_profile_peak = rad_scaling_all[day].rad_features.scaling_profile_peaks.pres_scaling_profile_peak
    # pres_proxy_peak = pres_beta_over_p_peak
    pres_proxy_peak = pres_beta_peak
    # qrad peak height
    pres_qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak
    
    s = np.absolute(rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak)
    
    # 1:1 line
    ax.plot([910,360],[910,360],'k-.',linewidth=0.5,alpha=0.5)
    # peaks
    ax.scatter(pres_qrad_peak,pres_proxy_peak,s=s,alpha=0.4)

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel(r'$Q_{rad}$ peak height (hPa)')
    ax.set_ylabel(r'$\beta$ peak height (hPa)')

#-- peak magnitude, using the simpler approximation for peak magnitudes
ax = axs[1]

scale_factor = 3
H_peak_all = {}
qrad_peak_all = {}

for day in days:
# for day in '20200126',:
    
    #- approximated peak
    # H_peak = rad_scaling_all[day].scaling_magnitude_lw_peak*1e8
    Ns = rad_scaling_all[day].rad_features.pw.size
    H_peak = np.full(Ns,np.nan)
    for i_s in range(Ns):
        # i_z = rad_scaling_all[day].rad_features.scaling_profile_peaks.i_scaling_profile_peak[i_s]
        H_peak[i_s] = rad_scaling_all[day].scaling_magnitude_lw_peak[i_s]
    H_peak_all[day] = H_peak
    
    #- true peak
    qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak
    qrad_peak_all[day] = qrad_peak
    
    s = 0.005*np.absolute(rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak)
    
    # plot
    ax.scatter(qrad_peak,scale_factor*H_peak,s=s,alpha=0.5)
    
ax.set_xlabel(r'$Q_{rad}$ peak magnitude (K/day)')
ax.set_ylabel('Simpler scaled magnitude (K/day)')
ax.set_ylim((-20.2,0.2))


plt.savefig(os.path.join(figdir,'paper_approx_peak_with_final_scalings.pdf'),bbox_inches='tight')
plt.savefig(os.path.join(figdir,'paper_approx_peak_with_final_scalings.png'),bbox_inches='tight')


#%% (b2) comparing radiative peak height and magnitude with full profile scaling -- color by density

    def scatterDensity(ax,x,y,s,alpha):
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        ax.scatter(x,y,c=z,s=s,alpha=0.4)

    fig,axs = plt.subplots(ncols=2,nrows=1,figsize=(10,5))
    
    #-- (a) peak height, using the approximation for the full profile, showing all profiles
    ax = axs[0]
    
    x = []
    y = []
    s = []
    
    for day in days:
    # for day in '20200126',:

        # qrad peak height
        pres_qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak
        x.append(pres_qrad_peak)        
        # proxy peak heights
        pres_beta_peak = rad_scaling_all[day].rad_features.beta_peaks.pres_beta_peak
        pres_beta_over_p_peak = rad_scaling_all[day].rad_features.beta_over_p_peaks.pres_beta_over_p_peak
        pres_scaling_profile_peak = rad_scaling_all[day].rad_features.scaling_profile_peaks.pres_scaling_profile_peak
        pres_proxy_peak = pres_scaling_profile_peak
        # pres_proxy_peak = pres_beta_peak
        y.append(pres_proxy_peak)
        
        s.append(np.absolute(rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak))
        
    # peaks
    x,y,s = np.hstack(x),np.hstack(y),np.hstack(s)
    scatterDensity(ax,x,y,s,alpha=0.4)
    # 1:1 line
    ax.plot([910,360],[910,360],'k',linewidth=0.5,alpha=0.5)

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel('$Q_{rad}$ peak height (hPa)')
    ax.set_ylabel('Estimated peak height (hPa)')
    ax.set_title('Approximating peak height\nusing full profile approximation')
        
        # ax.set_title(r'Approximating peak height\n as $p^\star = \arg\max_p \left(-\frac{g}{c_p}\frac{\beta}{p}\int B_\nu \phi_\nu\right)$')
    #-- (b) peak magnitude, using the approximation for the full profile, showing all profiles
    ax = axs[1]
    
    H_peak_all = {}
    qrad_peak_all = {}
    s = []
    
    for day in days:
    # for day in '20200126',:
        
        #- approximated peak
        # H_peak = rad_scaling_all[day].scaling_magnitude_lw_peak*1e8
        Ns = rad_scaling_all[day].rad_features.pw.size
        H_peak = np.full(Ns,np.nan)
        for i_s in range(Ns):
            i_z = rad_scaling_all[day].rad_features.scaling_profile_peaks.i_scaling_profile_peak[i_s]
            H_peak[i_s] = rad_scaling_all[day].scaling_profile[i_s,i_z]
        H_peak_all[day] = H_peak
        
        #- true peak
        qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak
        qrad_peak_all[day] = qrad_peak
        
        s.append(0.005*np.absolute(rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak))
        
    # plot
    x = np.hstack([qrad_peak_all[day] for day in days])
    y = np.hstack([H_peak_all[day] for day in days])
    s = np.hstack(s)
    scatterDensity(ax,x,y,s,alpha=0.5)
    
    ax.set_xlabel('$Q_{rad}$ peak magnitude (K/day)')
    ax.set_ylabel('Estimated peak magnitude (K/day)')
    ax.set_title('Approximating peak magnitude\nusing full profile approximation')
    
    #- linear fit
    slope, intercept, r, p, se = linregress(y, x)

    ymin,ymax = np.nanmin(y),np.nanmax(y)
    yrange = ymax-ymin
    y_fit = np.linspace(ymin-yrange/20,ymax+yrange/20)
    x_fit = intercept + slope*y_fit
    # show
    ax.plot(x_fit,y_fit,'k:')
    # write numbers
    ax.text(0.5,0.1,'$Q_{rad} = 1.45 Q_{rad}^{est} -0.4$\n r=%1.2f'%r,transform=ax.transAxes)
    
    plt.savefig(os.path.join(figdir,'paper_approx_peak_with_profile_scaling_color_density.pdf'),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,'paper_approx_peak_with_profile_scaling_color_density.png'),bbox_inches='tight')
    
    
#%% (b3) comparing radiative peak magnitude computed with variable and fixed B_star -- color by day

    fig,axs = plt.subplots(ncols=2,nrows=1,figsize=(10,5))
    
    #-- (a) variable B
    
    ax = axs[0]
    
    H_peak_all = {}
    qrad_peak_all = {}
    
    for day in days:
    # for day in '20200126',:
        
        #- approximated peak
        # H_peak = rad_scaling_all[day].scaling_magnitude_lw_peak*1e8
        Ns = rad_scaling_all[day].rad_features.pw.size
        H_peak = np.full(Ns,np.nan)
        for i_s in range(Ns):
            # i_z = rad_scaling_all[day].rad_features.scaling_profile_peaks.i_scaling_profile_peak[i_s]
            H_peak[i_s] = rad_scaling_all[day].scaling_magnitude_lw_peak[i_s]
        H_peak_all[day] = H_peak
        
        #- true peak
        qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak
        qrad_peak_all[day] = qrad_peak
        
        s = 0.005*np.absolute(rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak)
        
        # plot
        ax.scatter(qrad_peak,H_peak,s=s,alpha=0.5)
    
    ax.set_xlabel('$Q_{rad}$ peak magnitude (K/day)')
    ax.set_ylabel('Estimated peak magnitude (K/day)')
    ax.set_title('Using magnitude approximation')
    
    #-- (b) B = 0.004 SI units
    
    ax = axs[1]
    
    H_peak_all = {}
    qrad_peak_all = {}
    
    for day in days:
    # for day in '20200126',:
        
        #- approximated peak
        # H_peak = rad_scaling_all[day].scaling_magnitude_lw_peak*1e8
        Ns = rad_scaling_all[day].rad_features.pw.size
        H_peak = np.full(Ns,np.nan)
        for i_s in range(Ns):
            # i_z = rad_scaling_all[day].rad_features.scaling_profile_peaks.i_scaling_profile_peak[i_s]
            H_peak[i_s] = rad_scaling_all[day].scaling_magnitude_lw_peak_B040[i_s]
        H_peak_all[day] = H_peak
        
        #- true peak
        qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak
        qrad_peak_all[day] = qrad_peak
        
        s = 0.005*np.absolute(rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak)
        
        # plot
        ax.scatter(qrad_peak,H_peak,s=s,alpha=0.5)
    
    ax.set_xlabel('$Q_{rad}$ peak magnitude (K/day)')
    ax.set_ylabel('Estimated peak magnitude (K/day)')
    ax.set_title('Using magnitude approximation')
    
    plt.savefig(os.path.join(figdir,'paper_approx_peak_magnitude.pdf'),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,'paper_approx_peak_magnitude.png'),bbox_inches='tight')
    
#%% (b4) comparing radiative peak magnitude computed with variable and fixed B_star -- color by density

    fig,axs = plt.subplots(ncols=2,nrows=1,figsize=(10,5))
    
    #-- (a) variable B
    
    ax = axs[0]
    
    H_peak_all = {}
    qrad_peak_all = {}
    s = []
    
    for day in days:
    # for day in '20200126',:
        
        #- approximated peak
        # H_peak = rad_scaling_all[day].scaling_magnitude_lw_peak*1e8
        Ns = rad_scaling_all[day].rad_features.pw.size
        H_peak = np.full(Ns,np.nan)
        for i_s in range(Ns):
            # i_z = rad_scaling_all[day].rad_features.scaling_profile_peaks.i_scaling_profile_peak[i_s]
            H_peak[i_s] = rad_scaling_all[day].scaling_magnitude_lw_peak[i_s]
        H_peak_all[day] = H_peak
        
        #- true peak
        qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak
        qrad_peak_all[day] = qrad_peak
        
        s.append(0.005*np.absolute(rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak))
        
    # plot
    x = np.hstack([qrad_peak_all[day] for day in days])
    y = np.hstack([H_peak_all[day] for day in days])
    m_nans = np.logical_not(np.isnan(y))
    s = np.hstack(s)
    x,y,s = x[m_nans], y[m_nans], s[m_nans]
    scatterDensity(ax,x,y,s,alpha=0.5)
    
    ax.set_xlabel('$Q_{rad}$ peak magnitude (K/day)')
    ax.set_ylabel('Estimated peak magnitude (K/day)')
    ax.set_title('Using magnitude approximation')
    
    #-- (b) B = 0.004 SI units
    
    ax = axs[1]
    
    H_peak_all = {}
    qrad_peak_all = {}
    s = []
    
    for day in days:
    # for day in '20200126',:
        
        #- approximated peak
        # H_peak = rad_scaling_all[day].scaling_magnitude_lw_peak*1e8
        Ns = rad_scaling_all[day].rad_features.pw.size
        H_peak = np.full(Ns,np.nan)
        for i_s in range(Ns):
            # i_z = rad_scaling_all[day].rad_features.scaling_profile_peaks.i_scaling_profile_peak[i_s]
            H_peak[i_s] = rad_scaling_all[day].scaling_magnitude_lw_peak_B040[i_s]
        H_peak_all[day] = H_peak
        
        #- true peak
        qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak
        qrad_peak_all[day] = qrad_peak
        
        s.append(0.005*np.absolute(rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak))
        
    # plot
    x = np.hstack([qrad_peak_all[day] for day in days])
    y = np.hstack([H_peak_all[day] for day in days])
    m_nans = np.logical_not(np.isnan(y))
    s = np.hstack(s)
    x,y,s = x[m_nans], y[m_nans], s[m_nans]
    scatterDensity(ax,x,y,s,alpha=0.5)
    
    ax.set_xlabel('$Q_{rad}$ peak magnitude (K/day)')
    ax.set_ylabel('Estimated peak magnitude (K/day)')
    ax.set_title('Using magnitude approximation')
    
    plt.savefig(os.path.join(figdir,'paper_approx_peak_magnitude.pdf'),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,'paper_approx_peak_magnitude.png'),bbox_inches='tight')
    
#%% (c) pdf of W(z_peak)

    fig,ax = plt.subplots(figsize=(5,5))
    
    # Store W at peak for each profile each day
    W_peak = {}
    for day in days:
        
        Ns = rad_scaling_all[day].rad_features.pw.size
        W_peak[day] = []
        
        for i_s in range(Ns):
            
            i_peak = rad_scaling_all[day].rad_features.i_lw_peak[i_s]
            W_peak[day].append(rad_features_all[day].wp_z[i_s,i_peak])
            
    # Show histogram
    W_peak_all = np.hstack([W_peak[day] for day in days])
    
    plt.hist(W_peak_all)
    # hist, bin_edges = np.histogram()
            
            