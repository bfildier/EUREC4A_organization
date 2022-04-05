#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 18:39:15 2021

@author: bfildier

Test simplified scaling derived from a stepfunction in RH in EUREC4A radiative profiles.
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import pytz 
from datetime import timedelta, timezone
from datetime import datetime as dt
import sys,os,glob
import argparse
import pickle
from math import *
from scipy.stats import gaussian_kde
# thermo calculations
import metpy.calc as mpcalc
from metpy.units import units

##-- directories and modules

# workdir = '/Users/bfildier/Code/analyses/EUREC4A/EUREC4A_organization/scripts'
workdir = os.path.dirname(os.path.realpath(__file__))
repodir = os.path.dirname(workdir)
moduledir = os.path.join(repodir,'functions')
subdirname = 'radiative_scaling'
resultdir = os.path.join(repodir,'results','radiative_features')
figdir = os.path.join(repodir,'figures',subdirname)
inputdir = '/Users/bfildier/Dropbox/Data/EUREC4A/sondes_radiative_profiles/'
# scriptsubdir = 'compare_patterns'

# Load own module
projectname = 'EUREC4A_organization'
thismodule = sys.modules[__name__]

## Graphical parameters
plt.style.use(os.path.join(matplotlib.get_configdir(),'stylelib/presentation.mplstyle'))

## Own modules
sys.path.insert(0,moduledir)
print("Own modules available:", [os.path.splitext(os.path.basename(x))[0]
                                 for x in glob.glob(os.path.join(moduledir,'*.py'))])

from radiativefeatures import *
from radiativescaling import *
from thermoConstants import *
import thermoFunctions as tf
from conditionalstats import *
from matrixoperators import *

##--- local functions

def defineSimDirectories():
        
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
    
    Ndays = len(days)
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
    
    

#%% (a) show RH, CRH below and above each level as a function of p
    

    day = '20200126'
    date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
    data_day = data_all.sel(launch_time=day)
    f = rad_features_all[day]

    # colors
    var_col = f.pw
    norm = matplotlib.colors.Normalize(vmin=var_col.min(), vmax=var_col.max())
    cmap = plt.cm.nipy_spectral
    cols = cmap(norm(var_col))
    
    Ns = data_day.dims['launch_time']
    
    pres = data_day.pressure.data/100 # hPa

    ## -- Figure --
    
    fig,axs = plt.subplots(ncols=3,nrows=1,figsize=(15,5))
    fig.suptitle(day)

    k_bottom = np.where(np.logical_not(np.isnan(f.wp_z[0])))[0][0]

    for i_s in range(Ns):
        # ax.plot(f.wp_z[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
    
        # CRH above = W(p)/Wsat(p)
        CRHabove = f.wp_z[i_s]/f.wpsat_z[i_s]
        # CRH below = (W_s-W(p))/(Wsat_s-Wsat(p))
        CRHbelow = (f.wp_z[i_s,k_bottom]-f.wp_z[i_s])/(f.wpsat_z[i_s,k_bottom]-f.wpsat_z[i_s])
        # RH
        temp = data_day.temperature.data[i_s]
        p = data_day.pressure.data[i_s]
        qv = data_day.specific_humidity.data[i_s]
        # qvsat = tf.saturationSpecificHumidity(temp,p)
        qvsat= np.float64(
            mpcalc.saturation_mixing_ratio(
            #HALO_circling.p.sel(alt=10).values)* 100 * units.Pa, era5_circling["sst"].values * units.K
            p * units.Pa, temp * units.K
            #HALO_circling.p.sel(alt=10).values * 100 * units.Pa, Raphaela_fluxes_sst_circling['sstc'].values * units.K
            ).magnitude)
        rh = qv/qvsat

        axs[0].plot(rh,pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
        axs[1].plot(CRHabove,pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
        axs[2].plot(CRHbelow,pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
        
    # ax.set_xscale('log')
    # ax.set_xlim((0.5,100))
    # ax.set_xlabel(r'$W(p) \equiv \int_z^{TOA} q_v \frac{dp}{g}$ (mm) $\propto \tau$',labelpad=-1)

    #- all y axes
    for ax in axs:
        ax.set_ylim((490,1010))
        ax.invert_yaxis()
        ax.set_ylabel('p (hPa)')
        ax.set_xlim((-0.01,1.01))
        
    axs[0].set_xlabel('RH')
    axs[1].set_xlabel('CRH above p')
    axs[2].set_xlabel('CRH below p')


    plt.savefig(os.path.join(figdir,'RH_CRHabove_and_below_%s.pdf'%day),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,'RH_CRHabove_and_below_%s.png'%day),bbox_inches='tight')
    

#%% (b) scatter plot scaling magnitude with CRH above and CRH below

    m_to_cm = 1e2
    hPa_to_Pa = 1e2
    day_to_seconds = 86400
    
    def scatterDensity(ax,x,y,s,alpha):
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        ax.scatter(x,y,c=z,s=s,alpha=0.4)

    fig,axs = plt.subplots(ncols=2,figsize=(11,5))    
    plt.subplots_adjust(wspace=0.3)

    #-- (a) peak magnitude (2020-01-26), using the simpler approximation for peak magnitudes with RH below and RH above peak
    ax = axs[0]
    
    H_peak_all = {}
    qrad_peak_all = {}
    s = []
    
    # days2show = days
    days2show = ['20200126']
    
    for day in days2show:
        
        f = rad_scaling_all[day].rad_features
        k_bottom = np.where(np.logical_not(np.isnan(f.wp_z[0])))[0][0] 
        
        #- approximated peak
        # H_peak = rad_scaling_all[day].scaling_magnitude_lw_peak*1e8
        Ns = rad_scaling_all[day].rad_features.pw.size
        H_peak = np.full(Ns,np.nan)
        for i_s in range(Ns):
            i_peak = f.i_lw_peak[i_s]
            p_peak = f.pres_lw_peak[i_s]
            # CRH above = W(p)/Wsat(p)
            CRHabove = f.wp_z[i_s]/f.wpsat_z[i_s]
            # CRH below = (W_s-W(p))/(Wsat_s-Wsat(p))
            CRHbelow = (f.wp_z[i_s,k_bottom]-f.wp_z[i_s])/(f.wpsat_z[i_s,k_bottom]-f.wpsat_z[i_s])
            # constants
            delta_nu = 50 # cm-1
            B = 0.0045 # ....m-1
            alpha = 2
            C = -gg/c_pd * (1+alpha) * pi*B * delta_nu*m_to_cm/e
            
            H_peak[i_s] = C/(p_peak*hPa_to_Pa) * CRHbelow[i_peak]/CRHabove[i_peak] * day_to_seconds
            
        H_peak_all[day] = H_peak
        
        #- true peak
        qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak
        qrad_peak_all[day] = qrad_peak
        
        s.append(0.005*np.absolute(rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak))
        
    # plot
    x = np.hstack([qrad_peak_all[day] for day in days2show])
    y = np.hstack([H_peak_all[day] for day in days2show])
    s = np.hstack(s)
    scatterDensity(ax,x,y,s,alpha=0.5)
    
    # 1:1 line
    x_ex = np.array([-15,-2])
    ax.plot(x_ex,x_ex,'k:',alpha=0.4,label='1 : 1')
    
    ax.set_xlabel('$Q_{rad}$ peak magnitude (K/day)')
    ax.set_ylabel(r'$C \frac{1}{p_\perp} \times \frac{CRH_s}{CRH_t}$ (K/day)')
    ax.set_title(days2show[0])

    #-- peak magnitude (all days), using the simpler approximation for peak magnitudes with RH below and RH above peak
    ax = axs[1]
    
    H_peak_all = {}
    qrad_peak_all = {}
    s = []
    
    days2show = days
    # days2show = ['20200126']
    
    for day in days2show:
        
        f = rad_scaling_all[day].rad_features
        k_bottom = np.where(np.logical_not(np.isnan(f.wp_z[0])))[0][0] 
        
        #- approximated peak
        # H_peak = rad_scaling_all[day].scaling_magnitude_lw_peak*1e8
        Ns = rad_scaling_all[day].rad_features.pw.size
        H_peak = np.full(Ns,np.nan)
        for i_s in range(Ns):
            i_peak = f.i_lw_peak[i_s]
            p_peak = f.pres_lw_peak[i_s]
            # CRH above = W(p)/Wsat(p)
            CRHabove = f.wp_z[i_s]/f.wpsat_z[i_s]
            # CRH below = (W_s-W(p))/(Wsat_s-Wsat(p))
            CRHbelow = (f.wp_z[i_s,k_bottom]-f.wp_z[i_s])/(f.wpsat_z[i_s,k_bottom]-f.wpsat_z[i_s])
            # constants
            delta_nu = 50 # cm-1
            B = 0.0045 # ....m-1
            alpha = 2
            C = -gg/c_pd * (1+alpha) * pi*B * delta_nu*m_to_cm/e
            
            H_peak[i_s] = C/(p_peak*hPa_to_Pa) * CRHbelow[i_peak]/CRHabove[i_peak] * day_to_seconds
            
        H_peak_all[day] = H_peak
        
        #- true peak
        qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak
        qrad_peak_all[day] = qrad_peak
        
        s.append(0.005*np.absolute(rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak))
        
    # plot
    x = np.hstack([qrad_peak_all[day] for day in days2show])
    y = np.hstack([H_peak_all[day] for day in days2show])
    s = np.hstack(s)
    scatterDensity(ax,x,y,s,alpha=0.5)
    
    # 1:1 line
    x_ex = np.array([-18,-2])
    ax.plot(x_ex,x_ex,'k:',alpha=0.4,label='1 : 1')
    
    ax.set_xlabel('$Q_{rad}$ peak magnitude (K/day)')
    ax.set_ylabel(r'$C \frac{1}{p_\perp} \times \frac{CRH_s}{CRH_t}$ (K/day)')
    ax.set_title(r'All days, $\Delta \nu = 50$ cm$^{-1}$')

    ax.set_ylim((-20,None))

    plt.savefig(os.path.join(figdir,'approx_peak_with_rh_ratio.pdf'),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,'approx_peak_with_rh_ratio.png'),bbox_inches='tight')
    
#%% (c) scatter plot scaling magnitude with CRH above and CRH below / use spectral integral

    m_to_cm = 1e2
    day_to_seconds = 86400
    hPa_to_Pa = 1e2
    
    # colors
    var_col = np.arange(Ndays)
    norm = matplotlib.colors.Normalize(vmin=var_col.min(), vmax=var_col.max())
    cmap = plt.cm.nipy_spectral
    cols = cmap(norm(var_col))

    fig,axs = plt.subplots(ncols=2,figsize=(11,5))    
    plt.subplots_adjust(wspace=0.3)

    #-- (a) peak magnitude (color by density), using the simpler approximation for peak magnitudes with RH below and RH above peak
    ax = axs[0]
    
    H_peak_all = {}
    qrad_peak_all = {}
    s = []
    
    days2show = days
    # days2show = ['20200126']
    
    for day in days2show:
        
        rs = rad_scaling_all[day]
        f = rad_scaling_all[day].rad_features
        k_bottom = np.where(np.logical_not(np.isnan(f.wp_z[0])))[0][0] 
        
        #- approximated peak
        # H_peak = rad_scaling_all[day].scaling_magnitude_lw_peak*1e8
        Ns = rad_scaling_all[day].rad_features.pw.size
        H_peak = np.full(Ns,np.nan)
        for i_s in range(Ns):
            i_peak = f.i_lw_peak[i_s]
            p_peak = f.pres_lw_peak[i_s]
            # CRH above = W(p)/Wsat(p)
            CRHabove = f.wp_z[i_s]/f.wpsat_z[i_s]
            # CRH below = (W_s-W(p))/(Wsat_s-Wsat(p))
            CRHbelow = (f.wp_z[i_s,k_bottom]-f.wp_z[i_s])/(f.wpsat_z[i_s,k_bottom]-f.wpsat_z[i_s])
            # spectral integral
            spec_int = rs.spectral_integral_rot[i_s][i_peak]
            # constants
            alpha = 2
            C = -gg/c_pd * (1+alpha)
            
            H_peak[i_s] = C/(p_peak*hPa_to_Pa) * CRHbelow[i_peak]/CRHabove[i_peak] * spec_int * day_to_seconds
            
        H_peak_all[day] = H_peak
        
        #- true peak
        qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak
        qrad_peak_all[day] = qrad_peak
        
        s.append(0.005*np.absolute(rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak))
        
    # plot
    x = np.hstack([qrad_peak_all[day] for day in days2show])
    y = np.hstack([H_peak_all[day] for day in days2show])
    s = np.hstack(s)
    scatterDensity(ax,x,y,s,alpha=0.5)
    
    # 1:1 line
    x_ex = np.array([-18,-2])
    ax.plot(x_ex,x_ex,'k:',alpha=0.4,label='1 : 1')
    
    ax.set_xlabel('$Q_{rad}$ peak magnitude (K/day)')
    ax.set_ylabel(r'$-\frac{g}{c_p} \frac{1}{p_\perp} (1+\alpha) \frac{CRH_s}{CRH_t} \int B \phi d\nu$ (K/day)')
    ax.set_title('All days')

    #-- (b) peak magnitude (color by day), using the simpler approximation for peak magnitudes with RH below and RH above peak
    ax = axs[1]
    
    H_peak_all = {}
    qrad_peak_all = {}
    s = []
    
    days2show = days
    # days2show = ['20200126']
    
    for day in days2show:
        
        rs = rad_scaling_all[day]
        f = rad_scaling_all[day].rad_features
        k_bottom = np.where(np.logical_not(np.isnan(f.wp_z[0])))[0][0] 
        
        #- approximated peak
        # H_peak = rad_scaling_all[day].scaling_magnitude_lw_peak*1e8
        Ns = rad_scaling_all[day].rad_features.pw.size
        H_peak = np.full(Ns,np.nan)
        for i_s in range(Ns):
            i_peak = f.i_lw_peak[i_s]
            p_peak = f.pres_lw_peak[i_s]
            # CRH above = W(p)/Wsat(p)
            CRHabove = f.wp_z[i_s]/f.wpsat_z[i_s]
            # CRH below = (W_s-W(p))/(Wsat_s-Wsat(p))
            CRHbelow = (f.wp_z[i_s,k_bottom]-f.wp_z[i_s])/(f.wpsat_z[i_s,k_bottom]-f.wpsat_z[i_s])
            # spectral integral
            spec_int = rs.spectral_integral_rot[i_s][i_peak]
            # constants
            # delta_nu = 220 # cm-1
            alpha = 2
            C = -gg/c_pd * (1+alpha)
            
            H_peak[i_s] = C/(p_peak*hPa_to_Pa) * CRHbelow[i_peak]/CRHabove[i_peak] * spec_int * day_to_seconds
            
        H_peak_all[day] = H_peak
        
        #- true peak
        qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak
        qrad_peak_all[day] = qrad_peak
        
        s.append(0.005*np.absolute(rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak))
    
    # plot
    # x = np.hstack([qrad_peak_all[day] for day in days2show])
    # y = np.hstack([H_peak_all[day] for day in days2show])
    # s = np.hstack(s)
    
    for i_d in range(Ndays):
        
        day = days[i_d]
        x = qrad_peak_all[day]
        y = H_peak_all[day]
        
        ax.scatter(x,y,s[i_d],color=cols[i_d],alpha=0.5,label=day)
        
    # 1:1 line
    x_ex = np.array([-18,-2])
    ax.plot(x_ex,x_ex,'k:',alpha=0.4,label='1 : 1')
        
    ax.set_xlabel('$Q_{rad}$ peak magnitude (K/day)')
    ax.set_ylabel(r'$-\frac{g}{c_p} \frac{1}{p_\perp} (1+\alpha) \frac{CRH_s}{CRH_t} \int B \phi d\nu$ (K/day)')
    ax.set_title(r'All days')
    
    ax.legend(loc='lower center',ncol=3,fontsize=8,framealpha=0.7)

    # ax.set_ylim((-10,None))

    plt.savefig(os.path.join(figdir,'approx_peak_with_rh_ratio_spec_int_rot.pdf'),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,'approx_peak_with_rh_ratio_spec_int_rot.png'),bbox_inches='tight')
    
    
#%% (c2) scatter plot scaling magnitude with CRH above and CRH below / use spectral integral in rot + v-r bands

    m_to_cm = 1e2
    day_to_seconds = 86400
    hPa_to_Pa = 1e2
    
    # colors
    var_col = np.arange(Ndays)
    norm = matplotlib.colors.Normalize(vmin=var_col.min(), vmax=var_col.max())
    cmap = plt.cm.nipy_spectral
    cols = cmap(norm(var_col))

    fig,axs = plt.subplots(ncols=2,figsize=(11,5))    
    plt.subplots_adjust(wspace=0.3)

    #-- (a) peak magnitude (color by density), using the simpler approximation for peak magnitudes with RH below and RH above peak
    ax = axs[0]
    
    H_peak_all = {}
    qrad_peak_all = {}
    s = []
    
    days2show = days
    # days2show = ['20200126']
    
    for day in days2show:
        
        rs = rad_scaling_all[day]
        f = rad_scaling_all[day].rad_features
        k_bottom = np.where(np.logical_not(np.isnan(f.wp_z[0])))[0][0] 
        
        #- approximated peak
        # H_peak = rad_scaling_all[day].scaling_magnitude_lw_peak*1e8
        Ns = rad_scaling_all[day].rad_features.pw.size
        H_peak = np.full(Ns,np.nan)
        for i_s in range(Ns):
            i_peak = f.i_lw_peak[i_s]
            p_peak = f.pres_lw_peak[i_s]
            # CRH above = W(p)/Wsat(p)
            CRHabove = f.wp_z[i_s]/f.wpsat_z[i_s]
            # CRH below = (W_s-W(p))/(Wsat_s-Wsat(p))
            CRHbelow = (f.wp_z[i_s,k_bottom]-f.wp_z[i_s])/(f.wpsat_z[i_s,k_bottom]-f.wpsat_z[i_s])
            # spectral integral
            spec_int = rs.spectral_integral_rot[i_s][i_peak] + rs.spectral_integral_vr[i_s][i_peak]
            # constants
            alpha = 2
            C = -gg/c_pd * (1+alpha)
            
            H_peak[i_s] = C/(p_peak*hPa_to_Pa) * CRHbelow[i_peak]/CRHabove[i_peak] * spec_int * day_to_seconds
            
        H_peak_all[day] = H_peak
        
        #- true peak
        qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak
        qrad_peak_all[day] = qrad_peak
        
        s.append(0.005*np.absolute(rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak))
        
    # plot
    x = np.hstack([qrad_peak_all[day] for day in days2show])
    y = np.hstack([H_peak_all[day] for day in days2show])
    s = np.hstack(s)
    scatterDensity(ax,x,y,s,alpha=0.5)
    
    # 1:1 line
    x_ex = np.array([-18,-2])
    ax.plot(x_ex,x_ex,'k:',alpha=0.4,label='1 : 1')
    
    ax.set_xlabel('$Q_{rad}$ peak magnitude (K/day)')
    ax.set_ylabel(r'$-\frac{g}{c_p} \frac{1}{p_\perp} (1+\alpha) \frac{CRH_s}{CRH_t} \int B \phi d\nu$ (K/day)')
    ax.set_title('All days')

    #-- (b) peak magnitude (color by day), using the simpler approximation for peak magnitudes with RH below and RH above peak
    ax = axs[1]
    
    H_peak_all = {}
    qrad_peak_all = {}
    s = []
    
    days2show = days
    # days2show = ['20200126']
    
    for day in days2show:
        
        rs = rad_scaling_all[day]
        f = rad_scaling_all[day].rad_features
        k_bottom = np.where(np.logical_not(np.isnan(f.wp_z[0])))[0][0] 
        
        #- approximated peak
        # H_peak = rad_scaling_all[day].scaling_magnitude_lw_peak*1e8
        Ns = rad_scaling_all[day].rad_features.pw.size
        H_peak = np.full(Ns,np.nan)
        for i_s in range(Ns):
            i_peak = f.i_lw_peak[i_s]
            p_peak = f.pres_lw_peak[i_s]
            # CRH above = W(p)/Wsat(p)
            CRHabove = f.wp_z[i_s]/f.wpsat_z[i_s]
            # CRH below = (W_s-W(p))/(Wsat_s-Wsat(p))
            CRHbelow = (f.wp_z[i_s,k_bottom]-f.wp_z[i_s])/(f.wpsat_z[i_s,k_bottom]-f.wpsat_z[i_s])
            # spectral integral
            spec_int = rs.spectral_integral_rot[i_s][i_peak] + rs.spectral_integral_vr[i_s][i_peak]
            # constants
            # delta_nu = 220 # cm-1
            alpha = 2
            C = -gg/c_pd * (1+alpha)
            
            H_peak[i_s] = C/(p_peak*hPa_to_Pa) * CRHbelow[i_peak]/CRHabove[i_peak] * spec_int * day_to_seconds
            
        H_peak_all[day] = H_peak
        
        #- true peak
        qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak
        qrad_peak_all[day] = qrad_peak
        
        s.append(0.005*np.absolute(rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak))
    
    # plot
    # x = np.hstack([qrad_peak_all[day] for day in days2show])
    # y = np.hstack([H_peak_all[day] for day in days2show])
    # s = np.hstack(s)
    
    for i_d in range(Ndays):
        
        day = days[i_d]
        x = qrad_peak_all[day]
        y = H_peak_all[day]
        
        ax.scatter(x,y,s[i_d],color=cols[i_d],alpha=0.5,label=day)
        
    # 1:1 line
    x_ex = np.array([-18,-2])
    ax.plot(x_ex,x_ex,'k:',alpha=0.4,label='1 : 1')
        
    ax.set_xlabel('$Q_{rad}$ peak magnitude (K/day)')
    ax.set_ylabel(r'$-\frac{g}{c_p} \frac{1}{p_\perp} (1+\alpha) \frac{CRH_s}{CRH_t} \int B \phi d\nu$ (K/day)')
    ax.set_title(r'All days')
    
    ax.legend(loc='lower center',ncol=3,fontsize=8,framealpha=0.7)

    # ax.set_ylim((-10,None))

    plt.savefig(os.path.join(figdir,'approx_peak_with_rh_ratio_spec_int_rot_vr.pdf'),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,'approx_peak_with_rh_ratio_spec_int_rot_vr.png'),bbox_inches='tight')


#%% (d) show RH profile, int(RH) below and above each level as a function of p

    import metpy.calc as mpcalc
    from metpy.units import units

    hPa_to_Pa = 1e2
    Pa_to_hPa = 1e-2

    day = '20200126'
    date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
    data_day = data_all.sel(launch_time=day)
    f = rad_features_all[day]

    # colors
    var_col = f.pw
    norm = matplotlib.colors.Normalize(vmin=var_col.min(), vmax=var_col.max())
    cmap = plt.cm.nipy_spectral
    cols = cmap(norm(var_col))

    Ns = data_day.dims['launch_time']

    pres = data_day.pressure.data/100 # hPa

    ## -- Figure --
    
    fig,axs = plt.subplots(ncols=2,nrows=1,figsize=(9,5))
    
    k_bottom = np.where(np.logical_not(np.isnan(f.wp_z[0])))[0][0]

    for i_s in range(Ns):
        # ax.plot(f.wp_z[i_s],pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
    
        # CRH above = W(p)/Wsat(p)
        # CRHabove = f.wp_z[i_s]/f.wpsat_z[i_s]
        # CRH below = (W_s-W(p))/(Wsat_s-Wsat(p))
        # CRHbelow = (f.wp_z[i_s,k_bottom]-f.wp_z[i_s])/(f.wpsat_z[i_s,k_bottom]-f.wpsat_z[i_s])
        # RH profile
        
        temp = data_day.temperature.data[i_s]
        p = data_day.pressure.data[i_s]
        qv = data_day.specific_humidity.data[i_s]
        # qvsat = tf.saturationSpecificHumidity(temp,p)
        qvsat= np.float64(
            mpcalc.saturation_mixing_ratio(
            #HALO_circling.p.sel(alt=10).values)* 100 * units.Pa, era5_circling["sst"].values * units.K
            p * units.Pa, temp * units.K
            #HALO_circling.p.sel(alt=10).values * 100 * units.Pa, Raphaela_fluxes_sst_circling['sstc'].values * units.K
            ).magnitude)
        
        rh = qv/qvsat
        

        axs[0].plot(qv,p*Pa_to_hPa,c=cols[i_s],linewidth=0.5,alpha=0.5)
        axs[1].plot(rh,p*Pa_to_hPa,c=cols[i_s],linewidth=0.5,alpha=0.5)
        # axs[0].plot(CRHabove,pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
        # axs[1].plot(CRHbelow,pres[i_s],c=cols[i_s],linewidth=0.5,alpha=0.5)
        
    # ax.set_xscale('log')
    # ax.set_xlim((0.5,100))
    # ax.set_xlabel(r'$W(p) \equiv \int_z^{TOA} q_v \frac{dp}{g}$ (mm) $\propto \tau$',labelpad=-1)

    #- all y axes
    for ax in axs:
        ax.set_ylim((490,1010))
        ax.invert_yaxis()
        ax.set_ylabel('p (hPa)')
        
    axs[0].set_xlabel(r'$q_v$')
    axs[1].set_xlabel('RH')
    # axs[1].set_xlabel('CRH below p')


    plt.savefig(os.path.join(figdir,'qv_RH_profiles_%s.pdf'%day),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,'qv_RH_profiles_%s.png'%day),bbox_inches='tight')
    
    
#%% (e) 3 scalings -- scatter plot scaling magnitude with CRH above and CRH below / use spectral integral in rot + v-r bands

    m_to_cm = 1e2
    day_to_seconds = 86400
    hPa_to_Pa = 1e2
    
    # colors
    var_col = np.arange(Ndays)
    norm = matplotlib.colors.Normalize(vmin=var_col.min(), vmax=var_col.max())
    cmap = plt.cm.nipy_spectral
    cols = cmap(norm(var_col))

    fig,axs = plt.subplots(ncols=3,figsize=(17,5))    
    plt.subplots_adjust(wspace=0.3)


    #-- (a) peak magnitude (color by density), using the integral and the first approximation with beta
    ax = axs[0]
    
    H_peak_all = {}
    qrad_peak_all = {}
    s = []
    
    days2show = days
    # days2show = ['20200126']
    
    for day in days2show:
        
        rs = rad_scaling_all[day]
        f = rad_scaling_all[day].rad_features
        k_bottom = np.where(np.logical_not(np.isnan(f.wp_z[0])))[0][0] 
        
        #- approximated peak
        # H_peak = rad_scaling_all[day].scaling_magnitude_lw_peak*1e8
        Ns = rad_scaling_all[day].rad_features.pw.size
        H_peak = np.full(Ns,np.nan)
        for i_s in range(Ns):
            i_peak = f.i_lw_peak[i_s]
            p_peak = f.pres_lw_peak[i_s]
            
            # -- beta
            beta = f.beta_peak[i_s]
            # # Manual calculation
            # ln_W = np.log(f.wp_z)
            # ln_p = np.log(data_day.pressure.data) # Pa
            # ln_p_mean = np.nanmean(ln_p,axis=0)
            # beta_profiles,_ = mo.derivative(ln_W,ln_p_mean,axis=1)
            # i_peak = f.i_lw_peak[i_s]
            # beta = beta_profiles[i_s,i_peak]

            # spectral integral
            spec_int = rs.spectral_integral_rot[i_s][i_peak] + rs.spectral_integral_vr[i_s][i_peak]
            # constants
            C = -gg/c_pd
            
            H_peak[i_s] = C/(p_peak*hPa_to_Pa) * beta * spec_int * day_to_seconds
            
        H_peak_all[day] = H_peak
        
        #- true peak
        qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak
        qrad_peak_all[day] = qrad_peak
        
        s.append(0.005*np.absolute(rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak))
        
    # plot
    x = np.hstack([qrad_peak_all[day] for day in days2show])
    y = np.hstack([H_peak_all[day] for day in days2show])
    s = np.hstack(s)
    scatterDensity(ax,x,y,s,alpha=0.5)
    
    # 1:1 line
    x_ex = np.array([-18,-2])
    ax.plot(x_ex,x_ex,'k:',alpha=0.4,label='1 : 1')
    
    ax.set_xlabel('True $Q_{rad}$ (K/day)')
    ax.set_ylabel('$Q_{rad}$ estimate (K/day)')
    ax.set_title(r'$-\frac{g}{c_p} \frac{\beta_\perp}{p_\perp} \int B \phi d\nu$')



    #-- (b) peak magnitude (color by density), using the integral and the simpler approximation for beta with RH below and RH above peak
    ax = axs[1]
    
    H_peak_all = {}
    qrad_peak_all = {}
    s = []
    
    days2show = days
    # days2show = ['20200126']
    
    for day in days2show:
        
        rs = rad_scaling_all[day]
        f = rad_scaling_all[day].rad_features
        k_bottom = np.where(np.logical_not(np.isnan(f.wp_z[0])))[0][0] 
        
        #- approximated peak
        # H_peak = rad_scaling_all[day].scaling_magnitude_lw_peak*1e8
        Ns = rad_scaling_all[day].rad_features.pw.size
        H_peak = np.full(Ns,np.nan)
        for i_s in range(Ns):
            i_peak = f.i_lw_peak[i_s]
            p_peak = f.pres_lw_peak[i_s]
            # CRH above = W(p)/Wsat(p)
            CRHabove = f.wp_z[i_s]/f.wpsat_z[i_s]
            # CRH below = (W_s-W(p))/(Wsat_s-Wsat(p))
            CRHbelow = (f.wp_z[i_s,k_bottom]-f.wp_z[i_s])/(f.wpsat_z[i_s,k_bottom]-f.wpsat_z[i_s])
            # spectral integral
            spec_int = rs.spectral_integral_rot[i_s][i_peak] + rs.spectral_integral_vr[i_s][i_peak]
            # constants
            alpha = 2
            C = -gg/c_pd * (1+alpha)
            
            H_peak[i_s] = C/(p_peak*hPa_to_Pa) * CRHbelow[i_peak]/CRHabove[i_peak] * spec_int * day_to_seconds
            
        H_peak_all[day] = H_peak
        
        #- true peak
        qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak
        qrad_peak_all[day] = qrad_peak
        
        s.append(0.005*np.absolute(rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak))
        
    # plot
    x = np.hstack([qrad_peak_all[day] for day in days2show])
    y = np.hstack([H_peak_all[day] for day in days2show])
    s = np.hstack(s)
    scatterDensity(ax,x,y,s,alpha=0.5)
    
    # 1:1 line
    x_ex = np.array([-18,-2])
    ax.plot(x_ex,x_ex,'k:',alpha=0.4,label='1 : 1')
    
    ax.set_xlabel('True $Q_{rad}$ (K/day)')
    ax.set_ylabel('$Q_{rad}$ estimate (K/day)')
    ax.set_title(r'$-\frac{g}{c_p} \frac{1}{p_\perp} (1+\alpha) \frac{CRH_s}{CRH_t} \int B \phi d\nu$')



    #-- (c) peak magnitude (color by day), using the simpler approximation for peak magnitudes with RH below and RH above peak
    ax = axs[2]
    
    H_peak_all = {}
    qrad_peak_all = {}
    s = []
    
    days2show = days
    # days2show = ['20200126']
    
    for day in days2show:
        
        rs = rad_scaling_all[day]
        f = rad_scaling_all[day].rad_features
        k_bottom = np.where(np.logical_not(np.isnan(f.wp_z[0])))[0][0] 
        
        #- approximated peak
        # H_peak = rad_scaling_all[day].scaling_magnitude_lw_peak*1e8
        Ns = rad_scaling_all[day].rad_features.pw.size
        H_peak = np.full(Ns,np.nan)
        for i_s in range(Ns):
            i_peak = f.i_lw_peak[i_s]
            p_peak = f.pres_lw_peak[i_s]
            # CRH above = W(p)/Wsat(p)
            CRHabove = f.wp_z[i_s]/f.wpsat_z[i_s]
            # CRH below = (W_s-W(p))/(Wsat_s-Wsat(p))
            CRHbelow = (f.wp_z[i_s,k_bottom]-f.wp_z[i_s])/(f.wpsat_z[i_s,k_bottom]-f.wpsat_z[i_s])
            # approximation of spectral integral
            # spec_int = rs.spectral_integral_rot[i_s][i_peak] + rs.spectral_integral_vr[i_s][i_peak]
            piB_star = 0.0054
            delta_nu = 120 # cm-1
            spec_int_approx = piB_star * delta_nu*m_to_cm/e
            # constants
            # delta_nu = 220 # cm-1
            alpha = 2
            C = -gg/c_pd * (1+alpha)
            
            H_peak[i_s] = C/(p_peak*hPa_to_Pa) * CRHbelow[i_peak]/CRHabove[i_peak] * spec_int_approx * day_to_seconds
            
        H_peak_all[day] = H_peak
        
        #- true peak
        qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak
        qrad_peak_all[day] = qrad_peak
        
        s.append(0.005*np.absolute(rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak))
    
    # plot
    # x = np.hstack([qrad_peak_all[day] for day in days2show])
    # y = np.hstack([H_peak_all[day] for day in days2show])
    # s = np.hstack(s)
    
    for i_d in range(Ndays):
        
        day = days[i_d]
        x = qrad_peak_all[day]
        y = H_peak_all[day]
        
        ax.scatter(x,y,s[i_d],color=cols[i_d],alpha=0.5,label=day)
        
    # 1:1 line
    x_ex = np.array([-18,-2])
    ax.plot(x_ex,x_ex,'k:',alpha=0.4,label='1 : 1')
        
    ax.set_xlabel('True $Q_{rad}$ (K/day)')
    ax.set_ylabel('$Q_{rad}$ estimate (K/day)')
    ax.set_title(r'$-\frac{g}{c_p} \frac{1}{p_\perp} (1+\alpha) \frac{CRH_s}{CRH_t} B_{\nu^\star} \frac{\Delta \nu}{e}$')
    
    ax.legend(loc='lower center',ncol=3,fontsize=8,framealpha=0.7)
    
    
    for ax in axs:
        ax.set_ylim((-20,-1))

    plt.savefig(os.path.join(figdir,'approx_peak_3_scalings.pdf'),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,'approx_peak_3_scalings.png'),bbox_inches='tight')


#%% Find nu_star in both bands

    W = 3 # mm
    kappa_star = 1/W
    nu_star_rot = rs.nu(kappa_star,'rot') # m-1
    nu_star_vr = rs.nu(kappa_star,'vr') # m-1

    temp_ref = 290 # K
    piB_star_rot = pi*rs.planck(nu_star_rot,temp_ref)
    piB_star_vr = pi*rs.planck(nu_star_vr,temp_ref)

#%% why is the first scaling too small? check beta calculation

    day = '20200124'
    rs = rad_scaling_all[day]
    f = rad_scaling_all[day].rad_features
    data_day = data_all.sel(launch_time=day)
    mo = MatrixOperators()
    
    # Calculation in rs
    beta = f.beta_peak
    
    # Manual calculation
    ln_W = np.log(f.wp_z)
    ln_p = np.log(data_day.pressure.data) # Pa
    ln_p_mean = np.nanmean(ln_p,axis=0)
    beta_profiles,_ = mo.derivative(ln_W,ln_p_mean,axis=1)

    Ns = beta_profiles.shape[0]
    beta_manual = np.full((Ns),np.nan)
    for i_s in range(Ns):
        i_peak = f.i_lw_peak[i_s]
        beta_manual[i_s] = beta_profiles[i_s,i_peak]
        
    fig,ax = plt.subplots(figsize=(5,5))
    
    # points
    ax.scatter(beta_manual,beta)
    # 1:1 line
    x_ex = np.array([8,43])
    ax.plot(x_ex,x_ex,'k:',alpha=0.4,label='1 : 1')
    
    ax.set_xlabel(r'manual $\beta$')
    ax.set_ylabel(r'precomputed $\beta$')
    
    
#%% find alpha from data

    # campaign mean temperature profile
    temp_mean = np.nanmean(data_all.temperature.data,axis=0)
    pres_mean = np.nanmean(data_all.pressure.data,axis=0)
    
    fig,axs = plt.subplots(ncols=3,figsize=(15,5))
    
    #-- T(p)
    ax = axs[0]
    ax.plot(data_all.temperature.data,data_all.pressure.data/100,c='k',linewidth=0.1,alpha=0.05)
    ax.plot(temp_mean,pres_mean/100,'r')
    ax.invert_yaxis()
    
    #-- qvstar(p)
    qvstar_mean = np.float64(
            mpcalc.saturation_mixing_ratio(
            #HALO_circling.p.sel(alt=10).values)* 100 * units.Pa, era5_circling["sst"].values * units.K
            pres_mean * units.Pa, temp_mean * units.K
            #HALO_circling.p.sel(alt=10).values * 100 * units.Pa, Raphaela_fluxes_sst_circling['sstc'].values * units.K
            ).magnitude)
    
    qvstar = np.float64(
            mpcalc.saturation_mixing_ratio(
            #HALO_circling.p.sel(alt=10).values)* 100 * units.Pa, era5_circling["sst"].values * units.K
            data_all.pressure.data * units.Pa, data_all.temperature.data * units.K
            #HALO_circling.p.sel(alt=10).values * 100 * units.Pa, Raphaela_fluxes_sst_circling['sstc'].values * units.K
            ).magnitude)
    
    ax = axs[1]
    ax.plot(qvstar,data_all.pressure.data/100,c='k',linewidth=0.1,alpha=0.05)
    ax.plot(qvstar_mean,pres_mean/100,'r')
    ax.invert_yaxis()
    
    #-- alpha (p)
    ln_qvstar = np.log(qvstar)
    ln_p = np.log(pres_mean)
    alpha_mean,_ = mo.derivative(ln_qvstar,ln_p)
    
    # all
    
    ax = axs[2]
    ax.plot(alpha,pres_mean/100)
    ax.invert_yaxis()