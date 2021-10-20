#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 16:45:35 2021

@author: bfildier

Draw all QV and QRADLW profiles for a given day
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
# import matplotlib.image as mpimg
from math import ceil

# geodesic distances and displacements
import geopy
import geopy.distance
# map display
# import cartopy.crs as ccrs

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
    
if __name__ == "__main__":
    
    # arguments
    parser = argparse.ArgumentParser(description="Compute and store features from radiative profile data")
    parser.add_argument('--day', default='20200126',help="day of analysis")
    parser.add_argument('--overwrite',type=bool,nargs='?',default=False)
    
    args = parser.parse_args()
    day = args.day
    
    # day = '20200126'
    # date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
    
    days =          '20200122','20200124','20200126','20200128','20200131','20200202','20200205','20200207','20200209','20200211','20200213'
    name_pattern =  'Fish',    'Fish',    'Fish',    'Gravel',  'Fish',    'Flower',  'Gravel',  'Flower',    'Sugar',   'Sugar',   'Fish'
    confidence_pattern = 'High','Medium', 'Medium',     'Low',     'Low',     'High',    'High',    'High',  'Medium',  'Medium',  'High'
    col_pattern = {'':'silver',
                   'Fish':'navy',
                   'Gravel':'orange',
                   'Sugar':'seagreen',
                   'Flower':'firebrick'}
    
    # box of analysis
    lat_box = 11,16
    lon_box = -60,-52
    
    # varids
    varids = 'QRAD','QRADSW','QRADLW','QV'
    
    # paths
    defineSimDirectories(day)
    
    mo = MatrixOperators()
    
    ###--- Load data ---###
    
    # Profiles
    radprf = xr.open_dataset(os.path.join(inputdir,'rad_profiles_CF.nc'))
    # choose profiles for that day that start at bottom
    data_all = radprf.where(radprf.z_min<=50,drop=True)
    data_day = data_all.sel(launch_time=day)
    
    # coordinates
    z = data_all.alt.values/1e3 # km
    pres = np.nanmean(data_all.pressure.data,axis=dim_t)/100 # hPa
    
    # Radiative features
    rad_features = {}
    
    for day in days:

        #-- Radiative features
        features_filename = 'rad_features.pickle'
        print('loading %s'%features_filename)
        # load
        features_path = os.path.join(resultdir,day,features_filename)
        f = pickle.load(open(features_path,'rb'))
        # store
        rad_features_all[day] = f
    
    
#%%    ###--- draw profiles ---###
    
    
    # Data
    day = '20200213'
    date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
    data_day = data_all.sel(launch_time=day)    
    
    #- Mask
    # |qrad| > 5 K/day
    qrad_peak = np.absolute(rad_features_all[day].qrad_lw_peak)
    keep_large = qrad_peak > 5 # K/day
    # in box
    lon_day = data_all.sel(launch_time=day).longitude[:,50]
    lat_day = data_all.sel(launch_time=day).latitude[:,50]
    keep_box = np.logical_and(lon_day < lon_box[1], lat_day >= lat_box[0])
    # high-level peak
    keep_high = rad_features_all[day].z_net_peak > 5000 # m
    # combined
    k = np.logical_and(np.logical_and(keep_large,keep_box),keep_high)
    
    fig,axs = plt.subplots(ncols=2)
    
    # relative humidity    
    rh = data_day.relative_humidity.values[k,:]*100
    # lw cooling
    qradlw = rad_features_all[day].qrad_lw_smooth[k,:]
    
    for i_s in range(rh.shape[0]):
        
        axs[0].plot(rh[i_s],z,c='k',linewidth=0.3,alpha=0.4)
        axs[1].plot(qradlw[i_s],z,c='k',linewidth=0.3,alpha=0.4)
        
    axs[0].set_xlabel('Relative humidity (%)')
    axs[1].set_xlabel('LW cooling (K/day)')
    axs[0].set_ylabel('z (km)')
    
    plt.savefig(os.path.join(figdir,scriptsubdir,'profiles_high_level_peaks_above5km_%s.pdf'%day),bbox_inches='tight')
    
    
        
    # Data
    day = '20200213'
    date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
    data_day = data_all.sel(launch_time=day)    
    
    #- Mask
    # |qrad| > 5 K/day
    qrad_peak = np.absolute(rad_features_all[day].qrad_lw_peak)
    keep_large = qrad_peak > 5 # K/day
    # in box
    lon_day = data_all.sel(launch_time=day).longitude[:,50]
    lat_day = data_all.sel(launch_time=day).latitude[:,50]
    keep_box = np.logical_and(lon_day < lon_box[1], lat_day >= lat_box[0])
    # high-level peak
    keep_high =  np.logical_and(rad_features_all[day].z_net_peak < 5000, # m
                                rad_features_all[day].z_net_peak > 4000)
    # combined
    k = np.logical_and(np.logical_and(keep_large,keep_box),keep_high)
    
    fig,axs = plt.subplots(ncols=2)
    
    # relative humidity    
    rh = data_day.relative_humidity.values[k,:]*100
    # lw cooling
    qradlw = rad_features_all[day].qrad_lw_smooth[k,:]
    
    for i_s in range(rh.shape[0]):
        
        axs[0].plot(rh[i_s],z,c='k',linewidth=0.3,alpha=0.4)
        axs[1].plot(qradlw[i_s],z,c='k',linewidth=0.3,alpha=0.4)
        
    axs[0].set_xlabel('Relative humidity (%)')
    axs[1].set_xlabel('LW cooling (K/day)')
    axs[0].set_ylabel('z (km)')
    
    plt.savefig(os.path.join(figdir,scriptsubdir,'profiles_high_level_peaks_btw4and5km_%s.pdf'%day),bbox_inches='tight')
    