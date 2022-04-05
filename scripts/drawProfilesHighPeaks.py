#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 16:45:35 2021

@author: bfildier

Draw all QV and QRADLW profiles for a given day
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
from math import ceil

# geodesic distances and displacements
import geopy
import geopy.distance
# map display
# import cartopy.crs as ccrs

# ## Graphical parameters
# plt.style.use(os.path.join(matplotlib.get_configdir(),'stylelib/presentation.mplstyle'))

##-- directories and modules

workdir = '/Users/bfildier/Code/analyses/EUREC4A/EUREC4A_organization/scripts'
# workdir = os.path.dirname(os.path.realpath(__file__))
repodir = os.path.dirname(workdir)
moduledir = os.path.join(repodir,'functions')
subdirname = 'radiative_features'
resultdir = os.path.join(repodir,'results',subdirname)
inputdir = '/Users/bfildier/Dropbox/Data/EUREC4A/sondes_radiative_profiles/'
scriptdir = 'high_level_peaks'
figdir = os.path.join(repodir,'figures',scriptdir)


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

def defineSimDirectories():
        
    # create output directory if not there
    os.makedirs(os.path.join(figdir),exist_ok=True)

if __name__ == "__main__":
    
    # arguments
    parser = argparse.ArgumentParser(description="Compute and store features from radiative profile data")
    # parser.add_argument('--day', default='20200126',help="day of analysis")
    parser.add_argument('--overwrite',type=bool,nargs='?',default=False)
    
    args = parser.parse_args()
    # day = args.day
    
    day = '20200126'
    date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
    
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
    
    dim_t = 0
    
    # varids
    varids = 'QRAD','QRADSW','QRADLW','QV'
    
    # paths
    defineSimDirectories()
    
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
    rad_features_all = {}
    
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


    days_high_peaks = '20200213', '20200213', '20200211', '20200209', '20200209', '20200128', '20200124', '20200122'
    z_min_all = 5000, 4000, 4000, 3500, 5500, 4000, 3000, 3200
    z_max_all = 9000, 5000, 6000, 5500, 8500, 6000, 4500, 4000


    def getProfiles(rad_features, data_day, z_min, z_max):
        
        #- Mask
        # |qrad| > 5 K/day
        qrad_peak = np.absolute(rad_features.qrad_lw_peak)
        keep_large = qrad_peak > 5 # K/day
        # in box
        lon_day = data_day.longitude[:,50]
        lat_day = data_day.latitude[:,50]
        keep_box = np.logical_and(lon_day < lon_box[1], lat_day >= lat_box[0])
        # high-level peak
        keep_high =  np.logical_and(rad_features.z_net_peak < z_max, # m
                                    rad_features.z_net_peak > z_min)
        # combined
        k = np.logical_and(np.logical_and(keep_large,keep_box),keep_high)
        
        # relative humidity    
        rh = data_day.relative_humidity.values[k,:]*100
        # lw cooling
        qradlw = rad_features.qrad_lw_smooth[k,:]
        
        return rh, qradlw

    
    for day, z_min, z_max in zip(days_high_peaks,z_min_all,z_max_all):
        
        date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
        data_day = data_all.sel(launch_time=day)
        rad_features = rad_features_all[day]
        
        rh, qradlw = getProfiles(rad_features, data_day, z_min, z_max)
        
        fig,axs = plt.subplots(ncols=2)
        
        for i_s in range(rh.shape[0]):
            
            axs[0].plot(rh[i_s],z,c='k',linewidth=0.3,alpha=0.4)
            axs[1].plot(qradlw[i_s],z,c='k',linewidth=0.3,alpha=0.4)
            
        axs[0].set_xlabel('Relative humidity (%)')
        axs[1].set_xlabel('LW cooling (K/day)')
        axs[0].set_ylabel('z (km)')
        
        heights_label = ('btw%1.1fand%1.1fkm'%(z_min/1e3,z_max/1e3)).replace('.','p')
        
        plt.savefig(os.path.join(figdir,'profiles_high_level_peaks_%s_%s.pdf'%(heights_label,day)),bbox_inches='tight')


#%% All on same graph?


    days_high_peaks = '20200213', '20200213', '20200211', '20200209', '20200209', '20200128'
    z_min_all = 5000, 4000, 4000, 3500, 5500, 4000
    z_max_all = 9000, 5000, 6000, 5500, 8500, 6000
    
    
    def getProfiles(rad_features, data_day, z_min, z_max):
        
        #- Mask
        # |qrad| > 5 K/day
        qrad_peak = np.absolute(rad_features.qrad_lw_peak)
        keep_large = qrad_peak > 5 # K/day
        # in box
        lon_day = data_day.longitude[:,50]
        lat_day = data_day.latitude[:,50]
        keep_box = np.logical_and(lon_day < lon_box[1], lat_day >= lat_box[0])
        # high-level peak
        keep_high =  np.logical_and(rad_features.z_net_peak < z_max, # m
                                    rad_features.z_net_peak > z_min)
        # combined
        k = np.logical_and(np.logical_and(keep_large,keep_box),keep_high)
        
        # relative humidity    
        rh = data_day.relative_humidity.values[k,:]*100
        # lw cooling
        qradlw = rad_features.qrad_lw_smooth[k,:]
        
        return rh, qradlw

    
    fig,axs = plt.subplots(ncols=2,figsize=(7,4.5))
    plt.rc('legend',fontsize=7)
    plt.rc('legend',labelspacing=0.07)

    for day, z_min, z_max in zip(days_high_peaks,z_min_all,z_max_all):
        
        date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
        data_day = data_all.sel(launch_time=day)
        rad_features = rad_features_all[day]
        
        heights_label = r'%s; %1.1f $< z_p <$%1.1f km'%(day, z_min/1e3,z_max/1e3)
        rh, qradlw = getProfiles(rad_features, data_day, z_min, z_max)
        
        # compute interquartile range and median of all profiles types
        rh_Q1, rh_med, rh_Q3 = np.nanpercentile(rh,25,axis=0), np.nanpercentile(rh,50,axis=0), np.nanpercentile(rh,75,axis=0)
        qradlw_Q1, qradlw_med, qradlw_Q3 = np.nanpercentile(qradlw,25,axis=0), np.nanpercentile(qradlw,50,axis=0), np.nanpercentile(qradlw,75,axis=0)
        
        axs[0].plot(rh_med,z,linewidth=1,alpha=1,label=heights_label)
        axs[0].fill_betweenx(z,rh_Q1,rh_Q3,alpha=0.1)
        axs[1].plot(qradlw_med,z,linewidth=1,alpha=1)
        axs[1].fill_betweenx(z,qradlw_Q1,qradlw_Q3,alpha=0.1)
        
    axs[0].set_xlabel('Relative humidity (%)')
    axs[1].set_xlabel('LW cooling (K/day)')
    axs[0].set_ylabel('z (km)')
    axs[0].legend(loc='upper right')
    # axs[0].legend(labelspacing = 0.5)
    
    plt.savefig(os.path.join(figdir,'profiles_high_level_peaks.pdf'),bbox_inches='tight')
    