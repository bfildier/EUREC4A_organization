#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:51:28 2021
@author: bfildier

Draw radiative cooling peaks for all days and group by platform.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import xarray as xr
import sys,os,glob
import argparse
import pickle
from matplotlib import cm
from math import ceil

##-- directories and modules

workdir = os.path.dirname(os.path.realpath(__file__))
repodir = os.path.dirname(workdir)
moduledir = os.path.join(repodir,'functions')
subdirname = 'radiative_features'
resultdir = os.path.join(repodir,'results',subdirname)
figdir = os.path.join(repodir,'figures',subdirname)
inputdir = '/Users/bfildier/Dropbox/Data/EUREC4A/sondes_radiative_profiles/'
scriptsubdir = 'compare_platforms'

# Load own module
projectname = 'EUREC4A_organization'
thismodule = sys.modules[__name__]

## Own modules
sys.path.insert(0,moduledir)
print("Own modules available:", [os.path.splitext(os.path.basename(x))[0]
                                 for x in glob.glob(os.path.join(moduledir,'*.py'))])

from radiativefeatures import *
from conditionalstats import *
from matrixoperators import *

##--- local functions

def defineSimDirectories():
        
    # create output directory if not there
    os.makedirs(os.path.join(figdir,scriptsubdir),exist_ok=True)

if __name__ == "__main__":
    
    # arguments
    parser = argparse.ArgumentParser(description="Compute and store features from radiative profile data")
    parser.add_argument('--overwrite',type=bool,nargs='?',default=False)

    # output directory    
    defineSimDirectories()
    
    days =          '20200122','20200124','20200126','20200128','20200131','20200202','20200205','20200207','20200209','20200211','20200213'
    platforms = 'BCO', 'HALO', 'P3', 'MER', 'MET', 'ATL', 'RHB'

    # col_pattern = {'':'silver',
    #                'Fish':'navy',
    #                'Gravel':'orange',
    #                'Sugar':'seagreen',
    #                'Flower':'firebrick'}
    col_platform = {'BCO':'k',
                    'HALO':'mediumvioletred',
                    'P3':'orangered',
                    'MER':'lightseagreen',
                    'MET':'steelblue',
                    'ATL':'mediumseagreen',
                    'RHB':'olive'}
    m_platform = {'BCO':'o',
                  'HALO':'v',
                  'P3':'v',
                  'MER':'o',
                  'MET':'o',
                  'ATL':'o',
                  'RHB':'o'}
    
    dim_t,dim_z = 0,1
    
    # varids
    ref_varid = 'PW'
    cond_varids = 'QRAD','QRADSW','QRADLW','QV','UNORM','T','P'

    ###--- Load data ---###
    
    # Profiles
    radprf = xr.open_dataset(os.path.join(inputdir,'rad_profiles_CF.nc'))
    # choose profiles for that day that start at bottom
    data_all = radprf.where(radprf.z_min<=50,drop=True)
    
    z = data_all.alt.values/1e3 # km
    pres = np.nanmean(data_all.pressure.data,axis=dim_t)/100 # hPa
    
    rad_features_all = {}
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
    
    
#%% ##--- PLOTS ---##


    #--- scatter z_peak vs. PW
    #--- circle size: peak magnitude, all profiles for peak above 5 K/day
    #--- color platform, separate each day
    
    qrad_min = np.min([np.nanmin(np.absolute(rad_features_all[day].qrad_lw_peak)) for day in days])
    qrad_max = np.max([np.nanmax(np.absolute(rad_features_all[day].qrad_lw_peak)) for day in days])
    
    for day in days:
                
        pw = rad_features_all[day].pw
        z_peak = rad_features_all[day].z_lw_peak/1e3 # km
        qrad_peak = np.absolute(rad_features_all[day].qrad_lw_peak)
        k = keep_large = qrad_peak > 5 # K/day
        
        s = 50*(qrad_peak/qrad_max)**2

        # init
        fig,ax = plt.subplots(figsize=(5,4),dpi=200)

        # select data
        data_day = data_all.sel(launch_time=day)
        platforms_day = data_day.platform.values
        Ns = np.sum(k)
        
        # loop over soundings for that day
        for i_s  in range(Ns):
            
            pla = platforms_day[k][i_s]

            c = col_platform[pla]
            # m = m_platform[pla]
            m = 'o'
            
            ax.scatter(pw[k][i_s],z_peak[k][i_s],s=s[k][i_s],marker=m,color=c,edgecolor='',alpha=0.6)
        
        ax.set_xlabel('PW (mm)')
        ax.set_ylabel(r'$z_{peak}$ (km)')
        ax.set_xlim((28,91))
        ax.set_ylim((-0.3,9.6))
        
        # legend QRAD
        for d in 5,10,15:
            setattr(thismodule,"h_%d"%d,mlines.Line2D([], [], color='silver',alpha=0.6,
                                                      marker='o', linestyle='None',linewidth=0,
                          markersize=10*(d/qrad_max)**2, label=r'$\vert Q_{rad}\vert >%d K/day$'%d))
        leg1 = ax.legend(loc='upper left',handles=[h_5,h_10,h_15],fontsize=5)
        plt.gca().add_artist(leg1)
        # weird, marker size in legend is different than on figure -- adjust manually
            
        # legend pattern
        for pla in platforms:
            print(pla)
            lab = pla 
            if pla == '':
                lab = 'unknown'
            setattr(thismodule,"h_%s"%pla,mlines.Line2D([],[],color=col_platform[pla],marker='o',
                                                         linestyle='None',alpha=0.6,label=lab))
        ax.legend(loc='lower center',handles=[h_BCO,h_HALO,h_P3,h_MER,h_MET,h_ATL,h_RHB],ncol=5,fontsize=5)
        
        plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_allAbove5Kday_peak_heights_on_PW_sizeQRADLW_colPlatform_%s.pdf'%day),bbox_inches='tight')
        plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_allAbove5Kday_peak_heights_on_PW_sizeQRADLW_colPlatform_%s.png'%day),bbox_inches='tight')
        
        
    #--- scatter z_peak vs. PW
    #--- circle size: peak magnitude, all profiles for peak above 5 K/day
    #--- color platform
    
    qrad_min = np.min([np.nanmin(np.absolute(rad_features_all[day].qrad_lw_peak)) for day in days])
    qrad_max = np.max([np.nanmax(np.absolute(rad_features_all[day].qrad_lw_peak)) for day in days])

    # init
    fig,ax = plt.subplots(figsize=(5,4),dpi=200)

    for day in days:
                
        pw = rad_features_all[day].pw
        z_peak = rad_features_all[day].z_lw_peak/1e3 # km
        qrad_peak = np.absolute(rad_features_all[day].qrad_lw_peak)
        k = keep_large = qrad_peak > 5 # K/day
        
        s = 50*(qrad_peak/qrad_max)**2

        # select data
        data_day = data_all.sel(launch_time=day)
        platforms_day = data_day.platform.values
        Ns = np.sum(k)

        # loop over soundings for that day
        for i_s  in range(Ns):
            
            pla = platforms_day[k][i_s]

            c = col_platform[pla]
            # m = m_platform[pla]
            m = 'o'
            
            ax.scatter(pw[k][i_s],z_peak[k][i_s],s=s[k][i_s],marker=m,color=c,edgecolor='',alpha=0.6)
        
    ax.set_xlabel('PW (mm)')
    ax.set_ylabel(r'$z_{peak}$ (km)')
    ax.set_xlim((28,91))
    ax.set_ylim((-0.3,9.6))
    
    # legend QRAD
    for d in 5,10,15:
        setattr(thismodule,"h_%d"%d,mlines.Line2D([], [], color='silver',alpha=0.6,
                                                  marker='o', linestyle='None',linewidth=0,
                      markersize=10*(d/qrad_max)**2, label=r'$\vert Q_{rad}\vert >%d K/day$'%d))
    leg1 = ax.legend(loc='upper left',handles=[h_5,h_10,h_15],fontsize=5)
    plt.gca().add_artist(leg1)
    # weird, marker size in legend is different than on figure -- adjust manually
        
    # legend pattern
    for pla in platforms:
        print(pla)
        lab = pla 
        if pla == '':
            lab = 'unknown'
        setattr(thismodule,"h_%s"%pla,mlines.Line2D([],[],color=col_platform[pla],marker='o',
                                                     linestyle='None',alpha=0.6,label=lab))
    ax.legend(loc='lower center',handles=[h_BCO,h_HALO,h_P3,h_MER,h_MET,h_ATL,h_RHB],ncol=5,fontsize=5)
    
    plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_allAbove5Kday_peak_heights_on_PW_sizeQRADLW_colPlatform.pdf'),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_allAbove5Kday_peak_heights_on_PW_sizeQRADLW_colPlatform.png'),bbox_inches='tight')
    