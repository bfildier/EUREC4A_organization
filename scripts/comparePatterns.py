#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 16:05:26 2021
@author: bfildier

Draw radiative cooling peaks for all days and group by pattern.
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
from scipy.stats import gaussian_kde

# geodesic distances and displacements
import geopy
import geopy.distance
# map display
import cartopy.crs as ccrs

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
scriptsubdir = 'compare_patterns'

# Load own module
projectname = 'EUREC4A_organization'
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
    os.makedirs(os.path.join(figdir,scriptsubdir),exist_ok=True)

if __name__ == "__main__":
    
    # arguments
    parser = argparse.ArgumentParser(description="Compute and store features from radiative profile data")
    parser.add_argument('--overwrite',type=bool,nargs='?',default=False)

    # output directory    
    defineSimDirectories()
    
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

    # full box
    lat_box_full = 6,15.5
    lon_box_full = -60,-49

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

    ##-- compute radiative peaks from conditional distribution of QRADLW
    
    ##-- compute radiative features
    print("- compute radiative features") ###---- already computed!
    
    # cond_varid = 'QRADLW'
    # ref_varid = 'PW'
    z_peak_all = {}
    
    for day in days:
        
        print(day)
        
        # data
        QRAD_on_PW = cond_dist_all['QRADLW'][day].cond_mean.T
        QV_on_PW = cond_dist_all['QV'][day].cond_mean.T
        T_on_PW = cond_dist_all['T'][day].cond_mean.T
        P_on_PW = cond_dist_all['P'][day].cond_mean.T
    
        # Initialize
        f = Features(pres,z)
        # Find peaks in net Q_rad
        f.computePeaks(QRAD_on_PW,which='lw')
        # Compute PW
        f.computePW(QV_on_PW,T_on_PW,P_on_PW,z)
        # Compute water path above z
        f.computeWPaboveZ(QV_on_PW,pres,z_axis=dim_z)
        
        # store
        z_peak_all[day] = f

#%%
    ##-- Draw peak heights on same plot -- explore vizualisations 
    
    #--- circle size: uniform
    
    fig,ax = plt.subplots(figsize=(5,4))
    
    for day,pat in zip(days,name_pattern):
        
        pw = cond_dist_all['QRADLW'][day].on.percentiles
        z_peak = z_peak_all[day].z_lw_peak
        # s = 5*ref_dist_all[day].density*np.diff(ref_dist_all[day].bins)*ref_dist_all[day].size
        s = 25
        c = col_pattern[pat]
        print(day,c)
        
        ax.scatter(pw,z_peak,s=s,color=c,edgecolor='',alpha=0.6)
        
        ax.set_xlabel('PW (mm)')
        ax.set_ylabel(r'$z_{peak}$ (km)')
    
    plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_peak_heights_on_PW_all_days.pdf'),bbox_inches='tight')
    
    #--- circle size: sample size
    
    fig,ax = plt.subplots(figsize=(5,4))
    
    for day in days:
        
        pw = cond_dist_all['QRADLW'][day].on.percentiles
        z_peak = z_peak_all[day].z_lw_peak
        s = 5*ref_dist_all[day].density*np.diff(ref_dist_all[day].bins)*ref_dist_all[day].size
        
        ax.scatter(pw,z_peak,s=s,edgecolor='',alpha=0.6)
        
        ax.set_xlabel('PW (mm)')
        ax.set_ylabel(r'$z_{peak}$ (km)')
    
    plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_peak_heights_on_PW_sizeSample_all_days.pdf'),bbox_inches='tight')
    
    #--- circle size: peak magnitude
    
    fig,ax = plt.subplots(figsize=(5,4))
    
    for day in days:
        
        pw = cond_dist_all['QRADLW'][day].on.percentiles
        z_peak = z_peak_all[day].z_lw_peak
        s = 5*np.absolute(z_peak_all[day].qrad_lw_peak)
        
        ax.scatter(pw,z_peak,s=s,edgecolor='',alpha=0.6)
        
        ax.set_xlabel('PW (mm)')
        ax.set_ylabel(r'$z_{peak}$ (km)')
    
    plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_peak_heights_on_PW_sizeQRADLW_all_days.pdf'),bbox_inches='tight')
    
    #--- circle size: peak magnitude, all profiles
    
    fig,ax = plt.subplots(figsize=(5,4))
    
    for day in days:
        
        pw = rad_features_all[day].pw
        z_peak = rad_features_all[day].z_lw_peak
        s = 2*np.absolute(rad_features_all[day].qrad_lw_peak)
        
        ax.scatter(pw,z_peak,s=s,edgecolor='',alpha=0.2)
        
        ax.set_xlabel('PW (mm)')
        ax.set_ylabel(r'$z_{peak}$ (km)')
    
    plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_all_peak_heights_on_PW_sizeQRADLW_all_days.pdf'),bbox_inches='tight')
    
    #--- circle size: peak magnitude, all profiles, color pattern
    
    fig,ax = plt.subplots(figsize=(5,4),dpi=200)
    
    qrad_min = np.min([np.nanmin(np.absolute(rad_features_all[day].qrad_lw_peak)) for day in days])
    qrad_max = np.max([np.nanmax(np.absolute(rad_features_all[day].qrad_lw_peak)) for day in days])
    
    for day,pat in zip(days,name_pattern):
        
        pw = rad_features_all[day].pw
        z_peak = rad_features_all[day].z_lw_peak
        qrad_peak = np.absolute(rad_features_all[day].qrad_lw_peak)        
        s = 50*(qrad_peak/qrad_max)**2
        c = col_pattern[pat]
        
        ax.scatter(pw,z_peak,s=s,color=c,edgecolor='',alpha=0.6)
        
        ax.set_xlabel('PW (mm)')
        ax.set_ylabel(r'$z_{peak}$ (km)')
    
    plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_all_peak_heights_on_PW_sizeQRADLW_colPattern_all_days.pdf'),bbox_inches='tight')
    
    
#%%    ### SUMMARY FIGURE
    #--- (powerpoint) circle size: peak magnitude, all profiles for peak above 5 K/day, 
    #--- . color pattern, all days
    #--- . keep points in latlon box of analysis
    
    fig,ax = plt.subplots(figsize=(5,4),dpi=200)
    plt.rcParams["legend.markerscale"] = 0.4
    
    qrad_min = np.min([np.nanmin(np.absolute(rad_features_all[day].qrad_lw_peak)) for day in days])
    qrad_max = np.max([np.nanmax(np.absolute(rad_features_all[day].qrad_lw_peak)) for day in days])
    
    for day,pat in zip(days,name_pattern):
        
        pw = rad_features_all[day].pw
        z_peak = rad_features_all[day].z_lw_peak/1e3 # km
        qrad_peak = np.absolute(rad_features_all[day].qrad_lw_peak)
        keep_large = qrad_peak > 5 # K/day
        lon_day = data_all.sel(launch_time=day).longitude[:,50]
        lat_day = data_all.sel(launch_time=day).latitude[:,50]
        keep_box = np.logical_and(lon_day < lon_box[1], lat_day >= lat_box[0])
        k = np.logical_and(keep_large,keep_box)
        
        s = 50*(qrad_peak/qrad_max)**2
        c = col_pattern[pat]
        
        ax.scatter(pw[k],z_peak[k],s=s[k],color=c,edgecolor='',alpha=0.6)
        
        ax.set_xlabel('PW (mm)')
        ax.set_ylabel(r'$z_{peak}$ (km)')
        ax.set_ylim((-0.3,9.6))

    # # legend QRAD
    # for d in 5,10,15:
    #     setattr(thismodule,"h_%d"%d,mlines.Line2D([], [], color='silver',alpha=0.6,
    #                                               marker='o', linestyle='None',linewidth=0,
    #                   markersize=9*(d/qrad_max)**2, label=r'$\vert Q_{rad}\vert >%d K/day$'%d))
    # leg1 = ax.legend(loc='upper left',handles=[h_5,h_10,h_15],fontsize=7)
    # plt.gca().add_artist(leg1)
    

    ## Fully-manual legend QRAD
    rect = mpatches.Rectangle((0.015,0.82), width=0.3, height=0.16,edgecolor='grey', facecolor="none",linewidth=0.5,alpha=0.5, transform=ax.transAxes)
    ax.add_patch(rect)
    for qp,y in zip([5,10,15],[0.94,0.89,0.84]):
        s = 50*(qp/qrad_max)**2
        circle = mlines.Line2D([0], [0], marker='o', color='w',
                        markerfacecolor='r', markersize=s)
        ax.scatter(0.05,y+0.01,s=s,c='k',edgecolor='',transform=ax.transAxes)
        ax.text(0.1,y,s=r'$\vert Q_{rad}\vert >%d$ K/day'%qp,fontsize=7,transform=ax.transAxes)
        
    # legend pattern
    for pat in col_pattern.keys():
        print(pat)
        lab = pat 
        if pat == '':
            lab = 'unknown'
        setattr(thismodule,"h_%s"%pat,mpatches.Patch(color=col_pattern[pat],linewidth=0,alpha=0.6,label=lab))
    ax.legend(loc='lower center',handles=[h_Fish,h_Flower,h_Gravel,h_Sugar,h_],ncol=5,fontsize=6)
    
    plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_allAbove5Kday_peak_heights_on_PW_sizeQRADLW_colPattern_all_days.pdf'),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_allAbove5Kday_peak_heights_on_PW_sizeQRADLW_colPattern_all_days.png'),bbox_inches='tight')
    
    
    
    ### SUMMARY FIGURE
    #--- (powerpoint) circle size: peak magnitude, all profiles for peak above 5 K/day, 
    #--- . color pattern, grey for low confidence, more transparency for medium confidence
    #--- . keep points in latlon box of analysis
    
    fig,ax = plt.subplots(figsize=(5,4),dpi=200)
    
    qrad_min = np.min([np.nanmin(np.absolute(rad_features_all[day].qrad_lw_peak)) for day in days])
    qrad_max = np.max([np.nanmax(np.absolute(rad_features_all[day].qrad_lw_peak)) for day in days])
    
    for day,pat,conf in zip(days,name_pattern,confidence_pattern):
        
        pw = rad_features_all[day].pw
        z_peak = rad_features_all[day].z_lw_peak/1e3 # km
        qrad_peak = np.absolute(rad_features_all[day].qrad_lw_peak)
        keep_large = qrad_peak > 5 # K/day
        lon_day = data_all.sel(launch_time=day).longitude[:,50]
        lat_day = data_all.sel(launch_time=day).latitude[:,50]
        keep_box = np.logical_and(lon_day < lon_box[1], lat_day >= lat_box[0])
        k = np.logical_and(keep_large,keep_box)
        
        # circle diameter proportional to size of peak
        s = 50*(qrad_peak/qrad_max)**2
        c = col_pattern[pat]
        alpha=0.6
        
        if conf == 'Low':
            c = col_pattern['']
            print('- Low confidence:')
            print(day,pat,c)
        if conf == 'Medium':
            alpha = 0.3
            print('- Medium confidence:')
            print(day,pat,c)
        
        ax.scatter(pw[k],z_peak[k],s=s[k],color=c,edgecolor='',alpha=alpha)
        
        ax.set_xlabel('PW (mm)')
        ax.set_ylabel(r'$z_{peak}$ (km)')
        ax.set_ylim((-0.3,9.6))

    # # legend QRAD
    # for d in 5,10,15:
    #     setattr(thismodule,"h_%d"%d,mlines.Line2D([], [], color='silver',alpha=0.6,
    #                                               marker='o', linestyle='None',linewidth=0,
    #                   markersize=9*(d/qrad_max)**2, label=r'$\vert Q_{rad}\vert >%d K/day$'%d))
    # leg1 = ax.legend(loc='upper left',handles=[h_5,h_10,h_15],fontsize=7)
    # plt.gca().add_artist(leg1)
    
    ## Fully-manual legend QRAD
    rect = mpatches.Rectangle((0.015,0.82), width=0.3, height=0.16,edgecolor='grey', facecolor="none",linewidth=0.5,alpha=0.5, transform=ax.transAxes)
    ax.add_patch(rect)
    for qp,y in zip([5,10,15],[0.94,0.89,0.84]):
        s = 50*(qp/qrad_max)**2
        circle = mlines.Line2D([0], [0], marker='o', color='w',
                        markerfacecolor='r', markersize=s)
        ax.scatter(0.05,y+0.01,s=s,c='k',edgecolor='',transform=ax.transAxes)
        ax.text(0.1,y,s=r'$\vert Q_{rad}\vert >%d$ K/day'%qp,fontsize=7,transform=ax.transAxes)
        

    # legend pattern
    for pat in col_pattern.keys():
        print(pat)
        lab = pat 
        if pat == '':
            lab = 'Low confidence'
        setattr(thismodule,"h_%s"%pat,mpatches.Patch(color=col_pattern[pat],linewidth=0,alpha=0.6,label=lab))
    ax.legend(loc='lower center',handles=[h_Fish,h_Flower,h_Gravel,h_Sugar,h_],ncol=5,fontsize=6)
    
    plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_allAbove5Kday_peak_heights_on_PW_sizeQRADLW_colPattern_alphaConfidence_all_days.pdf'),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_allAbove5Kday_peak_heights_on_PW_sizeQRADLW_colPattern_alphaConfidence_all_days.png'),bbox_inches='tight')
    
    ### SUMMARY FIGURE
    #--- (powerpoint) circle size: peak magnitude, all profiles for peak above 5 K/day, 
    #--- . color -density-, all days
    #--- . only keep points in latlon box of analysis
    
    def scatterDensity(ax,x,y,s,alpha,edgecolor=None):
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        return ax.scatter(x,y,c=z,s=s,alpha=0.4,edgecolor=edgecolor)
    
    
    fig,ax = plt.subplots(figsize=(5,4),dpi=200)
    
    qrad_min = np.min([np.nanmin(np.absolute(rad_features_all[day].qrad_lw_peak)) for day in days])
    qrad_max = np.max([np.nanmax(np.absolute(rad_features_all[day].qrad_lw_peak)) for day in days])
    
    x = []
    y = []
    s = []
    
    for day,pat in zip(days,name_pattern):
        
        pw = rad_features_all[day].pw
        z_peak = rad_features_all[day].z_lw_peak/1e3 # km
        qrad_peak = np.absolute(rad_features_all[day].qrad_lw_peak)
        keep_large = qrad_peak > 5 # K/day
        lon_day = data_all.sel(launch_time=day).longitude[:,50]
        lat_day = data_all.sel(launch_time=day).latitude[:,50]
        keep_box = np.logical_and(lon_day < lon_box[1], lat_day >= lat_box[0])
        k = np.logical_and(keep_large,keep_box)
                
        x.extend(pw[k])
        y.extend(z_peak[k])
        s.extend(50*(qrad_peak[k]/qrad_max)**2)
    
    alpha = 1
    h = scatterDensity(ax,x,y,s,alpha)
    #ax.scatter(pw[k],z_peak[k],s=s[k],color=c,edgecolor='',alpha=0.6)
    
    ax.set_xlabel('PW (mm)')
    ax.set_ylabel(r'$z_{peak}$ (km)')
    ax.set_ylim((-0.3,9.6))

    # # legend QRAD
    # for d in 5,10,15:
    #     setattr(thismodule,"h_%d"%d,mlines.Line2D([], [], color='silver',alpha=0.6,
    #                                               marker='o', linestyle='None',linewidth=0,
    #                   markersize=9*(d/qrad_max)**2, label=r'$\vert Q_{rad}\vert >%d K/day$'%d))
    # leg1 = ax.legend(loc='upper left',handles=[h_5,h_10,h_15],fontsize=7)
    # plt.gca().add_artist(leg1)
    
    ## Fully-manual legend QRAD
    rect = mpatches.Rectangle((0.015,0.82), width=0.3, height=0.16,edgecolor='grey', facecolor="none",linewidth=0.5,alpha=0.5, transform=ax.transAxes)
    ax.add_patch(rect)
    for qp,y in zip([5,10,15],[0.94,0.89,0.84]):
        s = 50*(qp/qrad_max)**2
        circle = mlines.Line2D([0], [0], marker='o', color='w',
                        markerfacecolor='r', markersize=s)
        ax.scatter(0.05,y+0.01,s=s,c='k',edgecolor='',transform=ax.transAxes)
        ax.text(0.1,y,s=r'$\vert Q_{rad}\vert >%d$ K/day'%qp,fontsize=7,transform=ax.transAxes)

    # colorbar density
    axins1 = inset_axes(ax,
                    width="50%",  # width = 70% of parent_bbox width
                    height="2%",  # height : 5%
                    loc='lower right')
    cb = fig.colorbar(h, cax=axins1, orientation="horizontal")
    axins1.xaxis.set_ticks_position("top")
    axins1.tick_params(axis='x', labelsize=9)
    cb.set_label('Density',labelpad=-34)
    
    # # legend pattern
    # for pat in col_pattern.keys():
    #     print(pat)
    #     lab = pat
    #     if pat == '':
    #         lab = 'unknown'
    #     setattr(thismodule,"h_%s"%pat,mpatches.Patch(color=col_pattern[pat],linewidth=0,alpha=0.6,label=lab))
    # ax.legend(loc='lower center',handles=[h_Fish,h_Flower,h_Gravel,h_Sugar,h_],ncol=5,fontsize=5)
    
    plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_allAbove5Kday_peak_heights_on_PW_sizeQRADLW_colDensity_all_days.pdf'),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_allAbove5Kday_peak_heights_on_PW_sizeQRADLW_colDensity_all_days.png'),bbox_inches='tight')


    #--- circle size: peak magnitude, all profiles for peak above 5 K/day
    #--- . color pattern, separate each day
    #--- . only keep points in latlon box of analysis
    
    qrad_min = np.min([np.nanmin(np.absolute(rad_features_all[day].qrad_lw_peak)) for day in days])
    qrad_max = np.max([np.nanmax(np.absolute(rad_features_all[day].qrad_lw_peak)) for day in days])
    
    for day,pat in zip(days,name_pattern):
        
        fig,ax = plt.subplots(figsize=(5,4),dpi=200)
        
        pw = rad_features_all[day].pw
        z_peak = rad_features_all[day].z_lw_peak/1e3 # km
        qrad_peak = np.absolute(rad_features_all[day].qrad_lw_peak)
        keep_large = qrad_peak > 5 # K/day
        lon_day = data_all.sel(launch_time=day).longitude[:,10]
        lat_day = data_all.sel(launch_time=day).latitude[:,10]
        keep_box = np.logical_and(lon_day < lon_box[1], lat_day >= lat_box[0])
        k = np.logical_and(keep_large,keep_box)
        
        s = 50*(qrad_peak/qrad_max)**2
        c = col_pattern[pat]
        
        ax.scatter(pw[k],z_peak[k],s=s[k],color=c,edgecolor='',alpha=0.6)
        
        ax.set_xlabel('PW (mm)')
        ax.set_ylabel(r'$z_{peak}$ (km)')
        ax.set_xlim((17.5,54.5))
        ax.set_ylim((-0.3,9.6))
        
        # # legend QRAD
        # for d in 5,10,15:
        #     setattr(thismodule,"h_%d"%d,mlines.Line2D([], [], color='silver',alpha=0.6,
        #                                               marker='o', linestyle='None',linewidth=0,
        #                   markersize=10*(d/qrad_max)**2, label=r'$\vert Q_{rad}\vert >%d K/day$'%d))
        # leg1 = ax.legend(loc='top left',handles=[h_5,h_10,h_15],fontsize=5)
        # plt.gca().add_artist(leg1)
        # # weird, marker size in legend is different than on figure -- adjust manually
        
        ## Fully-manual legend QRAD
        rect = mpatches.Rectangle((0.015,0.82), width=0.3, height=0.16,edgecolor='grey', facecolor="none",linewidth=0.5,alpha=0.5, transform=ax.transAxes)
        ax.add_patch(rect)
        for qp,y in zip([5,10,15],[0.94,0.89,0.84]):
            s = 50*(qp/qrad_max)**2
            circle = mlines.Line2D([0], [0], marker='o', color='w',
                            markerfacecolor='r', markersize=s)
            ax.scatter(0.05,y+0.01,s=s,c='k',edgecolor='',transform=ax.transAxes)
            ax.text(0.1,y,s=r'$\vert Q_{rad}\vert >%d$ K/day'%qp,fontsize=7,transform=ax.transAxes)
            
        # legend pattern
        for pat in col_pattern.keys():
            print(pat)
            lab = pat 
            if pat == '':
                lab = 'unknown'
            setattr(thismodule,"h_%s"%pat,mpatches.Patch(color=col_pattern[pat],linewidth=0,alpha=0.6,label=lab))
        ax.legend(loc='lower center',handles=[h_Fish,h_Flower,h_Gravel,h_Sugar,h_],ncol=5,fontsize=5)
        
        plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_allAbove5Kday_peak_heights_on_PW_sizeQRADLW_colPattern_%s.pdf'%day),bbox_inches='tight')
        plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_allAbove5Kday_peak_heights_on_PW_sizeQRADLW_colPattern_%s.png'%day),bbox_inches='tight')
        
        
#%%    #--- peak magnitude vs. PW, circle size: peak height, all profiles for peak above 5 K/day, color pattern
    
    fig,ax = plt.subplots(figsize=(5,4),dpi=200)
    
    for day,pat in zip(days,name_pattern):
        
        pw = rad_features_all[day].pw
        qrad_peak = rad_features_all[day].qrad_lw_peak
        z_peak = rad_features_all[day].z_lw_peak/1e3 # km
        k = keep_large = np.absolute(qrad_peak) > 5 # K/day
        
        s = 60*((8-z_peak)/8)**2
        c = col_pattern[pat]
        
        ax.scatter(pw[k],qrad_peak[k],s=s[k],color=c,edgecolor='',alpha=0.6)
        
        ax.set_xlabel('PW (mm)')
        ax.set_ylabel(r'peak $Q_{rad}^{LW}$ (K/day)')
        # ax.set_ylim((-0.3,9.6))

    # # legend QRAD
    # for z_leg in 2,4,6:
    #     setattr(thismodule,"h_%d"%z_leg,mlines.Line2D([], [], color='silver',alpha=0.6,
    #                                               marker='o', linestyle='None',linewidth=0,
    #                   markersize=8*((8-z_leg)/8)**2, label=r'$z_{peak}<%d km$'%z_leg))
    # leg1 = ax.legend(loc='top left',handles=[h_2,h_4,h_6],fontsize=5)
    # plt.gca().add_artist(leg1)
    
    ## Fully-manual legend QRAD
    rect = mpatches.Rectangle((0.015,0.015), width=0.3, height=0.16,edgecolor='grey', facecolor="none",linewidth=0.5,alpha=0.5, transform=ax.transAxes)
    ax.add_patch(rect)
    for zp,y in zip([2,4,6],[0.135,0.085,0.035]):
        s = 60*((8-zp)/8)**2
        circle = mlines.Line2D([0], [0], marker='o', color='w',
                        markerfacecolor='r', markersize=s)
        ax.scatter(0.05,y+0.01,s=s,c='k',edgecolor='',transform=ax.transAxes)
        ax.text(0.1,y,s=r'$z_{peak} <%d$ K/day'%zp,fontsize=7,transform=ax.transAxes)
            
    # legend pattern
    for pat in col_pattern.keys():
        print(pat)
        lab = pat 
        if pat == '':
            lab = 'unknown'
        setattr(thismodule,"h_%s"%pat,mpatches.Patch(color=col_pattern[pat],linewidth=0,alpha=0.6,label=lab))
    ax.legend(loc='lower right',handles=[h_Fish,h_Flower,h_Gravel,h_Sugar,h_],ncol=5,fontsize=5)
    
    plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_allAbove5Kday_peak_size_on_PW_sizeZpeak_colPattern_all_days.pdf'),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_allAbove5Kday_peak_size_on_PW_sizeZpeak_colPattern_all_days.png'),bbox_inches='tight')

    #--- peak magnitude vs. PW, circle size: peak height, 
    #--- all profiles for peak above 5 K/day
    #--- color pattern, grey for low confidence, more transparency for medium confidence
    
    fig,ax = plt.subplots(figsize=(5,4),dpi=200)
    
    for day,pat,conf in zip(days,name_pattern,confidence_pattern):
        
        pw = rad_features_all[day].pw
        qrad_peak = rad_features_all[day].qrad_lw_peak
        z_peak = rad_features_all[day].z_lw_peak/1e3 # km
        k = keep_large = np.absolute(qrad_peak) > 5 # K/day
        
        s = 60*((8-z_peak)/8)**2
        c = col_pattern[pat]
        if conf == 'Low':
            c = col_pattern['']
        
        ax.scatter(pw[k],qrad_peak[k],s=s[k],color=c,edgecolor='',alpha=0.6)
        
        ax.set_xlabel('PW (mm)')
        ax.set_ylabel(r'peak $Q_{rad}^{LW}$ (K/day)')
        # ax.set_ylim((-0.3,9.6))

    # # legend QRAD
    # for z_leg in 2,4,6:
    #     setattr(thismodule,"h_%d"%z_leg,mlines.Line2D([], [], color='silver',alpha=0.6,
    #                                               marker='o', linestyle='None',linewidth=0,
    #                   markersize=8*((8-z_leg)/8)**2, label=r'$z_{peak}<%d km$'%z_leg))
    # leg1 = ax.legend(loc='top left',handles=[h_2,h_4,h_6],fontsize=5)
    # plt.gca().add_artist(leg1)
    
    ## Fully-manual legend QRAD
    rect = mpatches.Rectangle((0.015,0.015), width=0.3, height=0.16,edgecolor='grey', facecolor="none",linewidth=0.5,alpha=0.5, transform=ax.transAxes)
    ax.add_patch(rect)
    for zp,y in zip([2,4,6],[0.135,0.085,0.035]):
        s = 60*((8-zp)/8)**2
        circle = mlines.Line2D([0], [0], marker='o', color='w',
                        markerfacecolor='r', markersize=s)
        ax.scatter(0.05,y+0.01,s=s,c='k',edgecolor='',transform=ax.transAxes)
        ax.text(0.1,y,s=r'$z_{peak} <%d$ K/day'%zp,fontsize=7,transform=ax.transAxes)
        
    # legend pattern
    for pat in col_pattern.keys():
        print(pat)
        lab = pat 
        if pat == '':
            lab = 'Low condidence'
        setattr(thismodule,"h_%s"%pat,mpatches.Patch(color=col_pattern[pat],linewidth=0,alpha=0.6,label=lab))
    ax.legend(loc='lower right',handles=[h_Fish,h_Flower,h_Gravel,h_Sugar,h_],ncol=5,fontsize=5)
    
    plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_allAbove5Kday_peak_size_on_PW_sizeZpeak_colPattern_alphaConfidence_all_days.pdf'),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_allAbove5Kday_peak_size_on_PW_sizeZpeak_colPattern_alphaConfidence_all_days.png'),bbox_inches='tight')


#%%    #--- QRADLW profiles all, all profiles for peak above 5 K/day, color pattern
    
    fig,ax = plt.subplots(figsize=(5,4),dpi=200)
    
    for day,pat in zip(days,name_pattern):
        
        pw = rad_features_all[day].pw
        qrad_smooth = rad_features_all[day].qrad_lw_smooth
        qrad_peak = rad_features_all[day].qrad_lw_peak
        z_peak = rad_features_all[day].z_lw_peak/1e3 # km
        k = keep_large = np.absolute(qrad_peak) > 5 # K/day
        
        s = 60*((8-z_peak)/8)**2
        c = col_pattern[pat]
        
        for i_p in range(np.sum(k)):
            ax.plot(qrad_smooth[k][i_p],z,color=c,linewidth=0.1,alpha=0.2)
        
        # ax.set_xlabel('PW (mm)')
        # ax.set_ylabel(r'peak $Q_{rad}^{LW}$ (K/day)')
        # ax.set_ylim((-0.3,9.6))
    
    # legend pattern
    for pat in col_pattern.keys():
        print(pat)
        lab = pat 
        if pat == '':
            lab = 'unknown'
        setattr(thismodule,"h_%s"%pat,mpatches.Patch(color=col_pattern[pat],linewidth=0,alpha=0.6,label=lab))
    ax.legend(loc='upper left',handles=[h_Fish,h_Flower,h_Gravel,h_Sugar,h_],ncol=3,fontsize=5)
    
    plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_profiles_allAbove5Kday_colPattern_all_days.pdf'),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_profiles_allAbove5Kday_colPattern_all_days.png'),bbox_inches='tight')


    #--- QRADLW profiles all, all profiles for peak above 5 K/day,
    #--- color pattern, grey for low confidence, more transparency for medium confidence
    
    fig,ax = plt.subplots(figsize=(5,4),dpi=200)
    
    for day,pat,conf in zip(days,name_pattern,confidence_pattern):
        
        data_day = data_all.sel(launch_time=day)
        
        pw = rad_features_all[day].pw
        qrad_smooth = rad_features_all[day].qrad_lw_smooth
        qrad_peak = rad_features_all[day].qrad_lw_peak
        z_peak = rad_features_all[day].z_lw_peak/1e3 # km
        # filters
        keep_large = np.absolute(qrad_peak) > 5 # K/day
        keep_box = np.logical_and(np.logical_and(data_day.longitude.values[:,10] > lon_box[0],
                                              data_day.longitude.values[:,10] < lon_box[1]),
                               np.logical_and(data_day.latitude.values[:,10] > lat_box[0],
                                              data_day.latitude.values[:,10] < lat_box[1]))
        k = np.logical_and(keep_large,keep_box)

        s = 60*((8-z_peak)/8)**2
        c = col_pattern[pat]
        if conf == 'Low':
            c = col_pattern['']
        
        for i_p in range(np.sum(k)):
            ax.plot(qrad_smooth[k][i_p],z,color=c,linewidth=0.1,alpha=0.2)
        
        # ax.set_xlabel('PW (mm)')
        # ax.set_ylabel(r'peak $Q_{rad}^{LW}$ (K/day)')
        # ax.set_ylim((-0.3,9.6))

    # # legend QRAD
    # for z_leg in 2,4,6:
    #     setattr(thismodule,"h_%d"%z_leg,mlines.Line2D([], [], color='silver',alpha=0.6,
    #                                               marker='o', linestyle='None',linewidth=0,
    #                   markersize=8*((8-z_leg)/8)**2, label=r'$z_{peak}<%d km$'%z_leg))
    # leg1 = ax.legend(loc='top left',handles=[h_2,h_4,h_6],fontsize=5)
    # plt.gca().add_artist(leg1)
    
    # legend pattern
    for pat in col_pattern.keys():
        print(pat)
        lab = pat 
        if pat == '':
            lab = 'Low confidence'
        setattr(thismodule,"h_%s"%pat,mpatches.Patch(color=col_pattern[pat],linewidth=0,alpha=0.6,label=lab))
    ax.legend(loc='upper left',handles=[h_Fish,h_Flower,h_Gravel,h_Sugar,h_],ncol=3,fontsize=5)
    
    plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_profiles_allAbove5Kday_colPattern_alphaConfidence_all_days.pdf'),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_profiles_allAbove5Kday_colPattern_alphaConfidence_all_days.png'),bbox_inches='tight')



#%%    ##-- Draw peak heights and QRADLW|PW on separate plots

    cmap=plt.cm.seismic    

    for day in days:
    
        vmin=-10
        vmax=10
        
        fig,ax = plt.subplots(figsize=(6,5))
        
        #- QRADLW on PW background
        array = cond_dist_all['QRADLW'][day].cond_mean
        h = ax.imshow(array,
                  aspect=5,
                  origin='lower',
                  extent=[ref_var_min,ref_var_max,z[0],z[-1]],
                  cmap=cmap,
                  vmin=vmin,
                  vmax=vmax)
        
        #- peak height
        pw = cond_dist_all['QRADLW'][day].on.percentiles
        z_peak = z_peak_all[day].z_lw_peak
        
        ax.scatter(pw,z_peak,c='y',s=15,edgecolor='none')
        
        ax.set_xlabel('PW (mm)')
        ax.set_ylabel('z (km)')
        ax.set_title('QRADLW on PW - %s'%(day))
        
        # colorbar
        # plt.colorbar(h)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                     ax=ax,shrink=0.85,pad=0.06)
        cb.set_label(r'QRADLW (K/day)')
        
        plt.savefig(os.path.join(figdir,scriptsubdir,'QRADLW_on_PW_z_with_peaks_%s.pdf'%(day)),bbox_inches='tight')


#%%     Draw maps with p_peak (hPa) as colored scatter plot

    lon_range = lon_box_full
    lat_range = lat_box_full
    aspect_ratio = (np.diff(lon_range)/np.diff(lat_range))[0]
    
    # data
    data_all = radprf.where(radprf.z_min<=50,drop=True)
    
    # colors
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1000)
    cmap = plt.cm.terrain_r

    fig,ax = plt.subplots(subplot_kw={'projection':ccrs.PlateCarree()},figsize=(7*aspect_ratio,7))

    ax.coastlines(resolution='50m')
    ax.set_extent([*lon_range,*lat_range])
    gl = ax.gridlines(color='Grey',draw_labels=True)
    # gl.xlabel_style = {'size': 16}
    # gl.ylabel_style = {'size': 16}
    
    # # show background
    # ax.imshow(im_hour,extent=[*lon_box_goes,*lat_box_goes],origin='upper')

    for day in days:
        
        data_day = data_all.sel(launch_time=day)
        lon_day = data_day.longitude.values[:,100]
        lat_day = data_day.latitude.values[:,100]
        col_day = cmap(norm(rad_features_all[day].pres_lw_peak)) 
        # show sondes
        
        m = ax.scatter(lon_day,lat_day,marker='o',c=col_day,alpha=0.4,s=30,label='sondes')
    
    # box of analysis
    dlon = lon_box[1]-lon_box[0]
    dlat = lat_box[1]-lat_box[0]
    box = mpatches.Rectangle((lon_box[0],lat_box[0]), dlon, dlat,
                     facecolor='none',edgecolor='k', alpha=1)
    ax.add_patch(box)
    
    # white box behind colorbar
    cbbox = inset_axes(ax, '15%', '62%', loc = 'lower right')
    cbbox.set_facecolor([1,1,1,0.7])
    [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
    cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)

    # colorbar
    axins1 = inset_axes(cbbox,
                width="15%",  # width = 2% of parent_bbox width
                height="95%",  # height : 50%
                loc='lower right')
    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                 cax=axins1, orientation="vertical")
    cb.ax.invert_yaxis()
    axins1.yaxis.set_ticks_position("left")
    cb.set_label('LW peak height (hPa)',labelpad=-46)
    
    plt.rcParams['axes.titlepad'] = +26
    ax.set_title('LW cooling peak pressure')
    
    plt.savefig(os.path.join(figdir,scriptsubdir,'map_QRADLW_p_peak_all.pdf'),bbox_inches='tight')


#%%     Draw maps of z_peak (km) as colored scatter plot

    lon_range = lon_box_full
    lat_range = lat_box_full
    aspect_ratio = (np.diff(lon_range)/np.diff(lat_range))[0]
    
    # data
    data_all = radprf.where(radprf.z_min<=50,drop=True)
    
    # colors
    norm = matplotlib.colors.Normalize(vmin=0, vmax=8)
    cmap = plt.cm.terrain

    fig,ax = plt.subplots(subplot_kw={'projection':ccrs.PlateCarree()},figsize=(7*aspect_ratio,7))

    ax.coastlines(resolution='50m')
    ax.set_extent([*lon_range,*lat_range])
    gl = ax.gridlines(color='Grey',draw_labels=True)
    # gl.xlabel_style = {'size': 16}
    # gl.ylabel_style = {'size': 16}
    
    # # show background
    # ax.imshow(im_hour,extent=[*lon_box_goes,*lat_box_goes],origin='upper')

    for day in days:
        
        data_day = data_all.sel(launch_time=day)
        lon_day = data_day.longitude.values[:,100]
        lat_day = data_day.latitude.values[:,100]
        col_day = cmap(norm(rad_features_all[day].z_lw_peak/1e3)) 
        # show sondes
        
        m = ax.scatter(lon_day,lat_day,marker='o',c=col_day,alpha=0.4,s=30,label='sondes')
        
    # box of analysis
    dlon = lon_box[1]-lon_box[0]
    dlat = lat_box[1]-lat_box[0]
    box = mpatches.Rectangle((lon_box[0],lat_box[0]), dlon, dlat,
                     facecolor='none',edgecolor='k', alpha=1)
    ax.add_patch(box)
        
    # white box behind colorbar
    cbbox = inset_axes(ax, '15%', '62%', loc = 'lower right')
    cbbox.set_facecolor([1,1,1,0.7])
    [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
    cbbox.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)

    # colorbar
    axins1 = inset_axes(cbbox,
                width="15%",  # width = 2% of parent_bbox width
                height="95%",  # height : 50%
                loc='lower right')
    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                 cax=axins1, orientation="vertical")
    axins1.yaxis.set_ticks_position("left")
    cb.set_label('LW peak height (km)',labelpad=-46)
    
    plt.rcParams['axes.titlepad'] = +26
    ax.set_title('LW cooling peak height')
    
    plt.savefig(os.path.join(figdir,scriptsubdir,'map_QRADLW_z_peak_all.pdf'),bbox_inches='tight')