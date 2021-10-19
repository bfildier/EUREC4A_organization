#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 14:09:01 2021

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
    parser = argparse.ArgumentParser(description="Draw radiative profile data composited on PW")
    parser.add_argument('--day', default='20200126',help="day of analysis")
    parser.add_argument('--overwrite',type=bool,nargs='?',default=False)
    
    args = parser.parse_args()
    day = args.day
    
    # day = '20200126'
    date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
    
    # varids
    ref_varid = 'PW'
    cond_varids = 'QRAD','QRADSW','QRADLW','QV','UNORM'
    
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
    
    
#%%    ###--- Figures ---###
    
    ##-- Conditional QRAD (net, LW, SW) on PW
    
    cmap=plt.cm.seismic

    vmin=-10
    vmax=10

    for cond_varid in 'QRAD','QRADLW','QRADSW':
        
        fig,ax = plt.subplots(figsize=(6,5))
        
        array = getattr(thismodule,'cdist_%s_on_%s'%(cond_varid,ref_varid)).cond_mean
        h = ax.imshow(array,
                  aspect=5,
                  origin='lower',
                  extent=[ref_var_min,ref_var_max,f.z[0]/1000,f.z[-1]/1000],
                  cmap=cmap,
                  vmin=vmin,
                  vmax=vmax)
        
        ax.set_xlabel('PW (mm)')
        ax.set_ylabel('z (km)')
        ax.set_title('%s on %s'%(cond_varid,date.strftime("%Y-%m-%d")))
        
        # colorbar
        # plt.colorbar(h)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                     ax=ax,shrink=0.9,pad=0.06)
        cb.set_label(r'%s (K/day)'%cond_varid)
        
        plt.savefig(os.path.join(figdir,day,'%s_on_%s_z_%s.pdf'%(cond_varid,ref_varid,date.strftime("%Y%m%d"))),bbox_inches='tight')
        
    
    
    ##-- Conditional distribution QRAD_LW on PW with distribution of PW and mean QRAD_LW on sides
    
    fig = plt.figure(constrained_layout=True,figsize=(7,5))
    fig.suptitle('%s'%(date.strftime("%Y-%m-%d")),y=1.05)

    widths = [1, 4]
    heights = [4, 1]
    spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths,
                              height_ratios=heights)
    # spec.update(wspace=0.025,hspace=0.05)
    
    #- mean QRAD_LW
    ax = fig.add_subplot(spec[0, 0])
    QRADLW_mean = np.nanmean(cdist_QRADLW_on_PW.cond_mean,axis=1)
    z = data_day.alt/1e3
    ax.plot(QRADLW_mean,z,'k')
    ax.set_xlim((-6,0))
    ax.set_xlabel(r'$\left<QRADLW\right>$ (K/day)')
    ax.set_ylabel('z (km)')
                
    #- conditional QRAD_LW on PW
    ax = fig.add_subplot(spec[0, 1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    array = cdist_QRADLW_on_PW.cond_mean
    h = ax.imshow(array,
              aspect=5,
              origin='lower',
              extent=[ref_var_min,ref_var_max,f.z[0]/1000,f.z[-1]/1000],
              cmap=cmap,
              vmin=vmin,
              vmax=vmax)
        
    # # ax.set_xlabel('PW (mm)')
    # # ax.set_ylabel('z (km)')
    
    # colorbar
    # plt.colorbar(h)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                  ax=ax,shrink=0.9,pad=0.06)
    cb.set_label(r'QRADLW (K/day)')
    
    #- distribution of PW
    ax = fig.add_subplot(spec[1, 1])
    dPW = np.diff(dist_PW.bins)[0]
    ax.plot(dist_PW.percentiles,dist_PW.density*dist_PW.size*dPW,'k')
    ax.set_xlabel('PW (mm)')
    ax.set_xlim((dist_PW.bins[0],dist_PW.bins[-1]))
    ax.set_ylabel('count')
    
    # plt.subplots_adjust(wspace=0.1,hspace=0.1)
    
    # save
    plt.savefig(os.path.join(figdir,day,'QRADLW_on_%s_z__with_marginals_%s.pdf'%(ref_varid,date.strftime("%Y%m%d"))),bbox_inches='tight')
    
    
    ##-- Diurnal cycle of QRAD (net,LW,SW) at the peak of QRAD_LW
    
    print("Diurnal cycle of Qrad_NET/LW/SW @ Qrad_LW peak; Color PW")

    #-- separately

    rad_labels = {'net':'',
                  'sw':'SW',
                  'lw':'LW'}
    
    for rad_range in rad_labels.keys():
        
        PW_lims = [30,70] # km
        
        # colors
        var_col = f.pw
        norm = matplotlib.colors.Normalize(vmin=PW_lims[0],vmax=PW_lims[1])
        # cmap = plt.cm.nipy_spectral
        cmap = plt.cm.gnuplot2_r
        cols = cmap(norm(var_col),bytes=True) 
        
        t = np.array([(pytz.utc.localize(dt.strptime(str(t)[:19],"%Y-%m-%dT%H:%M:%S")) - date).seconds/3600 for t in f.launch_time])
        
        ##-- plot
        fig,ax = plt.subplots(figsize=(6,4.5))
        
        for i_lt in range(f.launch_time.size):
        
            x = t[i_lt]
            y = getattr(f,'qrad_%s_smooth'%rad_range)[i_lt,f.i_lw_peak[i_lt]]
        
            ax.scatter(x,y,c=[cols[i_lt][:3]/255])
            
        ax.set_xlabel(r'Time (hours)')
        ax.set_ylabel(r'$Q_{rad}^{%s} ( z_{peak}(Q_{rad}^{LW}))$  (K/day)'%rad_labels[rad_range])
        
        #- colorbar
        cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                     ax=ax,shrink=0.95,pad=0.09)
        cb.set_label('PW (mm)')
        
        # Save
        plt.savefig(os.path.join(figdir,day,'Qrad%s_at_QradLW_peak_vs_time_colorPW_%s.pdf'%(rad_labels[rad_range],date.strftime("%Y%m%d"))),bbox_inches='tight')
        
        
    #-- simultaneously
    
    PW_lims = [30,70] # km
    
    # colors
    var_col = f.pw
    norm = matplotlib.colors.Normalize(vmin=PW_lims[0],vmax=PW_lims[1])
    # cmap = plt.cm.nipy_spectral
    cmap = plt.cm.gnuplot2_r
    cols = cmap(norm(var_col),bytes=True) 
    
    
    t = np.array([(pytz.utc.localize(dt.strptime(str(t)[:19],"%Y-%m-%dT%H:%M:%S")) - date).seconds/3600 for t in f.launch_time])
    
    # plot
    fig,axs = plt.subplots(ncols=3,figsize=(18,4.5))
    
    for i_p in range(3):
        
        rad_range = list(rad_labels.keys())[i_p]
        ax = axs[i_p]
        
        for i_lt in range(f.pw.size):
        
            x = t[i_lt]
            y = getattr(f,'qrad_%s_smooth'%rad_range)[i_lt,f.i_lw_peak[i_lt]]
        
            ax.scatter(x,y,c=[cols[i_lt][:3]/255])
            
        ax.set_xlabel(r'Time (hours)')
        ax.set_ylabel(r'$Q_{rad}^{%s} ( z_{peak}(Q_{rad}^{LW}))$  (K/day)'%rad_labels[rad_range])
        ax.set_title(rad_labels[rad_range])
        
        #- colorbar
        cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                     ax=ax,shrink=0.95,pad=0.09)
        cb.set_label('PW (mm)')
        
    # Save
    plt.savefig(os.path.join(figdir,day,'Qrad_netLWSW_at_QradLW_peak_vs_time_colorPW_%s.pdf'%(date.strftime("%Y%m%d"))),bbox_inches='tight')
        
    
    ##-- z_peak(QRADLW) vs PW, color QRADLW at peak
    
    rad_range = 'lw'
    rad_labels = {'net':'',
                  'sw':'SW',
                  'lw':'LW'}
    
    print("Color by peak magnitude in a PW-Qrad%s_height plane"%rad_labels[rad_range])
    
    # colors
    var_col = getattr(f,'qrad_%s_peak'%rad_range)
    norm = matplotlib.colors.Normalize(vmin=None, vmax=None)
    cmap = plt.cm.nipy_spectral
    cols = cmap(norm(var_col),bytes=True)
    
    
    # filter in lon-lat box
    # getMaskXYT(time_current,times,lon_box,lat_box,data_day,time_mode=None)
    
    ##-- plot
    fig,ax = plt.subplots(figsize=(6,4.5))
    
    for i_lt in range(f.pw.size):
    
        x = f.pw[i_lt]
        y = getattr(getattr(f,'%s_peaks'%rad_range),'z_%s_peak'%rad_range).values[i_lt]/1000
        
        ax.scatter(x,y,c=[cols[i_lt][:3]/255])
        # ax.scatter(f.pw,f.qrad_peak,c=cols)
        
    ax.set_xlabel(r'PW (mm)')
    ax.set_ylabel(r'$z_{peak}(Q_{rad}^{%s})$ (km)'%rad_labels[rad_range])
    
    #- colorbar
    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                 ax=ax,shrink=0.95,pad=0.09)
    cb.set_label(r'$Q_{rad}^{%s}$ (K/day)'%rad_labels[rad_range])
    
    # Save
    plt.savefig(os.path.join(figdir,day,'Qrad%s_height_vs_PW_colorPeakMagnitude_%s.pdf'%(rad_labels[rad_range],date.strftime("%Y%m%d"))),bbox_inches='tight')
    
    
    ##-- Conditional UNORM on PW
    
    
    # compute mean unorm
    ref_bin_width = np.diff(cdist_UNORM_on_PW.on.bins)
    u_norm_mean = np.nansum(cdist_UNORM_on_PW.cond_mean*cdist_UNORM_on_PW.on.density*ref_bin_width,axis=1)
    u_norm_mean_2D = mo.duplicate(u_norm_mean,dims=(1,ref_bin_width.size),ref_axis=0)
    
    cmap=plt.cm.magma

    # vmin=-10
    # vmax=10
    vmin = 0
    vmax = 8

    cond_varid = 'UNORM'
        
    fig,ax = plt.subplots(figsize=(6,5))
    
    array = getattr(thismodule,'cdist_%s_on_%s'%(cond_varid,ref_varid)).cond_mean - u_norm_mean_2D
    h = ax.imshow(array,
              aspect=5,
              origin='lower',
              extent=[ref_var_min,ref_var_max,f.z[0]/1000,f.z[-1]/1000],
              cmap=cmap,
              vmin=vmin,
              vmax=vmax)
    
    ax.set_xlabel('PW (mm)')
    ax.set_ylabel('z (km)')
    ax.set_title('$\delta$%s on %s'%(cond_varid,date.strftime("%Y-%m-%d")))
    
    # colorbar
    # plt.colorbar(h)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                 ax=ax,shrink=0.9,pad=0.06)
    cb.set_label(r"$\delta$%s (m/s)"%cond_varid)
    
    plt.savefig(os.path.join(figdir,day,'%s_anomaly_on_%s_z_%s.pdf'%(cond_varid,ref_varid,date.strftime("%Y%m%d"))),bbox_inches='tight')
    
    
    ##-- Conditional std(UNORM) on PW
    
    
    # # compute mean unorm
    # ref_bin_width = np.diff(cdist_UNORM_on_PW.on.bins)
    # u_norm_mean = np.nansum(cdist_UNORM_on_PW.cond_mean*cdist_UNORM_on_PW.on.density*ref_bin_width,axis=1)
    # u_norm_mean_2D = mo.duplicate(u_norm_mean,dims=(1,ref_bin_width.size),ref_axis=0)
    
    cmap=plt.cm.magma

    # vmin=-10
    # vmax=10
    vmin = 0
    vmax = 8

    cond_varid = 'UNORM'
        
    fig,ax = plt.subplots(figsize=(6,5))
    
    array = np.sqrt(getattr(thismodule,'cdist_%s_on_%s'%(cond_varid,ref_varid)).cond_var)
    h = ax.imshow(array,
              aspect=5,
              origin='lower',
              extent=[ref_var_min,ref_var_max,f.z[0]/1000,f.z[-1]/1000],
              cmap=cmap,
              vmin=vmin,
              vmax=vmax)
    
    ax.set_xlabel('PW (mm)')
    ax.set_ylabel('z (km)')
    ax.set_title('$\sigma$(%s) on %s'%(cond_varid,date.strftime("%Y-%m-%d")))
    
    # colorbar
    # plt.colorbar(h)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                 ax=ax,shrink=0.9,pad=0.06)
    cb.set_label(r"$\sigma$(%s) (m/s)"%cond_varid)
    
    plt.savefig(os.path.join(figdir,day,'%s_std_on_%s_z_%s.pdf'%(cond_varid,ref_varid,date.strftime("%Y%m%d"))),bbox_inches='tight')
    