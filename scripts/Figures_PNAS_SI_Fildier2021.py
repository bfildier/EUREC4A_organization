#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 17:41:22 2022

@author: bfildier
"""


##-- modules

import scipy.io
import sys, os, glob
import numpy as np
import xarray as xr
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import Circle
from PIL import Image
from datetime import datetime as dt
from datetime import timedelta, timezone
import pytz
import matplotlib.image as mpimg
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pickle
from scipy.stats import gaussian_kde
from scipy.stats import linregress
from scipy import optimize
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

##-- directories

# workdir = os.path.dirname(os.path.realpath(__file__))
workdir = '/Users/bfildier/Code/analyses/EUREC4A/EUREC4A_organization/scripts'
repodir = os.path.dirname(workdir)
moduledir = os.path.join(repodir,'functions')
resultdir = os.path.join(repodir,'results','radiative_features')
figdir = os.path.join(repodir,'figures','paper')
inputdir = '/Users/bfildier/Dropbox/Data/EUREC4A/sondes_radiative_profiles/'
radinputdir = os.path.join(repodir,'input')
imagedir = os.path.join(repodir,'figures','snapshots','with_HALO_circle')
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
from thermoConstants import *
from thermoFunctions import *

mo = MatrixOperators()

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
    
    # box GOES images
    lat_box_goes = 10,16
    lon_box_goes = -60,-52
    
    # box of analysis
    lat_box = 11,16
    lon_box = -60,-52
    
    # HALO circle coordinates
    lon_center = -57.717
    lat_center = 13.3
    lon_pt_circle = -57.245
    lat_pt_circle = 14.1903
    r_circle = np.sqrt((lon_pt_circle - lon_center) ** 2 +
                       (lat_pt_circle - lat_center) ** 2)
    
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
    

    # load GOES images
    date = dt(2020,1,26)
    indir_goes_images = '/Users/bfildier/Data/satellite/GOES/images/%s'%date.strftime('%Y_%m_%d')
    image_vis_files = glob.glob(os.path.join(indir_goes_images,'*C02*00.jpg'))
    image_vis_files.sort()
    # in the visible channel
    images_vis = []
    for i in range(len(image_vis_files)):
        images_vis.append(mpimg.imread(image_vis_files[i]))
    

    # load Caroline's data for rad circulation
    c_inputdir = os.path.join(repodir,'input','Caroline')
    c_muller = scipy.io.loadmat(os.path.join(c_inputdir,'Qrad_pwbinnedvariables_ir90_t40_a50_nbins64.mat'))

    # load moist intrusion data
    # with piecewise linear fit and removed intrusions
    rad_file_MI_20200213 = os.path.join(radinputdir,'rad_profiles_moist_intrusions_20200213.nc')
    radprf_MI_20200213 = xr.open_dataset(rad_file_MI_20200213)
    
    # with piecewise linear fit and removed intrusions
    rad_file_MI_20200213lower = os.path.join(radinputdir,'rad_profiles_moist_intrusions_20200213lower.nc')
    radprf_MI_20200213lower = xr.open_dataset(rad_file_MI_20200213lower)

    # with rectangular intrusions
    rad_file_RMI_20200213 = os.path.join(radinputdir,'rad_profiles_rectangular_moist_intrusions.nc')
    radprf_RMI_20200213 = xr.open_dataset(rad_file_RMI_20200213)
    
    # warming
    rad_file_warming = os.path.join(radinputdir,'rad_profiles_stepRH20200126_idealized_warming.nc')
    radprf_warming = xr.open_dataset(rad_file_warming)

    # load info on all moist intrusions
    mi_file = os.path.join(repodir,'results','idealized_calculations','observed_moist_intrusions','moist_intrusions.pickle')
    moist_intrusions = pickle.load(open(mi_file,'rb'))

#%% Initiate figure count

i_fig = 2



#%% --- Figure S3

i_fig = 3

fig,axs = plt.subplots(ncols=3,nrows=4,figsize=(12,11),
                       subplot_kw={'projection':ccrs.PlateCarree()})
fig.subplots_adjust(wspace = 0.05, hspace = 0)


for i_sub in range(11):
    
    ax = axs.flatten()[i_sub]
    day_p = days[i_sub]
    name_p = name_pattern[i_sub]
    conf_p = confidence_pattern[i_sub]
    
    # if day_p == '20200126':
    #     date = dt.strptime(day_p,'%Y%m%d')
    #     time = '1400'
    #     indir_goes_images = '/Users/bfildier/Data/satellite/GOES/images/%s'%date.strftime('%Y_%m_%d')
    #     image_vis_file = os.path.join(indir_goes_images,'C02_GOES_M2_10N-16N-60W-52W_%s_%s.jpg'%(day_p,time))
    #     image = Image.open(image_vis_file)
    # else:
    #     image = Image.open(os.path.join(workdir,'../images/patterns/PNG','GOES16__%s_1400.png'%day_p))
    
    image = Image.open(os.path.join(workdir,'../images/patterns/PNG','GOES16__%s_1400.png'%day_p))
    
    
    # if len(glob.glob(os.path.join(workdir,'../images/patterns/PNG_brighter','GOES16__%s_1400.png'%day_p))) > 0:
    # image = Image.open(os.path.join(workdir,'../images/patterns/PNG_brighter','GOES16__%s_1400.png'%day_p))
        
    
    ax.imshow(np.asarray(image),extent=[*lon_box,*lat_box],origin='upper')
    # HALO circle
    circ = Circle((lon_center,
                   lat_center),
                  r_circle, linewidth=0.8,
                  ec='w',
                  fill=False)
    ax.add_patch(circ)
    # Barbados island
    res = '50m'
    land = cfeature.NaturalEarthFeature('physical', 'land', \
                                        scale=res, edgecolor='k', \
                                        facecolor=cfeature.COLORS['land'])
    ax.add_feature(land, facecolor='beige')
    
    #- add pattern name
    # ax.text(0.98,0.03,name_p,c='w',ha='right',transform=ax.transAxes,fontsize=14)
    #- add date
    str = day_p[:4]+'-'+day_p[4:6]+'-'+day_p[6:8]+'\n'+name_p+'\n'+conf_p
    ax.text(0.98,0.98,str,c='w',ha='right',va='top',transform=ax.transAxes,fontsize=12)
    ax.outline_patch.set_visible(False)

    # change frame color
    rect = mpatches.Rectangle((0,0), width=1, height=1,edgecolor=col_pattern[name_p], facecolor="none",linewidth=3,alpha=1, transform=ax.transAxes)
    ax.add_patch(rect)
        
    
# remove last subplot
ax = axs.flatten()[-1]
fig.delaxes(ax)


# # Figure layout
# fig = plt.figure(figsize=(5,11))

# gs = GridSpec(6, 2, width_ratios=[1,1], height_ratios=[1,1,0.05,2,0.42,2],hspace=0,wspace=0.1)
# ax1 = fig.add_subplot(gs[0],projection=ccrs.PlateCarree())
# ax2 = fig.add_subplot(gs[1],projection=ccrs.PlateCarree())
# ax3 = fig.add_subplot(gs[2],projection=ccrs.PlateCarree())
# ax4 = fig.add_subplot(gs[3],projection=ccrs.PlateCarree())




#--- save
plt.savefig(os.path.join(figdir,'FigureS%d.pdf'%i_fig),bbox_inches='tight')
plt.savefig(os.path.join(figdir,'FigureS%d.png'%i_fig),dpi=300,bbox_inches='tight')



#%% --- Figure S4

i_fig = 4

fig,axs = plt.subplots(ncols=2,figsize=(8,5))

pw_max = 45 # mm
z_max = 3.2 # km
rh_cloud = 0.95

qrad_clear_by_pattern = {'Fish':[],
                   'Flower':[],
                   'Gravel':[],
                   'Sugar':[]}
iscloud_by_pattern = {'Fish':[],
                   'Flower':[],
                   'Gravel':[],
                   'Sugar':[]}
rh_clear_by_pattern = {'Fish':[],
                   'Flower':[],
                   'Gravel':[],
                   'Sugar':[]}

for day,pat,conf in zip(days,name_pattern,confidence_pattern):
    
    pw = rad_features_all[day].pw
    z_peak = rad_features_all[day].z_lw_peak/1e3 # km
    qrad_peak = rad_features_all[day].qrad_lw_peak
    keep_large = qrad_peak < -5 # K/day
    lon_day = data_all.sel(launch_time=day).longitude[:,50]
    lat_day = data_all.sel(launch_time=day).latitude[:,50]
    keep_box = np.logical_and(lon_day < lon_box[1], lat_day >= lat_box[0])
    keep_for_panel_e = np.logical_and(keep_large,keep_box)
    
    keep_low = z_peak < z_max # km
    keep_dry = pw < pw_max
    keep_subset = np.logical_and(keep_low,keep_dry)
    
    k = np.logical_and(keep_for_panel_e,keep_subset)
    
    # is cloud ?
    data_day = data_all.sel(launch_time=day)
    iscloud_day = np.any(data_day.relative_humidity > rh_cloud,axis=1).data
    iscloud_by_pattern[pat].extend(list(iscloud_day[k]))
    
    # rh
    rh_prof = data_day.relative_humidity.data
    rh_save = np.array([rh_prof[:,i_z][k] for i_z in range(rh_prof.shape[1])])
    rh_save  = np.swapaxes(rh_save,0,1)
    rh_clear_by_pattern[pat].append(rh_save)
    
    # qrad profile
    qrad_prof = data_day.q_rad_lw.data
    qrad_save = np.array([qrad_prof[:,i_z][k] for i_z in range(qrad_prof.shape[1])])
    qrad_save  = np.swapaxes(qrad_save,0,1)
    qrad_clear_by_pattern[pat].append(qrad_save)

for pat in ['Fish','Flower']:
    
    isclear = np.logical_not(iscloud_by_pattern[pat])
    
    # rh
    ax = axs[0]
    
    rh_prof = np.vstack(rh_clear_by_pattern[pat])
    rh_clear = np.array([rh_prof[:,i_z][isclear] for i_z in range(rh_prof.shape[1])])
    rh_clear = np.swapaxes(rh_clear,0,1)
    
    rh_5,rh_50,rh_95 = np.nanpercentile(rh_clear,[5,50,95],axis=0)
    
    ax.fill_betweenx(y=z,x1=rh_5,x2=rh_95,color=col_pattern[pat],edgecolor=None,
                      alpha=0.12)
    ax.plot(rh_50,z,c=col_pattern[pat],alpha=0.5)
    for i_p in range(rh_clear.shape[0]):
        ax.plot(rh_clear[i_p],z,c=col_pattern[pat],alpha=0.08,linewidth=0.6)
    
    # qrad
    ax = axs[1]
    
    qrad_prof = np.vstack(qrad_clear_by_pattern[pat])
    qrad_clear = np.array([qrad_prof[:,i_z][isclear] for i_z in range(qrad_prof.shape[1])])
    qrad_clear = np.swapaxes(qrad_clear,0,1)
    
    qrad_5,qrad_50,qrad_95 = np.nanpercentile(qrad_clear,[5,50,95],axis=0)
    
    ax.fill_betweenx(y=z,x1=qrad_5,x2=qrad_95,color=col_pattern[pat],edgecolor=None,
                      alpha=0.12)
    ax.plot(qrad_50,z,c=col_pattern[pat],alpha=0.5)
    for i_p in range(rh_clear.shape[0]):
        ax.plot(qrad_clear[i_p],z,c=col_pattern[pat],alpha=0.08,linewidth=0.6)
    

axs[1].set_xlim((-16,6))
axs[0].set_ylabel('z (km)') 
axs[0].set_xlabel('Relative humidity')
axs[1].set_xlabel('Longwave cooling (K/day)')


#--- save
plt.savefig(os.path.join(figdir,'FigureS%d.pdf'%i_fig),bbox_inches='tight')
plt.savefig(os.path.join(figdir,'FigureS%d.png'%i_fig),dpi=300,bbox_inches='tight')



#%% Figure S5 -- moist intrusions

i_fig = 5

days_high_peaks = '20200213', '20200213', '20200211', '20200209', '20200209', '20200128'
z_min_all = 5000, 4000, 4000, 3500, 5500, 4000
z_max_all = 9000, 5000, 6000, 5500, 8500, 6000
z_breaks_0_all = [1.8,2,4,5,6.5,7], [1.8,2,4,5], [2,3.5,4.5,5], [2,2.5,4.5,5], [2,2.5,5,7], [2,2.5,4.5,5] 
rh_breaks_0_all = [0.8,0.1,0.7,0.1,0.7,0.1], [0.8,0.1,0.7,0.1], [0.8,0.3,0.7,0.05], [0.75,0.1,0.25,0.05], [0.75,0.1,0.25,0.05], [0.75,0.1,0.25,0.05] 
colors = 'blue','orange','green','red','purple','brown'

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
    
    # temperature
    temp = data_day.temperature.values[k,:]
    # relative humidity
    rh = data_day.relative_humidity.values[k,:]
    # lw cooling
    qradlw = rad_features.qrad_lw_smooth[k,:]
    
    return temp, rh, qradlw


def piecewise_linear(z:np.array,z_breaks:list,rh_breaks:list):
    """
    Define piecewise linear RH shape with constant value at top and bottom.

    Args:
        z (np.array): z coordinate
        z_breaks (list): z values of break points
        rh_breaks (list): rh values of break points

    Returns:
        np.array: piecewize rh
        
    """
    
    N_breaks = len(z_breaks)
    
    cond_list = [z <= z_breaks[0]]+\
                [np.logical_and(z > z_breaks[i-1],z <= z_breaks[i]) for i in range(1,N_breaks)]+\
                [z > z_breaks[N_breaks-1]]
    def make_piece(k):
        def f(z):
            return rh_breaks[k-1]+(rh_breaks[k]-rh_breaks[k-1])/(z_breaks[k]-z_breaks[k-1])*(z-z_breaks[k-1])
        return f 
    func_list = [lambda z: rh_breaks[0]]+\
                [make_piece(k) for k in range(1,N_breaks)]+\
                [lambda z: rh_breaks[N_breaks-1]]
                
    return np.piecewise(z,cond_list,func_list)

def piecewise_fit(z:np.array,rh:np.array,z_breaks_0:list,rh_breaks_0:list):    
    """
    Compute piecewise-linear fit of RH(z).

    Args:
        z (np.array): z coordinate
        rh (np.array): rh profile
        z_breaks_0 (list): initial z values of break points
        rh_breaks_0 (list): initial rh values of break points

    Returns:
        z_breaks (list): fitted z values of break points
        rh_breaks (list): fitted rh values of break points
        rh_id (np.array): piecewize rh fit

    """

    N_breaks = len(z_breaks_0)

    def piecewise_fun(z,*p):
        return piecewise_linear(z,p[0:N_breaks],p[N_breaks:2*N_breaks])

    mask = ~np.isnan(z) & ~np.isnan(rh)

    p , e = optimize.curve_fit(piecewise_fun, z[mask], rh[mask],p0=z_breaks_0+rh_breaks_0)

    rh_id = piecewise_linear(z,p[0:N_breaks],p[N_breaks:2*N_breaks])
    rh_breaks = list(p[N_breaks:2*N_breaks])
    z_breaks = list(p[0:N_breaks])
    
    return z_breaks,rh_breaks,rh_id


def computeWPaboveZ(qv,pres,p_top):
    """Calculates the integrated water path above each level.

    Arguments:
        - qv: specific humidity in kg/kg, Nz-vector
        - pres: pressure coordinate in hPa, Nz vector
        - p_top: pressure of upper integration level

    returns:
        - wp_z: water path above each level, Nz-vector"""

    Np = qv.shape[0]
    wp_z = np.full(Np,np.nan)

    p_increasing = np.diff(pres)[0] > 0
    
    if p_increasing:
        
        i_p_top = np.where(pres >= p_top)[0][0]
        
        for i_p in range(i_p_top,Np):
        # self.wp_z[:,i_z] = self.mo.pressureIntegral(arr=data.specific_humidity[:,i_z:],pres=pres[i_z:],p_levmin=pres[i_z],p_levmax=pres[-1],z_axis=z_axis)

            arr = qv
            p = pres
            p0 = p_top
            p1 = p[i_p]
            i_w = i_p
            
            wp_z[i_w] = mo.pressureIntegral(arr=arr,pres=p,p0=p0,p1=p1)

    else:
        
        i_p_top = np.where(pres >= p_top)[0][-1]

        for i_p in range(i_p_top):
            
            arr = np.flip(qv)
            p = np.flip(pres)
            p0 = p_top
            p1 = pres[i_p]
            i_w = i_p

            wp_z[i_w] = mo.pressureIntegral(arr=arr,pres=p,p0=p0,p1=p1)

    return wp_z



fig,axs = plt.subplots(ncols=3,figsize=(10.5,4.5))
plt.rc('legend',fontsize=7)
plt.rc('legend',labelspacing=0.07)

for day, z_min, z_max, z_breaks_0, rh_breaks_0,col \
in zip(days_high_peaks,z_min_all,z_max_all,z_breaks_0_all,rh_breaks_0_all,colors):
    
    date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
    data_day = data_all.sel(launch_time=day)
    rad_features = rad_features_all[day]
    
    heights_label = r'%s; %1.1f $< z_p <$%1.1f km'%(day, z_min/1e3,z_max/1e3)
    temp, rh, qradlw = getProfiles(rad_features, data_day, z_min, z_max)
    
    # compute interquartile range and median of all profiles types
    temp_Q1, temp_med, temp_Q3 = np.nanpercentile(temp,25,axis=0), np.nanpercentile(temp,50,axis=0), np.nanpercentile(temp,75,axis=0)
    rh_Q1, rh_med, rh_Q3 = np.nanpercentile(rh,25,axis=0), np.nanpercentile(rh,50,axis=0), np.nanpercentile(rh,75,axis=0)
    qradlw_Q1, qradlw_med, qradlw_Q3 = np.nanpercentile(qradlw,25,axis=0), np.nanpercentile(qradlw,50,axis=0), np.nanpercentile(qradlw,75,axis=0)
    
    #-- everything more quantitative about intrusions
    # piecewise-linear fit
    z_breaks_id,rh_breaks_id,rh_id = piecewise_fit(z,rh_med,z_breaks_0,rh_breaks_0)
    
    # remove intrusion
    rh_breaks_remint = rh_breaks_id.copy()
    rh_breaks_remint[-2] = rh_breaks_id[-3]
    z_breaks_remint = z_breaks_id.copy()
    z_breaks_remint[-2] = z_breaks_id[-3]
    rh_remint = piecewise_linear(z,z_breaks_remint,rh_breaks_remint)
    
    # intrusion anomaly
    rh_delta_int = rh_id - rh_remint
    
    # intrusion water path
    qvstar = saturationSpecificHumidity(temp_med,pres*100)
    qv_int = rh_delta_int*qvstar
    not_nan = ~np.isnan(qv_int)
#     p_notnan = pres[not_nan][0]
    p_top = 300 # hPa
    W_cumul = computeWPaboveZ(qv_int[not_nan],pres[not_nan],p_top)
    W_int = W_cumul[0]
    # center of mass (level)
    where_W_below_half = W_cumul < W_int/2
    p_center = pres[not_nan][where_W_below_half][0]
    z_center = z[not_nan][where_W_below_half][0]
    
    print('intrusion height: %3fhPa, %1.2fkm'%(p_center,z_center))
    print('intrusion mass: %2.2fmm'%W_int)
    
    # full RH
    axs[0].plot(rh_med*100,z,linewidth=1,alpha=1,label=heights_label,c=col)
    axs[0].fill_betweenx(z,rh_Q1*100,rh_Q3*100,alpha=0.1,facecolor=col)
    # Rh in intrusion
    axs[1].plot(rh_delta_int*100,z,linewidth=1,alpha=1,label='h=%1.1fkm, W=%1.2fmm'%(z_center,W_int),linestyle='-',c=col)
    # qrad
    axs[2].plot(qradlw_med,z,linewidth=1,alpha=1,c=col)
    axs[2].fill_betweenx(z,qradlw_Q1,qradlw_Q3,alpha=0.1,facecolor=col)
    
axs[0].set_xlabel('Relative humidity (%)')
axs[1].set_xlabel('RH in intrusion (%)')
axs[2].set_xlabel('LW cooling (K/day)')
axs[0].set_ylabel('z (km)')
axs[0].legend(loc='upper right')
axs[1].legend(loc='upper right',fontsize=9)
# axs[0].legend(labelspacing = 0.5)

#--- save
plt.savefig(os.path.join(figdir,'FigureS%d.pdf'%i_fig),bbox_inches='tight')
plt.savefig(os.path.join(figdir,'FigureS%d.png'%i_fig),dpi=300,bbox_inches='tight')

#%% Functions for Fig S6

def computeWPaboveZ(qv,pres,p_top):
    """Calculates the integrated water path above each level.

    Arguments:
        - qv: specific humidity in kg/kg, Nz-vector
        - pres: pressure coordinate in hPa, Nz vector
        - p_top: pressure of upper integration level

    returns:
        - wp_z: water path above each level, Nz-vector"""

    Np = qv.shape[0]
    wp_z = np.full(Np,np.nan)

    p_increasing = np.diff(pres)[0] > 0
    
    if p_increasing:
        
        i_p_top = np.where(pres >= p_top)[0][0]
        
        for i_p in range(i_p_top,Np):
        # self.wp_z[:,i_z] = self.mo.pressureIntegral(arr=data.specific_humidity[:,i_z:],pres=pres[i_z:],p_levmin=pres[i_z],p_levmax=pres[-1],z_axis=z_axis)

            arr = qv
            p = pres
            p0 = p_top
            p1 = p[i_p]
            i_w = i_p
            
            wp_z[i_w] = mo.pressureIntegral(arr=arr,pres=p,p0=p0,p1=p1)

    else:
        
        i_p_top = np.where(pres >= p_top)[0][-1]

        for i_p in range(i_p_top):
            
            arr = np.flip(qv)
            p = np.flip(pres)
            p0 = p_top
            p1 = pres[i_p]
            i_w = i_p

            wp_z[i_w] = mo.pressureIntegral(arr=arr,pres=p,p0=p0,p1=p1)

    return wp_z

#%% Figure S6 -- Moist intrusion, spectral figure

i_fig = 6

# fig,axs = plt.subplots(ncols=3,nrows=1,figsize=(13,4))
fig,axs = plt.subplots(ncols=2,nrows=1,figsize=(8.5,4))

W = 2.39 # mm
H1 = 3 # km
H2 = 5.95 # km
i_0 = np.where(np.isin(radprf_MI_20200213lower.name.data,'W_%2.2fmm_uniform_RH'%(W)))[0][0]
i_1 = np.where(np.isin(radprf_MI_20200213lower.name.data,'W_%2.2fmm_H_%1.2fkm'%(W,H1)))[0][0]
i_2 = np.where(np.isin(radprf_MI_20200213lower.name.data,'W_%2.2fmm_H_%1.2fkm'%(W,H2)))[0][0]

i_zint_0 = i_zint_1 = np.where(z >= H1)[0][0]
i_zint_2 = np.where(z >= H2)[0][0]

# inds = (1,5,i_0,i_1,i_2)
# N_prof = len(inds)
# radprf_s = [radprf_MI_20200213lower]*N_prof
# cols = 'grey','k','b','b','b'
# linestyles = '-','-','--','-',':'

inds = (1,i_1,i_2)
inds_int = (i_zint_0,i_zint_1,i_zint_2)
N_prof = len(inds)
radprf_s = [radprf_MI_20200213lower]*N_prof
cols = 'grey','b','b','b'
linestyles = '-','-',':'


z = np.array(radprf_MI_20200213.zlay[0]/1e3) # km

##-- (a) RH
ax = axs[0]

for i_prof,radprf,col,linestyle in zip(inds,radprf_s,cols,linestyles):

    rh_prof = radprf['rh'].data[i_prof]
    ax.plot(rh_prof,z,c=col,linestyle=linestyle)
    
ax.set_ylabel('z(km)')
ax.set_xlabel(r'Relative humidity')
ax.set_title('Varying intrusion height ($W=%1.2f$mm)'%W)
ax.set_xlim((-0.03,1.03))
ax.set_ylim((-0.15,8.15))

##-- (b) nu_star
ax = axs[1]


def computeNuProfile(i_prof,radprf,band='rot'):
    
    qv_prof = radprf['h2o'].data[i_prof]
    pres_prof = radprf['play'].data/1e2
    W_prof = computeWPaboveZ(qv_prof,pres_prof,0)
    kappa_prof = 1/W_prof
    nu_prof = rad_scaling_all['20200213'].nu(kappa_prof,band=band)/1e2 # in cm-1
    
    return nu_prof,W_prof
    
#- reference without intrusion (from piecewise linear fit)
#- reference with intrusion (from piecewise linear fit)
#- homogenized rh, same water path at peak level
# radprf_s = radprf_MI_20200213, radprf_MI_20200213,radprf_RMI_20200213

band = 'rot'

z_jump = moist_intrusions['20200213, lower']['fit']['z_breaks_id'][0]
i_jump = np.where(z >= z_jump)[0][0]

nu_peak_all = []
nu_int_all = []

for i_prof,i_int,radprf,col,linestyle in zip(inds,inds_int,radprf_s,cols,linestyles):

    nu_prof,W_prof = computeNuProfile(i_prof,radprf,band=band)
    nu_peak_all.append(nu_prof[i_jump])
    nu_int_all.append(nu_prof[i_int])
    ax.plot(nu_prof,z,c=col,linestyle=linestyle)

# add points at peak height
ax.scatter(nu_peak_all[1:2],[z_jump],facecolor='k')#cols[1:2])
inv = ax.transData.inverted()
# ax.plot([nu_peak_all[1]]*2,[-1,z_jump],c='k',linestyle='-',linewidth=0.5)
# add points at intrusion center of mass
ax.scatter(nu_int_all[1:],[z[i_zint_1],z[i_zint_2]],facecolor='none',edgecolor='k')#cols[1:])
# ax.plot([nu_int_all[1]]*2,[-1,z[i_zint_1]],c='k',linestyle='-',linewidth=0.5)
# ax.plot([nu_int_all[2]]*2,[-1,z[i_zint_2]],c='k',linestyle='-',linewidth=0.5)
ax.grid()

ax.set_ylabel(' z(km)')
ax.set_xlabel(r'$\tilde{\nu}$ (cm$^{-1}$)')
ax.set_title(r'Most emitting/absorbing $\nu$ ($\tau = 1$)')
ax.set_xlim((250,700))
ax.set_ylim((-0.15,8.15))


# ##-- (c) delta transmittivity at most-emitting wavenumber
# ax = axs[2]

# z_jump = moist_intrusions['20200213, lower']['fit']['z_breaks_id'][0]
# i_jump = np.where(z >= z_jump)[0][0]

# inds_c = (i_0,i_1,i_2)

# for i_prof,radprf,col,linestyle in zip(inds,radprf_s,cols,linestyles):
    
    
#     # compute transmittivity profile
#     qv_prof = radprf['qv'].data[i_prof]
#     pres_prof = radprf['play'].data/1e2
#     W_prof = computeWPaboveZ(qv_prof,pres_prof,0)
#     W_jump = W_prof[i_jump]
#     nu_max = nu_prof[i_jump]
#     kappa_nu = rad_scaling_all['20200213'].kappa(nu_max,band=band)
#     trans_prof = np.exp(-kappa_nu*W_prof/1e3)
#     # temp_prof = radprf['h2o'].data[i_prof]
#     # piB_prof = rad_scaling_all['20200213'].planck(nu_max,temp_prof)
    
#     # EXabove = trans_prof/trans_prof[i_jump]*()
    
#     # s_jump = slice(i_jump,None)
#     # i_zint = 
    
#     ax.plot(trans_prof[i_jump]/trans_prof[s_jump],z[s_jump],c=col,linestyle=linestyle)

# ax.set_ylim((-0.15,8.15))
# ax.set_xlim((-0.01,0.15))


# ##-- (c) delta nu_star: @peak - @int as a function of intrusion height for fixed W
# ax = axs[2]
# ax_qrad = ax.twiny()

# # W = 5.87 # mm
# # W = 0.82 # mm

# Ws = 1.13, 6.18
# cols = 'g','r'

# for W,col in zip(Ws,cols):

#     z_jump = moist_intrusions['20200213, lower']['fit']['z_breaks_id'][0]
#     i_jump = np.where(z >= z_jump)[0][0]
    
#     Nsample = 20
#     Hs = np.linspace(3,10,Nsample)
#     delta_nu = np.full((Nsample,),np.nan)
#     delta_qrad = np.full((Nsample,),np.nan)
    
#     for i_H in range(Nsample):
    
#         H = Hs[i_H]
#         i_prof = np.where(np.isin(radprf_MI_20200213lower.name.data,'W_%2.2fmm_H_%1.2fkm'%(W,H)))[0][0]
#         nu_prof,W_prof = computeNuProfile(i_prof,radprf,band=band)
        
#         # height of intrusion
#         # # here center of mass
#         # i_zint = np.where(z >= H)[0][0]
#         # here lower intrusion level        
#         rh_prof = radprf['rh'].data[i_prof]
#         drylevs = np.where(rh_prof == np.nanmin(rh_prof))[0]
        
#         if len(np.where(np.diff(drylevs) > 1)[0]) > 0:
#             i_zint = drylevs[np.where(np.diff(drylevs) > 1)[0][0]]
#         else:
#             i_zint = drylevs[-1]

#         # nu        
#         nu_int = nu_prof[i_zint]
#         nu_jump = nu_prof[i_jump]
#         # spectral shift
#         delta_nu[i_H] = nu_int - nu_jump
#         # qrad
#         delta_qrad[i_H] = radprf['q_rad_lw'][i_prof][i_jump]-radprf['q_rad_lw'][1][i_jump]
        
#     ax.plot(delta_nu,Hs,c=col,linestyle='--')
#     ax_qrad.plot(delta_qrad-delta_qrad[0],Hs,c=col)


# ##-- (c) Energy exchange term as a function of intrusion height for fixed W

# Ws = 1.13, 6.18
# cols = 'g','r'

# for W,col in zip(Ws,cols):

#     z_jump = moist_intrusions['20200213, lower']['fit']['z_breaks_id'][0]
#     i_jump = np.where(z >= z_jump)[0][0]
    
#     Nsample = 20
#     Hs = np.linspace(3,10,Nsample)
#     delta_nu = np.full((Nsample,),np.nan)
#     delta_qrad = np.full((Nsample,),np.nan)
    
#     for i_H in range(Nsample):
    
#         H = Hs[i_H]
#         i_prof = np.where(np.isin(radprf_MI_20200213lower.name.data,'W_%2.2fmm_H_%1.2fkm'%(W,H)))[0][0]
#         # compute reference kappa_star
#         qv_prof = radprf['h2o'].data[i_prof]
#         temp_prof = radprf['h2o'].data[i_prof]
#         pres_prof = radprf['play'].data/1e2
#         W_prof = computeWPaboveZ(qv_prof,pres_prof,0)
#         kappa_prof = 1/W_prof
#         kappa_star = kappa_prof[i_jump]
        
#         dEXa = np.full(len(z)-i_jump+1,np.nan)
#         for i_z in range(len(z)-i_jump+1):
            
#             # FINISH
        
        
        
        
#     #     nu_prof,W_prof = computeNuProfile(i_prof,radprf,band=band)
#     #     i_zint = np.where(z >= H)[0][0]
#     #     nu_int = nu_prof[i_zint]
#     #     nu_jump = nu_prof[i_jump]
#     #     # spectral shift
#     #     delta_nu[i_H] = nu_int - nu_jump
#     #     # qrad
#     #     delta_qrad[i_H] = radprf['q_rad_lw'][i_prof][i_jump]-radprf['q_rad_lw'][1][i_jump]
        
#     # ax.plot(delta_nu,Hs,c=col,linestyle='--')
#     # ax_qrad.plot(delta_qrad-delta_qrad[0],Hs,c=col)

#--- save
plt.savefig(os.path.join(figdir,'FigureS%d.pdf'%i_fig),bbox_inches='tight')
plt.savefig(os.path.join(figdir,'FigureS%d.png'%i_fig),dpi=300,bbox_inches='tight')

#%% Figure S7 -- Diurnal cycle

i_fig = 7

rad_range = 'lw'
rad_labels = {'net':'',
              'sw':'SW',
              'lw':'LW'}
rad_titles = {'net':'Net cooling',
              'sw':'Shortwave',
              'lw':'Longwave'}

PW_lims = [20,50] # km

# colors
cmap = plt.cm.ocean_r
vmin = PW_lims[0]
vmax = PW_lims[1]

def defineCol(var_col,cmap,vmin,vmax):
    
    norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
    scmap = cm.ScalarMappable(norm=norm, cmap=cmap)
    cols = cmap(norm(var_col),bytes=True)/255 

    return cols,scmap

##-- plot
# Figure layout
# fig = plt.figure(figsize=(18,11))

# gs = GridSpec(2, 4, width_ratios=[1,1,1,0.2], height_ratios=[1,1],hspace=0,wspace=0.1)
# ax1 = fig.add_subplot(gs[0],projection=ccrs.PlateCarree())
# ax2 = fig.add_subplot(gs[1],projection=ccrs.PlateCarree())
# ax3 = fig.add_subplot(gs[2],projection=ccrs.PlateCarree())
# ax4 = fig.add_subplot(gs[4],projection=ccrs.PlateCarree())
# ax5 = fig.add_subplot(gs[5],projection=ccrs.PlateCarree())
# ax6 = fig.add_subplot(gs[6],projection=ccrs.PlateCarree())
# ax7 = fig.add_subplot(gs[:,3])

# axs = [[ax1,ax2,ax3],[ax4,ax5,ax6]]

fig,axs = plt.subplots(ncols=3,nrows=2,figsize=(13,8))

x_left = np.nan
x_right = np.nan
y_bot = np.nan
y_top = np.nan

days2show = ['20200122','20200202']

for day,axs_row in zip(days2show,axs):

    print('--',day)
    
    f = rad_features_all[day]
    var_col = f.pw
    cols,scmap = defineCol(var_col,cmap,vmin,vmax)

    date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
    day_str = date.strftime("%Y-%m-%d")
    t = np.array([(pytz.utc.localize(dt.strptime(str(t)[:19],"%Y-%m-%dT%H:%M:%S")) - date).seconds/3600 for t in f.launch_time])
    
    for ax,rad_range in zip(axs_row,list(rad_labels.keys())):
        
        print('-',rad_range)
        
        for i_lt in range(f.launch_time.size):
        
            # print(i_lt,end='..')
            x = t[i_lt]
            y = getattr(f,'qrad_%s_smooth'%rad_range)[i_lt,f.i_lw_peak[i_lt]]
            c = f.pw[i_lt]
        
            ax.scatter(x=x,y=y,c=[cols[i_lt][:3]],vmin=vmin,vmax=vmax)
        
        # titles
        if day == days2show[0]:
            ax.set_title(rad_titles[rad_range])
        # x labels
        if day == days2show[1]:
            ax.set_xlabel('Time of day (hr)')
        # y labels
        if rad_range == 'net':
            ax.set_ylabel('Cooling (K/day) on day %s'%day_str)

            
        # Save boundaries for legend
        x,y,w,h = ax.get_position().bounds
        x_left = np.nanmin(np.array([x,x_left]))
        x_right = np.nanmax(np.array([x+w,x_right]))
        y_bot = np.nanmin(np.array([y,y_bot]))
        y_top = np.nanmax(np.array([y+h,y_top]))
        
        print()
        
# ax.set_xlabel(r'Time (hours)')
# ax.set_ylabel(r'$Q_{rad}^{%s} ( z_{peak}(Q_{rad}^{LW}))$  (K/day)'%rad_labels[rad_range])

#- colorbar
# cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
#              ax=ax7,shrink=0.95,pad=0.09)
# cb.set_label('PW (mm)')

# Color bar
dx = (x_right-x_left)/60
cax = plt.axes([x_right+2*dx,y_bot,dx,y_top-y_bot])
cbar = fig.colorbar(scmap, cax=cax,orientation='vertical')
cbar.ax.set_ylabel('PW (mm)',fontsize=14)


#--- save
plt.savefig(os.path.join(figdir,'FigureS%d.pdf'%i_fig),bbox_inches='tight')
plt.savefig(os.path.join(figdir,'FigureS%d.png'%i_fig),dpi=300,bbox_inches='tight')



#%% Figure S8

i_fig = 8

from itertools import product

SSTs = np.arange(296,342,2)

Nprof = len(SSTs)


radprf = radprf_warming
z = radprf.zlay[0] # km

#-- Show

varids = 'tlay','h2o','q_rad_lw'
Nv = len(varids)+1
linestyles = '-','-'
linewidths = 2,1

# cols
var_col = np.array(SSTs,dtype=float)
norm = matplotlib.colors.Normalize(vmin=SSTs[0], vmax=SSTs[-1])
cmap = plt.cm.inferno_r
# cols = cmap(norm(var_col),bytes=True)/255
cols = cmap(norm(var_col),bytes=True)/255

fig,axs_grd = plt.subplots(nrows=2,ncols=2,figsize=(9,9))
axs = axs_grd.flatten()

# fig = plt.figure(figsize=(4*Nv,5.5))

# gs = GridSpec(1, 3, width_ratios=[3,3,1], height_ratios=[1],hspace=0.25,wspace=0.3)
# ax1 = fig.add_subplot(gs[0])
# ax2 = fig.add_subplot(gs[1])
# ax3 = fig.add_subplot(gs[2])

# axs = np.array([ax1,ax2])

#-- RH
ax = axs[0]

ax.plot(radprf.rh[0],z,'b')
# value at 300K
SST = 300 # K
name = 'RHid_SST%d'%SST
i_p_file = np.where(radprf.name.data == name)[0][0]
var = radprf.rh.data[i_p_file,:]

ax.plot(radprf.rh[i_p_file],z,'k')


#-- T, qv and qrad LW
for i_ax in range(1,Nv):
    
    ax = axs.flatten()[i_ax]
    varid = varids[i_ax-1]
    
    ax.plot(radprf[varid].data[0],z,c='b',linewidth=linewidths[0],linestyle=linestyles[0])
    
    for i_prof in range(Nprof):
        
        SST = SSTs[i_prof]
        name = 'RHid_SST%d'%SST
        i_p_file = np.where(radprf.name.data == name)[0][0]
        
        var = radprf[varid].data[i_p_file,:]
        col = cols[i_prof]
        
        ax.plot(var,z,c=col,linewidth=linewidths[1],linestyle=linestyles[1],alpha=0.5)

# colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=axs.ravel().tolist(),aspect=35)
cbar.set_label('SST (K)')

for ax in axs[:2]:
    ax.set_ylim((-0.1,9.1))
    
for ax in axs[2:]:
    ax.set_ylim((-0.1,4.1))


axs[1].set_xlim((238,342))
axs[2].set_xscale('log')
axs[2].set_xlim((3e-4,6e-1))
axs[3].set_xlim((-35.3,5.3))

axs[0].set_xlabel('Relative humidity')
axs[1].set_xlabel('Temperature (K)')        
axs[2].set_xlabel('Specific humidity (kg/kg)')
axs[3].set_xlabel('LW cooling (K/day)')

axs[0].set_ylabel('z (km)')
axs[2].set_ylabel('z (km)')

# suptitle at midpoint of left and right x-positions
# mid = (fig.subplotpars.right + fig.subplotpars.left)/2
# plt.suptitle(r'Varying intrusion height at $W=%1.2f$ mm'%W,fontsize=15,x=mid)

#--- save
plt.savefig(os.path.join(figdir,'FigureS%d.pdf'%i_fig),bbox_inches='tight')
plt.savefig(os.path.join(figdir,'FigureS%d.png'%i_fig),dpi=300,bbox_inches='tight')



#%% Figure S9

i_fig = 9

