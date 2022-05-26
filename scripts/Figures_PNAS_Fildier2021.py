#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 09:37:27 2021

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

from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

#%%    ###--- Additional functions ---###

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


def confidence_ellipse(x, y, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)

    return ellipse
    


#-- functions for connecting zoomed subplots

from matplotlib.transforms import Bbox, TransformedBbox, blended_transform_factory

from mpl_toolkits.axes_grid1.inset_locator import BboxPatch, BboxConnector,\
    BboxConnectorPatch

def connect_bbox(bbox1, bbox2,
                 loc1a, loc2a, loc1b, loc2b,
                 prop_connect, prop_boxes=None):
    
    if prop_boxes is None:
        prop_boxes = prop_connect.copy()
        prop_boxes["alpha"] = prop_boxes.get("alpha", 1)*0.2

    c1 = BboxConnector(bbox1, bbox2, loc1=loc1a, loc2=loc2a, **prop_connect)
    c1.set_clip_on(False)
    c2 = BboxConnector(bbox1, bbox2, loc1=loc1b, loc2=loc2b, **prop_connect)
    c2.set_clip_on(False)

    bbox_patch1 = BboxPatch(bbox1, **prop_boxes)
    bbox_patch2 = BboxPatch(bbox2, **prop_boxes)

    p = BboxConnectorPatch(bbox1, bbox2,
                           loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                           **prop_connect)
    p.set_clip_on(False)

    return c1, c2, bbox_patch1, bbox_patch2, p

def zoom_effect_yaxis(ax1, ax2, **kwargs):
    """
    ax2 : the big main axes
    ax1 : the zoomed axes
    The xmin & xmax will be taken from the
    ax1.viewLim.
    """

    tt = ax1.transScale + (ax1.transLimits + ax2.transAxes)
    trans = blended_transform_factory(tt,ax2.transData)

    mybbox1 = ax1.bbox
    mybbox2 = TransformedBbox(ax1.viewLim, trans)

    prop_boxes = kwargs.copy()
    prop_boxes["ec"] = "darkgoldenrod"
    prop_boxes["fc"] = "cornsilk"
    prop_boxes["alpha"] = 0.2
    
    prop_connect = kwargs.copy()
    prop_connect["ec"] = "darkgoldenrod"
    prop_connect["linestyle"] = '-'
    prop_connect["linewidth"] = 0.8
    prop_connect["alpha"] = 0.2

    c1, c2, bbox_patch1, bbox_patch2, p = \
        connect_bbox(mybbox1, mybbox2,
                     loc1a=2, loc2a=1, loc1b=3, loc2b=4, 
                     prop_connect=prop_connect, prop_boxes=prop_boxes)

    ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p

def zoom_effect_xaxis(ax1, ax2, **kwargs):
    """
    ax2 : the big main axes
    ax1 : the zoomed axes
    The xmin & xmax will be taken from the
    ax1.viewLim.
    """

    tt = ax1.transScale + (ax1.transLimits + ax2.transAxes)
    trans = blended_transform_factory(ax2.transData,tt)

    mybbox1 = ax1.bbox
    mybbox2 = TransformedBbox(ax1.viewLim, trans)

    prop_boxes = kwargs.copy()
    prop_boxes["ec"] = "k"
    prop_boxes["fc"] = "none"
    prop_boxes["linestyle"] = '--'
    # prop_boxes["linewidth"] = 0.4
    prop_boxes["alpha"] = 0.2
    
    prop_connect = kwargs.copy()
    prop_connect["ec"] = "k"
    prop_connect["linestyle"] = '--'
    # prop_connect["linewidth"] = 0.8
    prop_connect["alpha"] = 0.4

    c1, c2, bbox_patch1, bbox_patch2, p = \
        connect_bbox(mybbox1, mybbox2,
                     loc1a=1, loc2a=4, loc1b=2, loc2b=3, 
                     prop_connect=prop_connect, prop_boxes=prop_boxes)

    ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p

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

    # load info on all moist intrusions
    mi_file = os.path.join(repodir,'results','idealized_calculations','observed_moist_intrusions','moist_intrusions.pickle')
    moist_intrusions = pickle.load(open(mi_file,'rb'))

#%%  Figure #1
    
    def findQradPeakLocation(values,n_smooth_0=15,return_all=False):
        """Returns a list of vertical indices for a given profile"""
        
        val_smooth_0 = np.convolve(values,np.repeat([1/n_smooth_0],n_smooth_0),
                                 'same')
        
        ind = np.nanargmin(val_smooth_0)
    
        if return_all:
            return ind, val_smooth_0[ind], val_smooth_0
        else:
            return ind, val_smooth_0[ind]


    # Example two profiles
    
    # works okay
    # day = '20200126'
    # i_h = 15
    # inds_prof = 2,4,3
    # cols = 'b','g','r'
    
    
    day = '20200126'
    i_h = 15

    # inds_prof = 2,6
    inds_prof = 2,4,3
    cols = 'b','g','r'
    n_profs = len(inds_prof)
    
    data_day = radprf.where(radprf.z_min<=50,drop=True).sel(launch_time=day)
    times = np.array([pytz.utc.localize(dt.strptime(str(d)[:19],'%Y-%m-%dT%H:%M:%S')) for d in data_day.launch_time.values])
    
    # time
    time_init = pytz.utc.localize(dt(2020,1,26))
    time_current = time_init+timedelta(hours=i_h)
    time_label = time_current.strftime("%Y%m%d_%H")
    
    mask_hour = np.logical_and(times > time_init+timedelta(hours=i_h), times <= time_init+timedelta(hours=i_h+1))
    data_hour = data_day.sel(launch_time=mask_hour)


    # Figure layout
    fig = plt.figure(figsize=(14,11))
    
    gs = GridSpec(2, 4, width_ratios=[1,1,1,1], height_ratios=[1, 1.5], hspace=0.25)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[0,2:4],projection=ccrs.PlateCarree())
    ax4 = fig.add_subplot(gs[1,:2])
    ax5 = fig.add_subplot(gs[1,2:4])
    
    cmap = plt.cm.BrBG_r
    vmax = 6
    vmin = -vmax
    
    #-- (a)
    ax = ax1
    
    for i_lt in range(n_profs):
        
        i_p = inds_prof[i_lt]
        rh = data_hour.relative_humidity.values[i_p]
        # ind,qrad_peak,qrad_smooth = findQradPeakLocation(qrad,return_all=True)
        # hour_label = dt.strptime(str(data_hour.launch_time.values[i_p])[:19],"%Y-%m-%dT%H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
        # print(ind,hour_label)
        # if ind is not None:
        #     z_peak = data_hour.alt[ind] 
        
        ax.plot(rh,z,alpha=0.8,c=cols[i_lt])
        # ax.scatter(qrad_peak,z_peak/1e3,marker='o',s=50,edgecolors='k',facecolors=cols[i_lt])
    
    ax.set_xlabel(r'Relative humidity')
    ax.set_ylabel(r'z (km)')
#     ax.set_title("Example dry and moist profiles")
    ax.set_xlim(-0.08,1.08)
    ax.set_ylim(-0.2,6.2)
    
    #-- (b) 
    ax = ax2
    
    for i_lt in range(n_profs):
        
        i_p = inds_prof[i_lt]
        qrad = data_hour.q_rad_lw.values[i_p]
        ind,qrad_peak,qrad_smooth = findQradPeakLocation(qrad,return_all=True)
        hour_label = dt.strptime(str(data_hour.launch_time.values[i_p])[:19],"%Y-%m-%dT%H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
        print(ind,hour_label)
        if ind is not None:
            z_peak = data_hour.alt[ind] 
        
        ax.plot(qrad,z,alpha=0.8,c=cols[i_lt])
        ax.plot(qrad_smooth,z,'k--',linewidth=1)
        ax.scatter(qrad_peak,z_peak/1e3,marker='o',s=50,edgecolors='k',facecolors=cols[i_lt])
    
    ax.set_xlabel(r'Longwave cooling (K/day)')
    ax.set_ylabel(r'z (km)')
#     ax.set_title("Example dry and moist profiles")
    ax.set_xlim(-14,1)
    ax.set_ylim(-0.2,6.2)
    
    #-- (c)
    ax = ax3
    
    im_hour = images_vis[i_h]
    ax.coastlines(resolution='50m')
    ax.set_extent([*lon_box,*lat_box])
    ax.imshow(im_hour,extent=[*lon_box,*lat_box],origin='upper')
    gl = ax.gridlines(color='Grey',draw_labels=True)
    
    # show sondes
    for i_lt in range(n_profs):

        i_p = inds_prof[i_lt]
        x_s = data_hour.longitude.values[i_p,10]
        y_s = data_hour.latitude.values[i_p,10]
        print(x_s,y_s)
        
        m = ax.scatter(x_s,y_s,marker='o',color=cols[i_lt],alpha=0.6,s=80,label='Dropsondes')
    
    #---- bottom panels, PW composites and circulation
    
    xlim = (8,59)
    ylim = (0,10)
    
    #-- (d)
    ax = ax4
    
    cond_varid = 'QRADLW'
    ref_varid = 'PW'
    
    array = cond_dist_all[cond_varid][day].cond_mean
    pw_perc = cond_dist_all[cond_varid][day].on.percentiles
    X,Y = np.meshgrid(pw_perc,z)
    ax.contourf(X,Y,array,cmap=cmap,vmin=vmin,vmax=vmax,levels=20)
    
    ax.set_xlabel('PW (mm)')
    ax.set_ylabel('z(km)')
    ax.set_title(r'EUREC$^4$A observations, 2020-01-26')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    #-- (e)
    ax = ax5
    
    # qrad
    X,Y = np.meshgrid(c_muller['pws'][:-1,0],c_muller['z1d'][:,0]/1e3)
    ax.contourf(X,Y,c_muller['Qrs'].T,cmap=cmap,vmin=vmin,vmax=vmax,levels=20)
    # cloud water
    ax.contour(X,Y,c_muller['qns'].T,colors='w',linewidths=3,origin='lower')
    # circulation
    ax.contour(X,Y,c_muller['psis'].T,colors='k',levels=4,origin='lower')
    
    ax.set_xlabel('PW (mm)')
    ax.set_ylabel('z(km)')
    ax.set_title('Simulation of deep convection, Muller (2015)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    


#%%

    from astral import LocationInfo
    from astral.sun import sun
    
    def timeDuringDaytime(date,t):

        city = LocationInfo("Barbados", "Barbados", "Etc/GMT", 3.1939,-59.5432)
        s = sun(city.observer, date=date,tzinfo=city.timezone)
        time_sunrise = s['sunrise']
        time_sunset = s['sunset']
        
        return np.logical_and(t > time_sunrise,t < time_sunset)
        

    def computeWeightedImage(rad_features,data_day,images,sigma=1):
        
        
        # weighted image
        lon_s, lat_s, time_s = getProfileCoords(rad_features, data_day)
        N_s = len(lon_s)
        
        # colors
        C_0 = images[0]
        N_lat,N_lon,N_c = C_0.shape
        # array of local coordinates (x,y)
        lon_x = np.linspace(*lon_box,N_lon)
        lat_y = np.linspace(*lat_box,N_lat)
        lon_yx, lat_yx = np.meshgrid(lon_x,lat_y)
        x = np.moveaxis(np.tile(lon_yx,(N_c,1,1)),0,-1)
        y = np.moveaxis(np.tile(lat_yx,(N_c,1,1)),0,-1)
    
        # initializing    
        weighted_sum = np.zeros(C_0.shape)
        normalizing_factor = np.zeros(C_0.shape)
        
        for i_s in range(N_s):
    
            # array of sonde coordinates (x_s,y_s)
            x_s = lon_s[i_s] * np.ones((N_lat,N_lon,N_c))
            y_s = lat_s[i_s] * np.ones((N_lat,N_lon,N_c))
            
            # find closest image in time
            if not timeDuringDaytime(date, time_s[i_s]):
                C_s = np.zeros(C_0.shape)
            else:
                i_h = time_s[i_s].hour
                if time_s[i_s].minute > 30 and i_h < len(images_vis)-1:
                    i_h = i_h+1
                C_s = images[i_h]
            
            # add to sums
            dist_s = np.sqrt(np.power(x-x_s,2.)+np.power(y-y_s,2.))
            weighted_sum = weighted_sum + C_s * np.exp(-dist_s/sigma)
            normalizing_factor = normalizing_factor + np.exp(-dist_s/sigma)


        return weighted_sum/normalizing_factor
    
#%% Test weighted images
    
    wimage_sigma01 = computeWeightedImage(rad_features,data_day,images_vis,sigma=0.1)
    wimage_sigma1 = computeWeightedImage(rad_features,data_day,images_vis,sigma=1)
    wimage_sigma3 = computeWeightedImage(rad_features,data_day,images_vis,sigma=3)
    wimage_sigma5 = computeWeightedImage(rad_features,data_day,images_vis,sigma=5)
    wimage_sigma20 = computeWeightedImage(rad_features,data_day,images_vis,sigma=20)
    wimage_sigma100 = computeWeightedImage(rad_features,data_day,images_vis,sigma=100)
    
    ims = [wimage_sigma01,wimage_sigma1,wimage_sigma100]
    
    fig,axs = plt.subplots(nrows=3,figsize=(9,18),subplot_kw={'projection':ccrs.PlateCarree()})
    
    for ax,im in zip(axs,ims):
        
        ax.coastlines(resolution='50m')
        ax.set_extent([*lon_box,*lat_box])
        ax.imshow((im).astype(int),extent=[*lon_box_goes,*lat_box_goes],origin='upper')
        gl = ax.gridlines(color='Grey',draw_labels=True)
    
#%% Figure 1 precompute weighted satellite image
    
    day = '20200126'
    rad_features = rad_features_all[day]
    wimage_sigma01 = computeWeightedImage(rad_features,data_day,images_vis,sigma=0.1)    

#%% Figure 1 second attempt

    def getProfileCoords(rad_features, data_day):
        
        #- Mask
        # |qrad| > 5 K/day
        qrad_peak = np.absolute(rad_features.qrad_lw_peak.data)
        keep_large = qrad_peak > 5 # K/day
        # in box
        lon_day = data_day.longitude.data[:,50]
        lat_day = data_day.latitude.data[:,50]
        keep_box = np.logical_and(lon_day < lon_box[1], lat_day >= lat_box[0])
        
        # combined
        k = np.logical_and(keep_large,keep_box)
        
        # longitude
        lon = lon_day[k]
        # latitude
        lat = lat_day[k]
        # time
        time_day = np.array([dt.strptime(str(launch_time)[:16],'%Y-%m-%dT%H:%M').replace(tzinfo=pytz.UTC) for launch_time in data_day.launch_time.data])
        time = time_day[k]
        
        return lon,lat,time
    
    def getProfiles(rad_features, data_day, z_min, z_max):
        
        #- Mask
        # |qrad| > 5 K/day
        qrad_peak = np.absolute(rad_features.qrad_lw_peak)
        keep_large = qrad_peak > 5 # K/day
        # in box
        lon_day = data_day.longitude[:,50]
        lat_day = data_day.latitude[:,50]
        keep_box = np.logical_and(lon_day < lon_box[1], lat_day >= lat_box[0])
        # z range
        keep_between_z =  np.logical_and(rad_features.z_net_peak <= z_max, # m
                                         rad_features.z_net_peak > z_min)
        # combined
        k = np.logical_and(np.logical_and(keep_large,keep_box),keep_between_z)
        
        # relative humidity    
        rh = data_day.relative_humidity.values[k,:]*100
        # lw cooling
        qradlw = rad_features.qrad_lw_smooth[k,:]
        # longitude
        lon = lon_day[k]
        # latitude
        lat = lat_day[k]
        # original indices
        ind = np.where(k)[0]
        
        return lon,lat,rh,qradlw,ind
    
    def getProfilesPWrange(rad_features, data_day, pw_min, pw_max):
        
        #- Mask
        # |qrad| > 5 K/day
        qrad_peak = np.absolute(rad_features.qrad_lw_peak)
        keep_large = qrad_peak > 5 # K/day
        # in box
        lon_day = data_day.longitude[:,50]
        lat_day = data_day.latitude[:,50]
        keep_box = np.logical_and(lon_day < lon_box[1], lat_day >= lat_box[0])
        # high-level peak
        keep_between_pw =  np.logical_and(rad_features.pw <= pw_max, # m
                                          rad_features.pw > pw_min)
        # combined
        k = np.logical_and(np.logical_and(keep_large,keep_box),keep_between_pw)
        
        # relative humidity    
        rh = data_day.relative_humidity.values[k,:]*100
        # lw cooling
        qradlw = rad_features.qrad_lw_smooth[k,:]
        # longitude
        lon = lon_day[k]
        # latitude
        lat = lat_day[k]
        # original indices
        ind = np.where(k)[0]
        
        return lon,lat,rh,qradlw,ind
    
    # Figure layout
    fig = plt.figure(figsize=(10,8))
    
    gs = GridSpec(2, 4, width_ratios=[1,1,1,1], height_ratios=[1, 1.5], hspace=0.34)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[0,2:4],projection=ccrs.PlateCarree())
    ax4 = fig.add_subplot(gs[1,:2])
    ax5 = fig.add_subplot(gs[1,2:4])
    
    cmap = plt.cm.BrBG_r
    vmax = 8
    vmin = -vmax
    
    day = '20200126'
    date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
    data_day = data_all.sel(launch_time=day)
    rad_features = rad_features_all[day]
    
    #-- background (c)
    ax = ax3
    
    # # image at specific time
    # i_h = 15 # here, index = time UTC
    # image = images_vis[i_h]
    # weighted image
    sigma = 0.1
    # wimage_sigma01 = computeWeightedImage(rad_features,data_day,images_vis,sigma=sigma) # ---> compute beforehand to speed up
    image = wimage_sigma01.astype(int)
    
    ax.coastlines(resolution='50m')
    ax.set_extent([*lon_box,*lat_box])
    ax.imshow(image,extent=[*lon_box_goes,*lat_box_goes],origin='upper')
    gl = ax.gridlines(color='Grey',draw_labels=True)
    
    
    #-- (a) & (b)
    
    # separate peaks above and below 2.5 km altitude
    cols = 'r','b'
    z_min_all = 100, 2200
    z_max_all = 2200, 100000
    pw_min_all = 10, 30
    pw_max_all = 30, 50
    alphas = 0.15, 0.2
    labs = r'peak below %1.1f km'%(z_min_all[1]/1e3), r'peak above %1.1f km'%(z_min_all[1]/1e3)
    # labs = r'PW $\le$ 30mm', r'PW $\gt$ 30mm'
    
    for col, z_min, z_max, lab, alpha in zip(cols,z_min_all,z_max_all,labs,alphas):
        
        lon, lat, rh, qradlw, ind = getProfiles(rad_features, data_day, z_min, z_max)

    # for col, pw_min, pw_max, alpha in zip(cols,pw_min_all,pw_max_all,alphas):
        
    #     lon, lat, rh, qradlw, ind = getProfilesPWrange(rad_features, data_day, pw_min, pw_max)
        
        rh_mean = np.nanmean(rh,axis=0)
        qradlw_mean = np.nanmean(qradlw,axis=0)
        
        for i_s in range(rh.shape[0]):
            
            # rh
            ax1.plot(rh[i_s],z,c=col,linewidth=0.1,alpha=alpha)
            # qradlw
            ax2.plot(qradlw[i_s],z,c=col,linewidth=0.1,alpha=alpha)
            # location
            time_init = pytz.utc.localize(dt(2020,1,26))
            time_image = time_init+timedelta(hours=i_h)
            time_current = dt.strptime(str(data_day.launch_time.data[ind[i_s]])[:16],'%Y-%m-%dT%H:%M')
            time_current = time_current.replace(tzinfo=pytz.UTC)
            delta_time = time_current-time_image
            delta_time_hour = (delta_time.days*86400 + delta_time.seconds)/3600
            # print(ind[i_s],time_current,delta_time_hour)
            
            # ax3.scatter(lon[i_s],lat[i_s],marker='o',color=col,alpha=exp(-np.abs(delta_time_hour)/2),s=20)
            ax3.scatter(lon[i_s],lat[i_s],marker='o',color=col,alpha=0.3,s=20)
            
        # rh
        ax1.plot(rh_mean,z,c=col,linewidth=1,alpha=1)
        # qradlw
        ax2.plot(qradlw_mean,z,c=col,linewidth=1,alpha=1,label=lab)
        
    # RH labels & range
    ax = ax1
    ax.set_xlabel(r'Relative humidity (%)')
    ax.set_ylabel(r'z (km)')
    ax.set_xlim(-8,108)
    ax.set_ylim(-0.2,6.2)
    # QradLW labels & range
    ax = ax2
    ax.set_xlabel(r'Longwave cooling (K/day)')
    # ax.set_ylabel(r'z (km)')
    ax.set_xlim(-14,1)
    ax.set_ylim(-0.2,6.2)
    ax.legend(loc='upper left', fontsize=6)
    
    
    #---- bottom panels, PW composites and circulation
    
    xlim = (8,59)
    ylim = (0,10)
    var_lab = r'Longwave cooling (K/day)'
    
    #-- (d)
    ax = ax4
    
    cond_varid = 'QRADLW'
    ref_varid = 'PW'
    
    array = cond_dist_all[cond_varid][day].cond_mean
    pw_perc = cond_dist_all[cond_varid][day].on.percentiles
    X,Y = np.meshgrid(pw_perc,z)
    ax.contourf(X,Y,array,cmap=cmap,vmin=vmin,vmax=vmax,levels=20)
    
    ax.set_xlabel(r'Precipitable water $W_{sfc}$ (mm)')
    ax.set_ylabel('z(km)')
    ax.set_title(r'EUREC$^4$A observations, 2020-01-26')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    xlim = (8,59)
    ylim = (0,10)
    
    #- colorbar 
    # colors
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cols = cmap(norm(array),bytes=True)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                 ax=ax,shrink=0.99,pad=0.04)
    cb.set_label(var_lab)

    #-- (e)
    ax = ax5

    # qrad
    X,Y = np.meshgrid(c_muller['pws'][:-1,0],c_muller['z1d'][:,0]/1e3)
    var = c_muller['Qrs_lw']
    ax.contourf(X,Y,var.T,cmap=cmap,vmin=vmin,vmax=vmax,levels=20)
    # cloud water
    ax.contour(X,Y,c_muller['qns'].T,colors='w',linewidths=1,origin='lower')
    # circulation
    ax.contour(X,Y,c_muller['psis'].T,colors='k',levels=4,linewidths=1,origin='lower')
    
    ax.set_xlabel(r'Precipitable water $W_{sfc}$ (mm)')
    # ax.set_ylabel('z(km)')
    ax.set_title('Deep convective aggregation,\n Muller & Bony (2015)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    #- colorbar 
    # colors
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cols = cmap(norm(var),bytes=True)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                 ax=ax,shrink=0.99,pad=0.04)
    cb.set_label(var_lab)
    
    #--- Add panel labeling
    axs = ax1,ax2,ax3,ax4,ax5
    pan_labs = '(a)','(b)','(c)','(d)','(e)'
    x_locs = 0.04,0.04,0.03,0.03,0.03
    y_locs = 0.04,0.04,0.04,0.93,0.93
    t_cols = 'k','k','w','k','k'
    for ax,pan_lab,x_loc,y_loc,t_col in zip(axs,pan_labs,x_locs,y_locs,t_cols):
        ax.text(x_loc,y_loc,pan_lab,transform=ax.transAxes,fontsize=14,color=t_col)
        
    #--- Save
    plt.savefig(os.path.join(figdir,'Figure1.pdf'),bbox_inches='tight')
    

#%%     Precalculations for Fig 2 -- all scalings

    #- First scaling magnitude (eq 8)
    
    H_peak_all = {}

    days2show = days
    
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
            
            # beta
            beta = f.beta_peak[i_s]
            # spectral integral
            spec_int = rs.spectral_integral_rot[i_s][i_peak] + rs.spectral_integral_vr[i_s][i_peak]
            # constants
            C = -gg/c_pd
            
            H_peak[i_s] = C/(p_peak*hPa_to_Pa) * beta * spec_int * day_to_seconds
            
        H_peak_all[day] = H_peak
        
        
    scaling_1 = np.hstack([H_peak_all[day] for day in days2show])
    
    #- Second scaling magnitude (eq 9)
    
    H_peak_all = {}
    
    days2show = days
    
    for day in days2show:
        
        rs = rad_scaling_all[day]
        f = rad_scaling_all[day].rad_features
        k_bottom = np.where(np.logical_not(np.isnan(f.wp_z[0])))[0][0] 
        
        #- approximated peak
        Ns = rad_scaling_all[day].rad_features.pw.size
        H_peak = np.full(Ns,np.nan)
        
        for i_s in range(Ns):

            i_peak = f.i_lw_peak[i_s]
            p_peak = f.pres_lw_peak[i_s]
            
            # beta
            beta = f.beta_peak[i_s]
            # approximation of spectral integral
            piB_star = 0.0054 # (sum of piB in rotational and v-r bands)
            delta_nu = 1 # cm-1
            spec_int_approx = piB_star * delta_nu*m_to_cm/e
            # constants
            C = -gg/c_pd
            
            H_peak[i_s] = C/(p_peak*hPa_to_Pa) * beta * spec_int_approx * day_to_seconds
        
        H_peak_all[day] = H_peak
        
    scaling_2_core = np.hstack([H_peak_all[day] for day in days2show])
    
    # spectral width fitted between scaling 1 and 2
    
    def f_linear(x,a):
        return a*x
    
    fit_params = curve_fit(f_linear,scaling_2_core,scaling_1,p0=1)
    
    # spectral width fitted
    delta_nu_fitted = fit_params[0][0]
    delta_nu = 120
    # scaling 2
    scaling_2 = delta_nu*scaling_2_core

    #- Third scaling magnitude (eq 10)
    
    H_peak_all = {}
    
    days2show = days
    
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
            piB_star = 0.0054
            delta_nu = delta_nu # cm-1
            spec_int_approx = piB_star * delta_nu*m_to_cm/e
            # constants
            alpha = 1.5
            C = -gg/c_pd * (1+alpha)
            
            H_peak[i_s] = C/(p_peak*hPa_to_Pa) * CRHbelow[i_peak]/CRHabove[i_peak] * spec_int_approx * day_to_seconds
        
        H_peak_all[day] = H_peak
    
    
    scaling_3 = np.hstack([H_peak_all[day] for day in days2show])
    
    
    # true qrad magnitude
    qrad_peak_all = {}    
    days2show = days
    
    for day in days2show:
        qrad_peak = rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak
        qrad_peak_all[day] = qrad_peak
    
    true_qrad = np.hstack([qrad_peak_all[day] for day in days2show])
    
    
    # marker size
    # s_all = []
    # days2show = days
    
    # for day in days2show:
    
    #     s_all.append(0.005*np.absolute(rad_scaling_all[day].rad_features.lw_peaks.pres_lw_peak))
    
    # s = np.hstack(s_all)
    s = 0.005

#%% --- Figure 2 ---

    label_jump = '^\dagger'
    m_to_cm = 1e2
    day_to_seconds = 86400
    hPa_to_Pa = 1e2

    # import mpl_scatter_density
    # import datashader as ds
    # from datashader.mpl_ext import dsshow
    # import pandas as pd
    
    def scatterDensity(ax,x,y,s,alpha):
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        return ax.scatter(x,y,c=z,s=s,alpha=0.4)

    # def scatterDensity(ax,x,y,s,alpha=0.4):
        
        # return ax.scatter_density(x,y,c=z,s=s,alpha=alpha)
        


    fig,axs = plt.subplots(ncols=2,nrows=2,figsize=(10,10))
    plt.subplots_adjust(hspace=0.3,wspace=0.3)
    
    #-- (a) peak height, using the approximation for the full profile, showing all profiles
    ax = axs[0,0]
    
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
        # pres_beta_over_p_peak = rad_scaling_all[day].rad_features.beta_over_p_peaks.pres_beta_over_p_peak
        # pres_scaling_profile_peak = rad_scaling_all[day].rad_features.scaling_profile_peaks.pres_scaling_profile_peak
        # pres_proxy_peak = pres_scaling_profile_peak
        pres_proxy_peak = pres_beta_peak
        y.append(pres_proxy_peak)
        
        s.append(np.absolute(rad_scaling_all[day].rad_features.lw_peaks.qrad_lw_peak))
    
    # peaks
    x,y,s = np.hstack(x),np.hstack(y),np.hstack(s)
    h = scatterDensity(ax,x,y,s,alpha=0.4)
    # 1:1 line
    ax.plot([910,360],[910,360],'k',linewidth=0.5,alpha=0.5)

    
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel('$Q_{rad}$ peak height (hPa)')
    ax.set_ylabel(r'$\beta$ peak height (hPa)')
    ax.set_title(r'Height as $\beta$ maximum')
    # square figure limits
    ylim = ax.get_ylim()
    ax.set_xlim(ylim)
    
    # ax.set_title(r'Approximating peak height\n as $p^\star = \arg\max_p \left(-\frac{g}{c_p}\frac{\beta}{p}\int B_\nu \phi_\nu\right)$')
    
    # colorbar density
    axins1 = inset_axes(ax,
                    width="50%",  # width = 70% of parent_bbox width
                    height="2%",  # height : 5%
                    loc='lower right')
    cb = fig.colorbar(h, cax=axins1, orientation="horizontal")
    axins1.xaxis.set_ticks_position("top")
    axins1.tick_params(axis='x', labelsize=9)
    cb.set_label('Gaussian kernel density',labelpad=-34)
    
    
    #-- (b) peak magnitude, using the approximation for the full profile, showing all profiles
    ax = axs[0,1]
    
    # plot
    x = true_qrad
    y = scaling_1
    s = 5
    h = scatterDensity(ax,x,y,s,alpha=0.5)
    
    # # 1:1 line
    # x_ex = np.array([-18,-2])
    # ax.plot(x_ex,x_ex,'k:',alpha=0.4,label='1 : 1')
    
    ax.set_xlabel('True LW $Q_{rad}$ (K/day)')
    ax.set_ylabel('Estimate (K/day)')
    ax.set_title(r'Magnitude scaling, eq. (8)')
    # square figure limits
    xlim = (-20.4,-1.6)
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    
    #- linear fit
    xmin,xmax = np.min(x), np.max(x) 
    xrange = xmax-xmin
    a_fit , pcov = optimize.curve_fit(lambda x,a:a*x, x, y,p0=1)
    x_fit = np.linspace(xmin-xrange/40,xmax+xrange/40)
    y_fit = a_fit*x_fit
    cov = np.cov(x,y)
    r_fit = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    
    # show
    ax.plot(x_fit,y_fit,'k:')
    # write numbers
    ax.text(0.05,0.05,'$Q_{rad}^{est} = %1.2f Q_{rad} $\n r=%1.2f'%(a_fit,r_fit),transform=ax.transAxes)
    
    # colorbar density
    axins1 = inset_axes(ax,
                    width="50%",  # width = 70% of parent_bbox width
                    height="2%",  # height : 5%
                    loc='lower right')
    cb = fig.colorbar(h, cax=axins1, orientation="horizontal")
    axins1.xaxis.set_ticks_position("top")
    axins1.tick_params(axis='x', labelsize=9)
    cb.set_label('Gaussian kernel density',labelpad=-34)
    
    #-- (c) peak magnitude, using the intermediate approximation (beta and 1 wavenumber), showing all profiles
    ax = axs[1,0]
        
    # plot
    x = true_qrad
    y = scaling_2
    h = scatterDensity(ax,x,y,s,alpha=0.5)

    # # 1:1 line
    # x_ex = np.array([-18,-2])
    # ax.plot(x_ex,x_ex,'k:',alpha=0.4,label='1 : 1')
    
    ax.set_xlabel('True LW $Q_{rad}$ (K/day)')
    ax.set_ylabel('Estimate (K/day)')
    ax.set_title(r'Magnitude scaling, eq. (9)')
    # square figure limits
    xlim = (-20.4,-1.6)
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    
    #- linear fit
    a_fit , pcov = optimize.curve_fit(lambda x,a:a*x, x, y,p0=1)
    x_fit = np.linspace(xmin-xrange/40,xmax+xrange/40)
    y_fit = a_fit*x_fit
    # y_fit = 1*x_fit
    cov = np.cov(x,y)
    r_fit = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    
    # show
    ax.plot(x_fit,y_fit,'k:')
    # write numbers
    ax.text(0.05,0.05,'$Q_{rad}^{est} = %1.2f Q_{rad} $\n r=%1.2f'%(a_fit,r_fit),transform=ax.transAxes)
    
    # colorbar density
    axins1 = inset_axes(ax,
                    width="50%",  # width = 70% of parent_bbox width
                    height="2%",  # height : 5%
                    loc='lower right')
    cb = fig.colorbar(h, cax=axins1, orientation="horizontal")
    axins1.xaxis.set_ticks_position("top")
    axins1.tick_params(axis='x', labelsize=9)
    cb.set_label('Gaussian kernel density',labelpad=-34)
    
    #-- (d) peak magnitude, using the simplified scaling (RH step function and 1 wavenumber), showing all profiles
    ax = axs[1,1]

    # plot
    x = true_qrad
    y = scaling_3
    
    h = scatterDensity(ax,x,y,s,alpha=0.5)

    # # 1:1 line
    # x_ex = np.array([-18,-2])
    # ax.plot(x_ex,x_ex,'k:',alpha=0.4,label='1 : 1')
    
    ax.set_xlabel('True LW $Q_{rad}$ (K/day)')
    ax.set_ylabel('Estimate (K/day)')
    ax.set_title(r'Magnitude scaling, eq. (10)')
    # square figure limits
    xlim = (-20.4,-1.6)
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    
    #- linear fit
    a_fit , pcov = optimize.curve_fit(lambda x,a:a*x, x, y,p0=1)
    x_fit = np.linspace(xmin-xrange/40,xmax+xrange/40)
    y_fit = a_fit*x_fit
    # y_fit = 1*x_fit
    cov = np.cov(x,y)
    r_fit = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    
    # show
    ax.plot(x_fit,y_fit,'k:')
    # write numbers
    ax.text(0.05,0.05,'$Q_{rad}^{est} = %1.2f Q_{rad} $\n r=%1.2f'%(a_fit,r_fit),transform=ax.transAxes)
    
    # colorbar density
    axins1 = inset_axes(ax,
                    width="50%",  # width = 70% of parent_bbox width
                    height="2%",  # height : 5%
                    loc='lower right')
    cb = fig.colorbar(h, cax=axins1, orientation="horizontal")
    axins1.xaxis.set_ticks_position("top")
    axins1.tick_params(axis='x', labelsize=9)
    cb.set_label('Gaussian kernel density',labelpad=-34)
    
    #--- Add panel labeling
    pan_labs = '(a)','(b)','(c)','(d)'
    for ax,pan_lab in zip(axs.flatten(),pan_labs):
        ax.text(0.03,0.93,pan_lab,transform=ax.transAxes,fontsize=14)
    
    #--- save
    plt.savefig(os.path.join(figdir,'Figure2.pdf'),bbox_inches='tight')
    
        
#%% --- Figure 2 in column ---

    label_jump = '^\dagger'
    m_to_cm = 1e2
    day_to_seconds = 86400
    hPa_to_Pa = 1e2

    def scatterDensity(ax,x,y,s,alpha):
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        ax.scatter(x,y,c=z,s=s,alpha=0.4)
    

    fig,axs = plt.subplots(ncols=1,nrows=4,figsize=(4.5,20))
    plt.subplots_adjust(hspace=0.3,wspace=0.3)
    
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
        # pres_beta_over_p_peak = rad_scaling_all[day].rad_features.beta_over_p_peaks.pres_beta_over_p_peak
        # pres_scaling_profile_peak = rad_scaling_all[day].rad_features.scaling_profile_peaks.pres_scaling_profile_peak
        # pres_proxy_peak = pres_scaling_profile_peak
        pres_proxy_peak = pres_beta_peak
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
    ax.set_ylabel(r'$\beta$ peak height (hPa)')
    ax.set_title(r'Height as $\beta$ maximum')
    # square figure limits
    ylim = ax.get_ylim()
    ax.set_xlim(ylim)
    
        # ax.set_title(r'Approximating peak height\n as $p^\star = \arg\max_p \left(-\frac{g}{c_p}\frac{\beta}{p}\int B_\nu \phi_\nu\right)$')
    
    
    #-- (b) peak magnitude, using the approximation for the full profile, showing all profiles
    ax = axs[1]
    
    H_peak_all = {}
    qrad_peak_all = {}
    s = []
    
    days2show = days
    
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
            
            # beta
            beta = f.beta_peak[i_s]
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
    
    # # 1:1 line
    # x_ex = np.array([-18,-2])
    # ax.plot(x_ex,x_ex,'k:',alpha=0.4,label='1 : 1')
    
    ax.set_xlabel('True LW $Q_{rad}$ (K/day)')
    ax.set_ylabel('Estimate (K/day)')
    ax.set_title(r'Magnitude as $-\frac{g}{c_p} \frac{\beta%s}{p%s} \int B \phi d\nu$'%(label_jump,label_jump))
    # square figure limits
    xlim = (-20.4,-1.6)
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    
    #- linear fit
    xmin,xmax = np.min(x), np.max(x) 
    xrange = xmax-xmin
    a_fit , pcov = optimize.curve_fit(lambda x,a:a*x, x, y,p0=1)
    x_fit = np.linspace(xmin-xrange/40,xmax+xrange/40)
    y_fit = a_fit*x_fit
    cov = np.cov(x,y)
    r_fit = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    
    # show
    ax.plot(x_fit,y_fit,'k:')
    # write numbers
    ax.text(0.65,0.05,'$Q_{rad}^{est} = %1.2f Q_{rad} $\n r=%1.2f'%(a_fit,r_fit),transform=ax.transAxes)
    


    #-- (c) peak magnitude, using the intermediate approximation (beta and 1 wavenumber), showing all profiles
    ax = axs[2]
    
    H_peak_all = {}
    qrad_peak_all = {}
    s = []
    
    days2show = days
    
    for day in days2show:
        
        rs = rad_scaling_all[day]
        f = rad_scaling_all[day].rad_features
        k_bottom = np.where(np.logical_not(np.isnan(f.wp_z[0])))[0][0] 
        
        #- approximated peak
        Ns = rad_scaling_all[day].rad_features.pw.size
        H_peak = np.full(Ns,np.nan)
        
        for i_s in range(Ns):

            i_peak = f.i_lw_peak[i_s]
            p_peak = f.pres_lw_peak[i_s]
            
            # beta
            beta = f.beta_peak[i_s]
            # approximation of spectral integral
            piB_star = 0.0054 # (sum of piB in rotational and v-r bands)
            delta_nu = 120 # cm-1
            spec_int_approx = piB_star * delta_nu*m_to_cm/e
            # constants
            C = -gg/c_pd
            
            H_peak[i_s] = C/(p_peak*hPa_to_Pa) * beta * spec_int_approx * day_to_seconds
        
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

    # # 1:1 line
    # x_ex = np.array([-18,-2])
    # ax.plot(x_ex,x_ex,'k:',alpha=0.4,label='1 : 1')
    
    ax.set_xlabel('True LW $Q_{rad}$ (K/day)')
    ax.set_ylabel('Estimate (K/day)')
    ax.set_title(r'Magnitude as $-\frac{g}{c_p} \frac{\beta%s}{p%s} B_{\nu^\star} \frac{\Delta \nu}{e}$'%(label_jump,label_jump))
    # square figure limits
    xlim = (-20.4,-1.6)
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    
    #- linear fit
    a_fit , pcov = optimize.curve_fit(lambda x,a:a*x, x, y,p0=1)
    x_fit = np.linspace(xmin-xrange/40,xmax+xrange/40)
    y_fit = a_fit*x_fit
    # y_fit = 1*x_fit
    cov = np.cov(x,y)
    r_fit = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    
    # show
    ax.plot(x_fit,y_fit,'k:')
    # write numbers
    ax.text(0.65,0.05,'$Q_{rad}^{est} = %1.2f Q_{rad} $\n r=%1.2f'%(a_fit,r_fit),transform=ax.transAxes)
    
    
    #-- (d) peak magnitude, using the simplified scaling (RH step function and 1 wavenumber), showing all profiles
    ax = axs[3]
    
    H_peak_all = {}
    qrad_peak_all = {}
    s = []
    
    days2show = days
    
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
            piB_star = 0.0054
            delta_nu = 120 # cm-1
            spec_int_approx = piB_star * delta_nu*m_to_cm/e
            # constants
            alpha = 1.5
            C = -gg/c_pd * (1+alpha)
            
            H_peak[i_s] = C/(p_peak*hPa_to_Pa) * CRHbelow[i_peak]/CRHabove[i_peak] * spec_int_approx * day_to_seconds
        
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

    # # 1:1 line
    # x_ex = np.array([-18,-2])
    # ax.plot(x_ex,x_ex,'k:',alpha=0.4,label='1 : 1')
    
    ax.set_xlabel('True LW $Q_{rad}$ (K/day)')
    ax.set_ylabel('Estimate (K/day)')
    ax.set_title(r'Magnitude as $-\frac{g}{c_p} \frac{1}{p%s} (1+\alpha) \frac{CRH_s}{CRH_t} B_{\nu^\star} \frac{\Delta \nu}{e}$'%label_jump)
    # square figure limits
    xlim = (-20.4,-1.6)
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    
    #- linear fit
    a_fit , pcov = optimize.curve_fit(lambda x,a:a*x, x, y,p0=1)
    x_fit = np.linspace(xmin-xrange/40,xmax+xrange/40)
    y_fit = a_fit*x_fit
    # y_fit = 1*x_fit
    cov = np.cov(x,y)
    r_fit = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    
    # show
    ax.plot(x_fit,y_fit,'k:')
    # write numbers
    ax.text(0.65,0.05,'$Q_{rad}^{est} = %1.2f Q_{rad} $\n r=%1.2f'%(a_fit,r_fit),transform=ax.transAxes)
    
    #--- Add panel labeling
    pan_labs = '(a)','(b)','(c)','(d)'
    for ax,pan_lab in zip(axs,pan_labs):
        ax.text(0.03,0.93,pan_lab,transform=ax.transAxes,fontsize=14)
    
    # # colorbar density
    # axins1 = inset_axes(ax,
    #                     width="50%",  # width = 70% of parent_bbox width
    #                     height="2%",  # height : 5%
    #                     loc='lower right')
    # cb = fig.colorbar(h, cax=axins1, orientation="horizontal")
    # axins1.xaxis.set_ticks_position("top")
    # axins1.tick_params(axis='x', labelsize=9)
    # cb.set_label('Density',labelpad=-34)
    
    #--- save
    plt.savefig(os.path.join(figdir,'Figure2bis.pdf'),bbox_inches='tight')




    
    
#%% --- Figure 4, old version ---

i_fig = 4

names_4patterns = 'Fish','Flower','Gravel','Sugar'
days_4patterns = '20200122','20200202','20200205','20200209'
days_high_peaks = '20200213', '20200213', '20200211', '20200209', '20200209', '20200128'    

# Figure layout
fig = plt.figure(figsize=(5,7))

gs = GridSpec(5, 2, width_ratios=[1,1], height_ratios=[1,1,0.05,1,1],hspace=0,wspace=0.1)
ax1 = fig.add_subplot(gs[0],projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(gs[1],projection=ccrs.PlateCarree())
ax3 = fig.add_subplot(gs[2],projection=ccrs.PlateCarree())
ax4 = fig.add_subplot(gs[3],projection=ccrs.PlateCarree())
ax5 = fig.add_subplot(gs[3:5,:])

#--- (a-d) maps of patterns

axs = ax1,ax2,ax3,ax4
for ax,name_p,day_p in zip(axs,names_4patterns,days_4patterns):
    
    image = Image.open(os.path.join(workdir,'../images/patterns/PNG','GOES16__%s_1400.png'%day_p))
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
    pat_name = day_p[:4]+'-'+day_p[4:6]+'-'+day_p[6:8]+'\n'+name_p
    ax.text(0.98,0.98,pat_name,c='w',ha='right',va='top',transform=ax.transAxes,fontsize=10)
    ax.outline_patch.set_visible(False)
    
    rect = mpatches.Rectangle((0,0), width=1, height=1,edgecolor=col_pattern[name_p], facecolor="none",linewidth=3,alpha=1, transform=ax.transAxes)
    ax.add_patch(rect)


#--- (e) scatter of peak heights as a function of PW

ax = ax5

qrad_min = np.min([np.nanmin(np.absolute(rad_features_all[day].qrad_lw_peak)) for day in days])
qrad_max = np.max([np.nanmax(np.absolute(rad_features_all[day].qrad_lw_peak)) for day in days])

x = []
y = []
s = []

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
    
    # if conf == 'Low':
    #     c = col_pattern['']
    #     print('- Low confidence:')
    #     print(day,pat,c)
    # if conf == 'Medium':
    #     alpha = 0.3
    #     print('- Medium confidence:')
    #     print(day,pat,c)
    
    if day in days_high_peaks:
        alpha = 0.2
    
    ax.scatter(pw[k],z_peak[k],s=s[k],color=c,edgecolor='',alpha=alpha)
    
    ax.set_xlabel('PW (mm)')
    ax.set_ylabel(r'$z_{peak}$ (km)')
    ax.set_ylim((-0.3,8.1))

## Fully-manual legend QRAD
# (to the left)
# rect = mpatches.Rectangle((0.015,0.82), width=0.3, height=0.16,edgecolor='grey', facecolor="none",linewidth=0.5,alpha=0.5, transform=ax.transAxes)
# ax.add_patch(rect)
# for qp,y in zip([5,10,15],[0.94,0.89,0.84]):
#     s = 50*(qp/qrad_max)**2
#     circle = mlines.Line2D([0], [0], marker='o', color='w',
#                     markerfacecolor='r', markersize=s)
#     ax.scatter(0.05,y+0.01,s=s,c='k',edgecolor='',transform=ax.transAxes)
#     ax.text(0.1,y,s=r'$\vert Q_{rad}\vert >%d$ K/day'%qp,fontsize=7,transform=ax.transAxes)
# (to the right)
rect = mpatches.Rectangle((0.69,0.82), width=0.3, height=0.16,edgecolor='grey', facecolor="none",linewidth=0.5,alpha=0.5, transform=ax.transAxes)
ax.add_patch(rect)
for qp,y in zip([5,10,15],[0.94,0.89,0.84]):
    s = 50*(qp/qrad_max)**2
    circle = mlines.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor='r', markersize=s)
    ax.scatter(0.72,y+0.01,s=s,c='k',edgecolor='',transform=ax.transAxes)
    ax.text(0.76,y,s=r'$\vert Q_{rad}\vert >%d$ K/day'%qp,fontsize=7,transform=ax.transAxes)

## legend pattern
for pat in col_pattern.keys():
    print(pat)
    lab = pat
    if pat == '':
        lab = 'w/ moist instrusion'
    setattr(thismodule,"h_%s"%pat,mpatches.Patch(color=col_pattern[pat],linewidth=0,alpha=0.6,label=lab))
ax.legend(loc='lower center',handles=[h_Fish,h_Flower,h_Gravel,h_Sugar,h_],ncol=5,fontsize=6)

#--- Add panel labeling
pan_labs = '(a)','(b)','(c)','(d)','(e)'
pan_cols = 'w','w','w','w','k'
axs = ax1,ax2,ax3,ax4,ax5
for ax,pan_lab,pan_col in zip(axs,pan_labs,pan_cols):
    ax.text(0.02,0.98,pan_lab,c=pan_col,ha='left',va='top',
            transform=ax.transAxes,fontsize=12)

#--- save
plt.savefig(os.path.join(figdir,'Figure%d_old.pdf'%i_fig),bbox_inches='tight')
plt.savefig(os.path.join(figdir,'Figure%d_old.png'%i_fig),dpi=300,bbox_inches='tight')




#%% --- Figure 4, new, with magnitude afo PW ---

i_fig = 4

names_4patterns = 'Fish','Flower','Gravel','Sugar'
days_4patterns = '20200122','20200202','20200205','20200209'
days_high_peaks = '20200213', '20200211', '20200209','20200128', '20200124', '20200122'
z_min_all = 4, 4, 3.5, 4, 3, 3.2
z_max_all = 9, 6, 8.5, 6, 4.5, 4

# Figure layout
fig = plt.figure(figsize=(5,11))

gs = GridSpec(6, 2, width_ratios=[1,1], height_ratios=[1,1,0.05,2,0.42,2],hspace=0,wspace=0.1)
ax1 = fig.add_subplot(gs[0],projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(gs[1],projection=ccrs.PlateCarree())
ax3 = fig.add_subplot(gs[2],projection=ccrs.PlateCarree())
ax4 = fig.add_subplot(gs[3],projection=ccrs.PlateCarree())
ax5 = fig.add_subplot(gs[3:4,:])
ax6 = fig.add_subplot(gs[5:6,:])

#--- (a-d) maps of patterns

axs = ax1,ax2,ax3,ax4
for ax,name_p,day_p in zip(axs,names_4patterns,days_4patterns):
    
    image = Image.open(os.path.join(workdir,'../images/patterns/PNG','GOES16__%s_1400.png'%day_p))
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
    str_ = day_p[:4]+'-'+day_p[4:6]+'-'+day_p[6:8]+'\n'+name_p
    ax.text(0.98,0.98,str_,c='w',ha='right',va='top',transform=ax.transAxes,fontsize=10)
    ax.outline_patch.set_visible(False)
    
    # change frame color
    rect = mpatches.Rectangle((0,0), width=1, height=1,edgecolor=col_pattern[name_p], facecolor="none",linewidth=3,alpha=1, transform=ax.transAxes)
    ax.add_patch(rect)


#--- (e) scatter of peak heights as a function of PW

ax = ax5

qrad_min = np.min([np.nanmin(np.absolute(rad_features_all[day].qrad_lw_peak)) for day in days])
qrad_max = np.max([np.nanmax(np.absolute(rad_features_all[day].qrad_lw_peak)) for day in days])

x = []
y = []
s = []

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
    
    # if conf == 'Low':
    #     c = col_pattern['']
    #     print('- Low confidence:')
    #     print(day,pat,c)
    # if conf == 'Medium':
    #     alpha = 0.3
    #     print('- Medium confidence:')
    #     print(day,pat,c)
    
    if day in days_high_peaks:
        i_d = np.where(np.array(days_high_peaks) == day)[0][0]
        alpha_high = 0.2
        k_high = np.logical_and(z_peak[k] <= z_max_all[i_d],
                                z_peak[k] > z_min_all[i_d])
        k_low = np.logical_not(k_high)

        ax.scatter(pw[k][k_high],z_peak[k][k_high],s=s[k][k_high],color=c,edgecolor='',alpha=alpha_high)
        ax.scatter(pw[k][k_low],z_peak[k][k_low],s=s[k][k_low],color=c,edgecolor='',alpha=alpha)
        
    else:
        ax.scatter(pw[k],z_peak[k],s=s[k],color=c,edgecolor='',alpha=alpha)
    
    pass

ax.set_xlabel('PW (mm)')
ax.set_ylabel(r'Peak height (km)',labelpad=20)
ax.set_ylim((-0.3,8.1))


#--- (f) scatter of peak magnitudes as a function of PW

ax = ax6

pw_max = 45 # mm
z_max = 3.2 # km
rh_cloud = 0.95

pw_by_pattern = {'Fish':[],
                   'Flower':[],
                   'Gravel':[],
                   'Sugar':[]}
qrad_by_pattern = {'Fish':[],
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
    
    # circle diameter proportional to size of peak
    # s = 50*(qrad_peak/qrad_max)**2

    # store data
    pw_by_pattern[pat].extend(list(pw[k]))
    qrad_by_pattern[pat].extend(list(qrad_peak[k]))
    
    # is cloud ?
    data_day = data_all.sel(launch_time=day)
    iscloud_day = np.any(data_day.relative_humidity > rh_cloud,axis=1).data
    iscloud_by_pattern[pat].extend(list(iscloud_day[k]))
    
    # rh
    rh_prof = data_day.relative_humidity.data
    rh_save = np.array([rh_prof[:,i_z][k] for i_z in range(rh_prof.shape[1])])
    rh_save  = np.swapaxes(rh_save,0,1)
    rh_clear_by_pattern[pat].append(rh_save)

def piecewise_linear(x,x_breaks,y_breaks):
    
    N_breaks = len(x_breaks)
    
    cond_list = [np.logical_and(x >= x_breaks[i],x <= x_breaks[i+1]) for i in range(N_breaks-1)]
    def make_piece(k):
        def f(x):
            return (x-x_breaks[k])/(x_breaks[k+1]-x_breaks[k])*(y_breaks[k+1]-y_breaks[k]) + y_breaks[k]
        return f 
    func_list = [make_piece(k) for k in range(N_breaks-1)]
    
    print('N_breaks =',N_breaks)
    print('N_cond =',len(cond_list))
    print('N_func =',len(func_list))
    
    return np.piecewise(x,cond_list,func_list)

def piecewise_fit(x,y,x_breaks_0,y_breaks_0):

    N_breaks = len(x_breaks_0)
    
    def piecewise_fun(x,*p):
        return piecewise_linear(x,[x_breaks_0[0]]+list(p[0:N_breaks-2])+[x_breaks_0[-1]],
                                p[N_breaks-2:2*N_breaks-2])

    mask = ~np.isnan(x) & ~np.isnan(y)
    p0 = tuple(list(x_breaks_0[1:-1])+list(y_breaks_0))
    p , e = optimize.curve_fit(piecewise_fun, x[mask], y[mask],p0=p0)
    print('N_breaks is',N_breaks)
    print('p0 is',p0)
    print('p fitted is',p)

    return p


#- add information to guide the eye
for pat in qrad_by_pattern.keys():
    print(pat,np.size(qrad_by_pattern[pat]))
    
    # data
    x = np.array(pw_by_pattern[pat])
    y = np.array(qrad_by_pattern[pat])
    # x_inds = x.argsort()
    # x = x[x_inds[::1]]
    # y = y[x_inds[::1]]
    c = col_pattern[pat]
    
    #- show points
    # marker
    isclear = np.logical_not(iscloud_by_pattern[pat])

    iscloud = iscloud_by_pattern[pat]
    # show clear first
    ax.scatter(x[isclear],y[isclear],s=10,color=c,edgecolor='',alpha=0.6)
    # show cloudy
    ax.scatter(x[iscloud],y[iscloud],s=10,marker='o',color='',edgecolor=c,alpha=0.6)


    #- then show fit
    y_mean = np.mean(y[isclear])
    x5,x25,x75,x95 = np.percentile(x[isclear],[5,25,75,95])
    ax.plot([x5,x95],[y_mean,y_mean],c=c,linewidth=0.8)
    ax.plot([x25,x75],[y_mean,y_mean],c=c,linewidth=3)
    # ax.boxplot(x[isclear],notch=True)

    # # piecewise linear fit
    # x_min = np.min(x)
    # x_max = np.max(x)
    # x_breaks_0 = [x_min,28,x_max]
    # y_breaks_0 = [-6,-6,-6]
    # N_breaks = len(x_breaks_0)
    # p = piecewise_fit(x,y,x_breaks_0,y_breaks_0)
    # x_fit = np.linspace(x_min,x_max,50)
    # y_fit = piecewise_linear(x_fit,[x_fit[0],p[0:N_breaks-2],x_fit[-1]],
    #                             p[N_breaks-2:2*N_breaks-2])
    # ax.plot(x_fit,y_fit,c=c)
    
    # # fit (piecewise linear for Fish, linear otherwise)
    # if pat in ['Fish',]:
    #     x_min = np.min(x)
    #     x_max = np.max(x)
    #     x_breaks_0 = [x_min,28,x_max]
    #     y_breaks_0 = [-6,-6,-6]
    #     N_breaks = len(x_breaks_0)
    #     p = piecewise_fit(x,y,x_breaks_0,y_breaks_0)
    #     x_fit = np.linspace(x_min,x_max,50)
    #     y_fit = piecewise_linear(x_fit,[x_fit[0],p[0:N_breaks-2],x_fit[-1]],
    #                                 p[N_breaks-2:2*N_breaks-2])
    #     print(y_fit)
    # else:
    #     x_fit = np.linspace(np.min(x),np.max(x),50)
    #     y_fit = np.poly1d(np.polyfit(x, y, 1))(x_fit)

    # ax.plot(x_fit,y_fit,c=c)

    # # fitting ellipses
    # if pat == 'Fish':
    #     k1 = x < 28 # mm
    #     k2 = x > 28 # mm
    #     for k in (k1,k2):
    #         ell = confidence_ellipse(x[k], y[k], n_std=1,color=c,edgecolor='',alpha=0.15)
    #         ax.add_patch(ell)
    # else:
    #     ell = confidence_ellipse(x, y, n_std=1,color=c,edgecolor='',alpha=0.15)
    #     ax.add_patch(ell)
    
    # # binned averages
    # x_bins = np.linspace(20,44,13)
    # x_centers = np.convolve(x_bins,[0.5,0.5],mode='valid')
    # y_mean = np.full((len(x_centers),),np.nan)
    # for i in range(len(x_centers)):
    #     k_xi = np.logical_and(x > x_bins[i],x <= x_bins[i+1])
    #     if np.sum(k_xi) >= 4:
    #         y_mean[i] = np.nanmean(y[k_xi])
        
    # ax.plot(x_centers,y_mean,c=c)
    
    # # moving averages
    # x_centers = np.linspace(21,41,21)
    # y_mean = np.full((len(x_centers),),np.nan)
    # for i in range(len(x_centers)):
    #     k_xi = np.logical_and(x > x_centers[i]-3,x <= x_centers[i]+3)
    #     if np.sum(k_xi) >= 4:
    #         y_mean[i] = np.nanmean(y[k_xi])
        
    # ax.plot(x_centers,y_mean,c=c)
    
    # # linear fit
    # if pat == 'Fish':
    #     k1 = x < 28 # mm
    #     k2 = x > 28 # mm
    #     for k in (k1,k2):
    #         x_fit = np.linspace(np.min(x[k]),np.max(x[k]),50)
    #         y_fit = np.poly1d(np.polyfit(x[k], y[k], 1))(x_fit)
    #         ax.plot(x_fit,y_fit,c=c)
    # else:
    #     x_fit = np.linspace(np.min(x),np.max(x),50)
    #     y_fit = np.poly1d(np.polyfit(x, y, 1))(x_fit)
    #     ax.plot(x_fit,y_fit,c=c)
    
    # # quadratic fit
    # if pat in ['Fish','Flower']:
    #     deg = 2
    # else:
    #     deg = 1
    # x_fit = np.linspace(np.min(x),np.max(x),50)
    # y_fit = np.poly1d(np.polyfit(x, y, deg))(x_fit)
    # ax.plot(x_fit,y_fit,c=c)
    
    



ax.set_xlabel('PW (mm)')
ax.set_ylabel(r'Peak magnitude (K/day)')
ax.set_ylim((-14.1,-4.3))

#- add zoom effect
zoom_effect_xaxis(ax6, ax5)

# change frame color
# for side in ('left','right'):
#     ax.spines[side].set_visible(False)
ax.patch.set_visible(False)
rect = mpatches.Rectangle((0,0), width=1, height=1,edgecolor='k', facecolor="none",linestyle='--',linewidth=1,alpha=1, transform=ax.transAxes)
ax.add_patch(rect)


#--- (e) legends
ax = ax5

## Fully-manual legend QRAD
rect = mpatches.Rectangle((0.69,0.82), width=0.3, height=0.16,edgecolor='grey', facecolor="none",linewidth=0.5,alpha=1, transform=ax.transAxes)
ax.add_patch(rect)
for qp,y in zip([5,10,15],[0.94,0.89,0.84]):
    s = 50*(qp/qrad_max)**2
    circle = mlines.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor='r', markersize=s)
    ax.scatter(0.72,y+0.01,s=s,c='k',edgecolor='',transform=ax.transAxes)
    ax.text(0.76,y,s=r'$\vert Q_{rad}\vert >%d$ K/day'%qp,fontsize=7,transform=ax.transAxes)

## legend pattern
for pat in col_pattern.keys():
    print(pat)
    lab = pat
    if pat == '':
        lab = 'w/ moist instrusion'
    setattr(thismodule,"h_%s"%pat,mpatches.Patch(color=col_pattern[pat],linewidth=0,alpha=0.6,label=lab))
ax.legend(loc='lower center',handles=[h_Fish,h_Flower,h_Gravel,h_Sugar,h_],ncol=5,fontsize=6)

#--- (f) legends
ax = ax6

## Fully-manual legend cloudy-clear
circle_full = mlines.Line2D([0], [0], marker='o', color='none',markeredgecolor='none',
                    markerfacecolor='k',markersize=5,label='cloud-free')
circle_open = mlines.Line2D([0], [0], marker='o', color='none',markeredgecolor='k',
                    markerfacecolor='none',markersize=5,label='cloudy air')
ax.legend(handles=[circle_full,circle_open],fontsize=8)


# # show clear first
#     ax.scatter(x[isclear],y[isclear],s=10,color=c,edgecolor='',alpha=0.6)
#     # show cloudy
#     ax.scatter(x[iscloud],y[iscloud],s=10,marker='o',color='',edgecolor=c,alpha=0.6)


#--- Add panel labeling
pan_labs = '(a)','(b)','(c)','(d)','(e)','(f)'
pan_cols = 'w','w','w','w','k','k'
axs = ax1,ax2,ax3,ax4,ax5,ax6
for ax,pan_lab,pan_col in zip(axs,pan_labs,pan_cols):
    ax.text(0.02,0.98,pan_lab,c=pan_col,ha='left',va='top',
            transform=ax.transAxes,fontsize=12)

#--- save
plt.savefig(os.path.join(figdir,'Figure%d.pdf'%i_fig),bbox_inches='tight')
plt.savefig(os.path.join(figdir,'Figure%d.png'%i_fig),dpi=300,bbox_inches='tight')



#%% --- Figure 3 functions ---

def z2p(z_0,z,pres):
    """Assume z is increasing"""
    
    i_z = np.where(z>=z_0)[0][0]
    
    return pres[i_z]

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

def waterPath(qv,pres,p_bottom,p_top):

    p_increasing = np.diff(pres)[0] > 0

    if p_increasing:

        arr = qv
        p = pres

    else:

        arr = np.flip(qv)
        p = np.flip(pres)

    p0 = p_top
    p1 = p_bottom

    return mo.pressureIntegral(arr=arr,pres=p,p0=p0,p1=p1)


#%% --- Figure 3 ---

i_fig = 3

hPa_to_Pa = 1e2

#--- show

# Figure layout
fig = plt.figure(figsize=(13.5,4.5))

gs = GridSpec(1, 3, width_ratios=[1.5,1.5,3], height_ratios=[1],hspace=0.25,wspace=0.3)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

# peak height
# z_jump = 1.66883978
z_jump = moist_intrusions['20200213, lower']['fit']['z_breaks_id'][0]
z = radprf_MI_20200213.zlay[0]/1e3 # km
k_jump = np.where(z>=z_jump)[0][0]
pres_jump = radprf_MI_20200213.play[k_jump].data/1e2 # hPa

labs = 'no intrusion','fitted from data','homogenized, same W','idealized, same W'

#-- (a) and (b)
# radprf_ab = radprf_MI_20200213
# i_ref = 4
# i_mi = 2
# i_rectRH = 4
# i_homoRH = 3
radprf = radprf_MI_20200213lower
i_ref = 1
i_mi = 2
i_rectRH = 3
i_homoRH = 4

for ax, varid in zip((ax1,ax2),('rh','q_rad_lw')):
    
    #- peak level
    ax.axhline(z_jump,c='darkgoldenrod',linestyle='--',linewidth=0.8,alpha=0.8)
        
    #- reference with intrusion (from piecewise linear fit)
    z = np.array(radprf.zlay[i_ref]/1e3) # km
    var_mi = radprf[varid].data[i_mi]
    # show
    if ax == ax1:
        lab = labs[1]
    ax.plot(var_mi,z,'k',label=lab)

    #- reference without intrusion (from piecewise linear fit)
    z = np.array(radprf.zlay[i_ref]/1e3) # km
    var_ref = radprf[varid].data[i_ref]
    # show
    if ax == ax1:
        lab = labs[0]
    ax.plot(var_ref,z,'grey',label=lab)
    
    #- rectangle-RH intrusion, same water path, same height
    z = np.array(radprf.zlay[i_ref]/1e3) # km
    var_rectRH = radprf[varid].data[i_rectRH]
    # show
    if ax == ax1:
        lab = labs[3]
    ax.plot(var_rectRH,z,'b',label=lab)
    
    #- homogenized rh, same water path at peak level
    z = np.array(radprf.zlay[i_ref]/1e3) # km
    var_homoRH = radprf[varid].data[i_homoRH]
    # show
    if ax == ax1:
        lab = labs[2]
    ax.plot(var_homoRH,z,'b--',label=lab)
    


ax1.set_ylabel('z (km)')
ax2.set_ylabel('z (km)')
ax1.set_xlabel('Relative humidity')
ax2.set_xlabel(r'LW $Q_{rad}$ (K/day)')

ax1.set_ylim((-0.15,10.15))
ax2.set_ylim((0,2.3))
ax2.set_xlim((-13.1,0.1))

ax1.legend(fontsize=7,loc='upper right')

#- connecting (a) and (b)
zoom_effect_yaxis(ax2, ax1)


#-- (c) intrusions at all heights and water paths
ax = ax3
# radprf = radprf_RMI_20200213
radprf = radprf_MI_20200213lower

#- get data at peak

Nsample = 20

# indices in file
inds_id = slice(1,6)
inds_uniformRH = slice(6,26)
inds_varyWandRH = slice(26,None)

# reference 
qvstar = radprf.qv[i_ref]/radprf.rh[i_ref]
i_levmax = np.where(np.isnan(qvstar))[0][0] # assuming data is ordered from bottom to top
p_levmax = pres[i_levmax]
z_jump = moist_intrusions['20200213, lower']['fit']['z_breaks_id'][0]
z_jump_FT = moist_intrusions['20200213, lower']['fit']['z_breaks_id'][1]
i_jump = np.where(z>z_jump)[0][0]
i_jump_FT = np.where(z>z_jump_FT)[0][0]
p_jump = z2p(z_jump,z,pres) # hPa
play = radprf_MI_20200213.play/hPa_to_Pa # hPa
# W_ref = waterPath(radprf.qv[i_ref],play,p_jump,p_levmax)
# coordinates
W_all = [float(str(radprf.name[inds_uniformRH][i].data)[2:6]) for i in range(Nsample)]
# W_all = np.array(W_all)-W_ref
H_all = [float(str(radprf.name[inds_uniformRH.stop:inds_uniformRH.stop+Nsample][i].data)[11:15]) for i in range(Nsample)]

# peak magnitudes
qradlw_peak = np.full((Nsample,Nsample),np.nan)
for i_W in range(Nsample):
    for i_H in range(Nsample):
        
        W = W_all[i_W]#+W_ref
        H = H_all[i_H]
        name = 'W_%2.2fmm_H_%1.2fkm'%(W,H)
        i_prof = np.where(np.isin(radprf.name.data,name))[0][0]
        
        qradlw_peak[i_W,i_H] = radprf.q_rad_lw[i_prof,k_jump].data

qradlw_peak_ref = radprf['q_rad_lw'].data[i_ref].data[i_jump]
delta_qradlw_peak = qradlw_peak-qradlw_peak_ref

#- filled and unfilled contours

qrad_2_show = delta_qradlw_peak.T
cmap = plt.cm.ocean
vmin = np.min(qrad_2_show)
vmax = np.max(qrad_2_show)
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
ax.contourf(W_all,H_all,qrad_2_show,levels=30,cmap=cmap,vmin=vmin,vmax=vmax)
cont = ax.contour(W_all,H_all,qrad_2_show,levels=np.linspace(2,6,3),colors=('grey',),
                  linestyles=('-',),linewidths=(0.8,),vmin=vmin,vmax=vmax)
plt.clabel(cont, fmt = '%2.1d', colors = 'grey', fontsize=10) #contour line labels


#- position for data on the left
qv_ref = radprf_MI_20200213['h2o'].data[i_ref]
pres_ref = radprf_MI_20200213['play'].data/1e2
W_ref = computeWPaboveZ(qv_ref,pres_ref,0)
qv_rectRH = radprf_RMI_20200213['h2o'].data[i_rectRH]
pres_rectRH = radprf_RMI_20200213['play'].data/1e2
W_rectRH = computeWPaboveZ(qv_rectRH,pres_rectRH,0)
W_mi_alone = W_rectRH - W_ref
print(W_rectRH[k_jump],W_ref[k_jump],W_mi_alone[k_jump])
# FINISH - put white dot

# Each day's intrusion
for daylab in moist_intrusions.keys():
    
    W_int = moist_intrusions[daylab]['stats']['W_int']
    z_int_bottom = moist_intrusions[daylab]['stats']['z_int_bottom']
    z_int_center = moist_intrusions[daylab]['stats']['z_int_center']
    z_int_top = moist_intrusions[daylab]['stats']['z_int_top']
    
    # ax.plot([W_int,W_int],[z_int_bottom,z_int_top],linewidth=3,color='cornsilk')
    # ax.scatter(W_int,z_int_center,marker='_',color='k',edgecolor='none')
    # r_width = 0.05
    # rect_int = mpatches.Rectangle((W_int,z_int_bottom), r_width, z_int_top-z_int_bottom, ec=None,fc='cornsilk')
    # ax.add_patch(rect_int)
    
    # if daylab in ['20200209','20200211']:
    #     ax.scatter(W_int+0.1,z_int_center,marker='>',c='cornsilk',edgecolor='none')
    #     ax.text(W_int+0.32,z_int_center,' '+daylab,c='cornsilk',ha='right',va='bottom',rotation='vertical')
    # else:
    #     ax.scatter(W_int-0.04,z_int_center,marker='<',c='cornsilk',edgecolor='none')
    #     ax.text(W_int,z_int_center,' '+daylab,c='cornsilk',ha='right',va='bottom',rotation='vertical')
    if daylab in ['20200209','20200211']:
        ax.scatter(W_int,z_int_center,marker='o',c='cornsilk',edgecolor='none')
        ax.text(W_int+0.32,z_int_center,' '+daylab,c='cornsilk',ha='right',va='bottom',rotation='vertical')
    elif daylab in ['20200128','20200213, upper']:
        ax.scatter(W_int,z_int_center,marker='o',c='cornsilk',edgecolor='none')
        ax.text(W_int,z_int_center,' '+daylab,c='cornsilk',ha='right',va='bottom',rotation='vertical')
    else:
        ax.scatter(W_int,z_int_center,marker='o',c='cornsilk',edgecolor='blue')
        ax.text(W_int,z_int_center,' '+daylab,c='cornsilk',ha='right',va='bottom',rotation='vertical')
    
# Equivalent height of cooling peak reduction by a uniform increas in RH
# (the center height of a rectangle intrusion that gives the same peak reduction)
H_equiv = np.zeros((Nsample,))
for i_W in range(Nsample):
        
    W = W_all[i_W]#+W_ref
    name = 'W_%2.2fmm_uniform_RH'%(W)
    i_prof = np.where(np.isin(radprf.name.data,name))[0][0]
    qradlw_peak_uRH = radprf.q_rad_lw[i_prof,k_jump].data
    # print(i_prof,qradlw_peak_uRH)
    delta_qradlw_uRH = qradlw_peak_uRH - qradlw_peak_ref
    
    i_H = np.where(delta_qradlw_peak[i_W,:] >= delta_qradlw_uRH)[0][0]
    H_equiv[i_W] = H_all[i_H]

ax.plot(W_all,H_equiv,'r')

# Center of mass of a uniform free-tropospheric RH (only depends on the shape of qvstar)
W_qvstar = computeWPaboveZ(qvstar,pres[:-1],p_levmax)
i_z = np.where(W_qvstar >= W_qvstar[i_jump_FT]/2)[0][0]
z_center_uRH = z[i_z]
i_H = np.where(H_all >= z_center_uRH)[0][0]
H_qvstar_center = H_all[i_H]
H_uniform = H_qvstar_center*np.ones((Nsample,))

ax.plot(W_all,H_uniform,'r')

# print(H_equiv, H_uniform)

# labels
ax.set_xlabel('Intrusion water path (mm)')
ax.set_ylabel('Intrusion center level (km)')
ax.set_ylim([H_all[0],H_all[-1]])

# colorbar
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),
                 ax=ax,shrink=0.99,pad=0.04)
cb.set_label(r'$\Delta Q_{rad,LW}$ peak (K/day)')

ax.set_title(r'Lower peak reduction (at %1.2f km)'%(z_jump))



#--- Add panel labeling
pan_labs = '(a)','(b)','(c)'
pan_cols = 'k','k','w'
axs = ax1,ax2,ax3
for ax,pan_lab,pan_col in zip(axs,pan_labs,pan_cols):
    t = ax.text(0.04,0.02,pan_lab,c=pan_col,ha='left',va='bottom',
            transform=ax.transAxes,fontsize=14)
    # if ax != ax3:
        # t.set_bbox(dict(facecolor='w', alpha=0.8, edgecolor='w'))

#--- save
plt.savefig(os.path.join(figdir,'Figure%d.pdf'%i_fig),bbox_inches='tight')
plt.savefig(os.path.join(figdir,'Figure%d.png'%i_fig),dpi=300,bbox_inches='tight')









#%% --- Figure 3 with spectral figure ---

i_fig = 3

# Figure layout
fig = plt.figure(figsize=(12,8))

gs = GridSpec(2, 3, width_ratios=[1.5,1.5,3], height_ratios=[3,2],hspace=0.25,wspace=0.3)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])
ax4 = fig.add_subplot(gs[1,:])



#--- show



#-- (a) and (b)
# radprf_ab = radprf_MI_20200213
# i_ref = 4
# i_mi = 2
# i_rectRH = 4
# i_homoRH = 3
radprf = radprf_MI_20200213lower
i_ref = 1
i_mi = 2
i_rectRH = 3
i_homoRH = 4

for ax, varid in zip((ax1,ax2),('rh','q_rad_lw')):
    
    #- peak level
    ax.axhline(z_jump,c='darkgoldenrod',linestyle='--',linewidth=0.8,alpha=0.8)
        
    #- reference with intrusion (from piecewise linear fit)
    z = np.array(radprf.zlay[i_ref]/1e3) # km
    var_mi = radprf[varid].data[i_mi]
    # show
    if ax == ax1:
        lab = labs[1]
    ax.plot(var_mi,z,'k',label=lab)

    #- reference without intrusion (from piecewise linear fit)
    z = np.array(radprf.zlay[i_ref]/1e3) # km
    var_ref = radprf[varid].data[i_ref]
    # show
    if ax == ax1:
        lab = labs[0]
    ax.plot(var_ref,z,'grey',label=lab)
    
    #- rectangle-RH intrusion, same water path, same height
    z = np.array(radprf.zlay[i_ref]/1e3) # km
    var_rectRH = radprf[varid].data[i_rectRH]
    # show
    if ax == ax1:
        lab = labs[3]
    ax.plot(var_rectRH,z,'b',label=lab)
    
    #- homogenized rh, same water path at peak level
    z = np.array(radprf.zlay[i_ref]/1e3) # km
    var_homoRH = radprf[varid].data[i_homoRH]
    # show
    if ax == ax1:
        lab = labs[2]
    ax.plot(var_homoRH,z,'b--',label=lab)
    


ax1.set_ylabel('z (km)')
ax2.set_ylabel('z (km)')
ax1.set_xlabel('Relative humidity')
ax2.set_xlabel(r'LW $Q_{rad}$ (K/day)')

ax1.set_ylim((-0.15,10.15))
ax2.set_ylim((0,2.3))
ax2.set_xlim((-13.1,0.1))

ax1.legend(fontsize=7,loc='upper right')

#- connecting (a) and (b)
zoom_effect_yaxis(ax2, ax1)


#-- (c) intrusions at all heights and water paths
ax = ax3
# radprf = radprf_RMI_20200213
radprf = radprf_MI_20200213lower

#- get data at peak

Nsample = 20

# indices in file
inds_id = slice(1,6)
inds_uniformRH = slice(6,26)
inds_varyWandRH = slice(26,None)

# reference 
qvstar = radprf.qv[i_ref]/radprf.rh[i_ref]
i_levmax = np.where(np.isnan(qvstar))[0][0] # assuming data is ordered from bottom to top
p_levmax = pres[i_levmax]
z_jump = moist_intrusions['20200213, lower']['fit']['z_breaks_id'][0]
z_jump_FT = moist_intrusions['20200213, lower']['fit']['z_breaks_id'][1]
i_jump = np.where(z>z_jump)[0][0]
i_jump_FT = np.where(z>z_jump_FT)[0][0]
p_jump = z2p(z_jump,z,pres) # hPa
play = radprf_MI_20200213.play/hPa_to_Pa # hPa
# W_ref = waterPath(radprf.qv[i_ref],play,p_jump,p_levmax)
# coordinates
W_all = [float(str(radprf.name[inds_uniformRH][i].data)[2:6]) for i in range(Nsample)]
# W_all = np.array(W_all)-W_ref
H_all = [float(str(radprf.name[inds_uniformRH.stop:inds_uniformRH.stop+Nsample][i].data)[11:15]) for i in range(Nsample)]

# peak magnitudes
qradlw_peak = np.full((Nsample,Nsample),np.nan)
for i_W in range(Nsample):
    for i_H in range(Nsample):
        
        W = W_all[i_W]#+W_ref
        H = H_all[i_H]
        name = 'W_%2.2fmm_H_%1.2fkm'%(W,H)
        i_prof = np.where(np.isin(radprf.name.data,name))[0][0]
        
        qradlw_peak[i_W,i_H] = radprf.q_rad_lw[i_prof,k_jump].data

qradlw_peak_ref = radprf['q_rad_lw'].data[i_ref].data[i_jump]
delta_qradlw_peak = qradlw_peak-qradlw_peak_ref

#- filled contour

qrad_2_show = delta_qradlw_peak.T
cmap = plt.cm.ocean
vmin = np.min(qrad_2_show)
vmax = np.max(qrad_2_show)
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
ax.contourf(W_all,H_all,qrad_2_show,levels=30,cmap=cmap,vmin=vmin,vmax=vmax)
cont = ax.contour(W_all,H_all,qrad_2_show,levels=np.linspace(2,6,3),colors=('grey',),
                  linestyles=('-',),linewidths=(0.8,),vmin=vmin,vmax=vmax)
plt.clabel(cont, fmt = '%2.1d', colors = 'grey', fontsize=10) #contour line labels


#- position for data on the left
qv_ref = radprf_MI_20200213['h2o'].data[i_ref]
pres_ref = radprf_MI_20200213['play'].data/1e2
W_ref = computeWPaboveZ(qv_ref,pres_ref,0)
qv_rectRH = radprf_RMI_20200213['h2o'].data[i_rectRH]
pres_rectRH = radprf_RMI_20200213['play'].data/1e2
W_rectRH = computeWPaboveZ(qv_rectRH,pres_rectRH,0)
W_mi_alone = W_rectRH - W_ref
print(W_rectRH[k_jump],W_ref[k_jump],W_mi_alone[k_jump])
# FINISH - put white dot

# Each day's intrusion
for daylab in moist_intrusions.keys():
    
    W_int = moist_intrusions[daylab]['stats']['W_int']
    z_int_bottom = moist_intrusions[daylab]['stats']['z_int_bottom']
    z_int_center = moist_intrusions[daylab]['stats']['z_int_center']
    z_int_top = moist_intrusions[daylab]['stats']['z_int_top']
    
    # ax.plot([W_int,W_int],[z_int_bottom,z_int_top],linewidth=3,color='cornsilk')
    # ax.scatter(W_int,z_int_center,marker='_',color='k',edgecolor='none')
    # r_width = 0.05
    # rect_int = mpatches.Rectangle((W_int,z_int_bottom), r_width, z_int_top-z_int_bottom, ec=None,fc='cornsilk')
    # ax.add_patch(rect_int)
    
    # if daylab in ['20200209','20200211']:
    #     ax.scatter(W_int+0.1,z_int_center,marker='>',c='cornsilk',edgecolor='none')
    #     ax.text(W_int+0.32,z_int_center,' '+daylab,c='cornsilk',ha='right',va='bottom',rotation='vertical')
    # else:
    #     ax.scatter(W_int-0.04,z_int_center,marker='<',c='cornsilk',edgecolor='none')
    #     ax.text(W_int,z_int_center,' '+daylab,c='cornsilk',ha='right',va='bottom',rotation='vertical')
    if daylab in ['20200209','20200211']:
        ax.scatter(W_int,z_int_center,marker='o',c='cornsilk',edgecolor='none')
        ax.text(W_int+0.32,z_int_center,' '+daylab,c='cornsilk',ha='right',va='bottom',rotation='vertical')
    elif daylab in ['20200128','20200213, upper']:
        ax.scatter(W_int,z_int_center,marker='o',c='cornsilk',edgecolor='none')
        ax.text(W_int,z_int_center,' '+daylab,c='cornsilk',ha='right',va='bottom',rotation='vertical')
    else:
        ax.scatter(W_int,z_int_center,marker='o',c='cornsilk',edgecolor='blue')
        ax.text(W_int,z_int_center,' '+daylab,c='cornsilk',ha='right',va='bottom',rotation='vertical')
    
# Equivalent height of cooling peak reduction by a uniform increas in RH
# (the center height of a rectangle intrusion that gives the same peak reduction)
H_equiv = np.zeros((Nsample,))
for i_W in range(Nsample):
        
    W = W_all[i_W]#+W_ref
    name = 'W_%2.2fmm_uniform_RH'%(W)
    i_prof = np.where(np.isin(radprf.name.data,name))[0][0]
    qradlw_peak_uRH = radprf.q_rad_lw[i_prof,k_jump].data
    # print(i_prof,qradlw_peak_uRH)
    delta_qradlw_uRH = qradlw_peak_uRH - qradlw_peak_ref
    
    i_H = np.where(delta_qradlw_peak[i_W,:] >= delta_qradlw_uRH)[0][0]
    H_equiv[i_W] = H_all[i_H]

ax.plot(W_all,H_equiv,'r')

# Center of mass of a uniform free-tropospheric RH (only depends on the shape of qvstar)
W_qvstar = computeWPaboveZ(qvstar,pres[:-1],p_levmax)
i_z = np.where(W_qvstar >= W_qvstar[i_jump_FT]/2)[0][0]
z_center_uRH = z[i_z]
i_H = np.where(H_all >= z_center_uRH)[0][0]
H_qvstar_center = H_all[i_H]
H_uniform = H_qvstar_center*np.ones((Nsample,))

ax.plot(W_all,H_uniform,'r')

# print(H_equiv, H_uniform)

# labels
ax.set_xlabel('Intrusion water path (mm)')
ax.set_ylabel('Intrusion center level (km)')
ax.set_ylim([H_all[0],H_all[-1]])

# colorbar
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),
                 ax=ax,shrink=0.99,pad=0.04)
cb.set_label(r'$\Delta Q_{rad,LW}$ peak (K/day)')

ax.set_title(r'Lower peak reduction (at %1.2f km)'%(z_jump))


#-- (d) spectral highlights
ax = ax4

z = np.array(radprf_MI_20200213.zlay[i_ref]/1e3) # km

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
inds = (1,3,4)
# radprf_s = radprf_MI_20200213, radprf_MI_20200213,radprf_RMI_20200213
radprf_s = [radprf_MI_20200213lower]*3
cols = 'grey','b','b'
linestyles = '-','-','--'

for i_prof,radprf,col,linestyle in zip(inds,radprf_s,cols,linestyles):

    for band in 'rot','vr':

        nu_prof,W_prof = computeNuProfile(i_prof,radprf,band=band)
        ax.plot(nu_prof,z,c=col,linestyle=linestyle)


ax.set_ylabel(' z(km)')
ax.set_xlabel(r'$\tilde{\nu}$ (cm$^{-1}$)')
ax.set_xlim((300,1600))
ax.set_ylim((-0.15,10.15))

#--- save
plt.savefig(os.path.join(figdir,'Figure%dspec.pdf'%i_fig),bbox_inches='tight')
plt.savefig(os.path.join(figdir,'Figure%dspec.png'%i_fig),dpi=300,bbox_inches='tight')

