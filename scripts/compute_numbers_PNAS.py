#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:46:19 2022

Numbers in PNAS main 2022

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

mo = MatrixOperators()

##--- local functions

def defineSimDirectories():
    """Create specific subdirectories"""
        
    # create output directory if not there
    os.makedirs(os.path.join(figdir),exist_ok=True)


if __name__ == "__main__":
    
    # arguments
    parser = argparse.ArgumentParser(description="Compute paper numbers from all precomputed data")
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

    # load info on all moist intrusions
    mi_file = os.path.join(repodir,'results','idealized_calculations','observed_moist_intrusions','moist_intrusions.pickle')
    moist_intrusions = pickle.load(open(mi_file,'rb'))


#%%   Wavenumbers

    print('-- compute reference wavenumbers --')
    print()
    
    T_ref = 290 # K
    W_ref = 3 # mm
    
    print('choose reference temperature T = %3.1fK'%T_ref)
    print('choose reference water path W = %3.1fmm'%W_ref)
    print()
    
    print("> compute reference wavenumber ")   
    kappa_ref = 1/W_ref # mm-1
    
    rs = rad_scaling_all['20200202']
    nu_ref_rot = rs.nu(kappa_ref,'rot')
    nu_ref_vr = rs.nu(kappa_ref,'vr')
    
    print('reference wavenumber in rotational band: nu = %3.1f cm-1'%(nu_ref_rot/1e2))
    print('reference wavenumber in vibration-rotation band: nu = %3.1f cm-1'%(nu_ref_vr/1e2))
    print()
    
    print("> Planck function at both reference wavenumbers")
    
    piB_ref_rot = pi*rs.planck(nu_ref_rot,T_ref)
    piB_ref_vr = pi*rs.planck(nu_ref_vr,T_ref)  
    
    print('reference Planck term in rotational band: piB = %3.4f J.s-1.sr-1.m-2.cm'%(piB_ref_rot*1e2))
    print('reference Planck term in vibration-rotation band: piB = %3.4f J.s-1.sr-1.m-2.cm'%(piB_ref_vr*1e2))
    
    
#%%  Alpha

    
    
    
    
    
    
    
    
    

