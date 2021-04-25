#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute and store features of radiative cooling data and sonde data during 
the EUREC4A campaign.

Created on Tue Feb  2 10:51:58 2021

@author: bfildier
"""


import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
from matplotlib import gridspec
import xarray as xr
import pandas as pd
import pytz
from datetime import datetime as dt
from datetime import timedelta, timezone
import sys,os,glob
import IPython
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
while os.path.basename(repodir) != projectname:
    repodir = os.path.dirname(repodir)
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
    os.makedirs(os.path.join(resultdir,day),exist_ok=True)


#%%

if __name__ == "__main__":
    
    # arguments
    parser = argparse.ArgumentParser(description="Compute and store features from radiative profile data")
    
    day = '20200126'
    date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
    ref_varid = 'PW'
    
    defineSimDirectories(day)
    
    ##-- import data
    # load radiative profiles
    radprf = xr.open_dataset(os.path.join(inputdir,'rad_profiles_CF.nc'))
    
    # choose profiles for that day that start at bottom
    data_all = radprf.where(radprf.z_min<=50,drop=True)
    data_day = data_all.sel(launch_time=day)
    times = np.array([pytz.utc.localize(dt.strptime(str(d)[:19],'%Y-%m-%dT%H:%M:%S')) for d in data_day.launch_time.values])
    
    dim_t,dim_z = 0,1
    
    # compute all PW values
    mo = MatrixOperators()
    
    QV_all = data_day.specific_humidity.data # kg(w)/kg(a)
    pres = np.nanmean(data_day.pressure.data,axis=dim_t)*100 # hPa
    PW_all = mo.pressureIntegral(QV_all,pres,z_axis=dim_z)
    PW_min = np.nanmin(PW_all)
    PW_max = np.nanmax(PW_all)
    
    ##-- compute radiative features
    
    # Initialize
    f = Features()
    # Find peaks in net Q_rad
    f.computeQradPeaks(data_day,which='net')
    # Find peaks in LW Q_rad
    f.computeQradPeaks(data_day,which='lw')
    # Compute PW
    f.computePW(data_day)
    # Compute PW truncated at qrad peak
    f.computePW(data_day,i_z_max=f.i_net_peak,attr_name='pw_below_net_qrad_peak')
    # Compute PW truncated at lw qrad peak
    f.computePW(data_day,i_z_max=f.i_lw_peak,attr_name='pw_below_lw_qrad_peak')
    
    ##-- compute conditional statistics
    
    #-- reference PW distribution
    #- if reference distribution exists, load it
    ref_filename = 'dist_%s.pickle'%(ref_varid)
    ref_dist_path = os.path.join(resultdir,day,ref_filename)
    ref_dist_exists = len(glob.glob(ref_dist_path)) > 0

    if ref_dist_exists:

        print('load existing reference %s distribution'%ref_varid)
        ref_dist = pickle.load(open(ref_dist_path,'rb'))
    
    else:
        
        print("compute reference %s distribution"%ref_varid)
        # fix range to the total range for each time slice
        ref_var_min = getattr(thismodule,'%s_min'%ref_varid)
        ref_var_max = getattr(thismodule,'%s_max'%ref_varid)
        # compute the distribution
        ref_dist = DistributionOverTime(name=output_varid,time_ref=data2D.time)
        ref_dist.computeDistributions(ref_var_smooth,vmin=ref_var_min,vmax=ref_var_max)
        ref_dist.storeSamplePoints(ref_var_smooth,method='shuffle_mask',
                                         sizemax=time_window*np.prod(ref_var_smooth.shape[1:]))
    
    
    
    
    