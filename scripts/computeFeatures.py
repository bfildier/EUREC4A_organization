#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute and store features of radiative cooling data and sonde data during 
the EUREC4A campaign.

Created on Tue Feb  2 10:51:58 2021

@author: bfildier
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
import IPython
from matplotlib import cm
import matplotlib.image as mpimg
from math import ceil

# geodesic distances and displacements
import geopy
import geopy.distance
# map display
import cartopy.crs as ccrs

## Graphical parameters
plt.style.use(os.path.join(matplotlib.get_configdir(),'stylelib/presentation.mplstyle'))

#%%

if __name__ == "__main__":
    
    # arguments
    parser = argparse.ArgumentParser(description="Compute and store features from radiative profile data")
    
    day = '20200126'
    date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
    
    ##-- import data
    # load radiative profiles
    radprf = xr.open_dataset(os.path.join(indir_sonde_qrad,'rad_profiles_CF.nc'))
    
    # choose profiles for that day that start at bottom
    data_day = radprf.where(radprf.z_min<=50,drop=True).sel(launch_time=day)
    times = np.array([pytz.utc.localize(dt.strptime(str(d)[:19],'%Y-%m-%dT%H:%M:%S')) for d in data_day.launch_time.values])
    
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