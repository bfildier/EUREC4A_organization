#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import numpy as np
import pandas as pd
import datetime

import seaborn as sns
import colorcet as cc
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt

import cartopy.crs as ccrs


# In[2]:


dataDir = pathlib.Path("/Users/robert/Documents/Data/CKDMIP/idealized")
h2o_base   = 'ckdmip_idealized_lw_spectra_h2o_constant'

h2o_nocont = h2o_base.replace("h2o_constant", "h2o-no-continuum_constant") 


# In[3]:


# These are the base-10 logarithms of the a-l water vapor files 
np.linspace(-7, -1.5, num=12) 
# Here's a way to iterate through the letters a-l
[chr(i) for i in range(ord('a'), ord('l')+1)]


# In[4]:


wv_key = 'k'# 10^-7 kg/kg; log 10 values are np.linspace(-7, -1.5, num=12)
wv_tau  = xr.open_dataset(dataDir.joinpath('{}-{}.h5'.format(h2o_base,   wv_key)), 
                          engine='netcdf4', chunks='auto').optical_depth
wv_cont = wv_tau -           xr.open_dataset(dataDir.joinpath('{}-{}.h5'.format(h2o_nocont, wv_key)), 
                          engine='netcdf4', chunks='auto').optical_depth 
             


# In[16]:


level  = 52 # Second from bottom (53 levels)
column =  3 # Six temperatures spanning +/- 10, 30, 50 degrees around an simple temperature profile 
wv_tau_window = wv_tau.isel( level=level, column=column).sel(wavenumber=slice(400,1200))
wv_cnt_window = wv_cont.isel(level=level, column=column).sel(wavenumber=slice(400,1200))

# Resolution is every .0002 cm-1, so 5000 points per cm-1. 
wv_tau_window.plot.line()
wv_cnt_window.plot.line()
plt.yscale('log')


# In[6]:


wv_cnt_window.plot.line()
plt.yscale('log')


# In[21]:


wv_tau_window.plot.line()
plt.yscale('log')


# In[8]:


plt.plot(wv_tau_window.rolling(wavenumber=50000, center=True).mean()) # Every 10 cm-1
plt.yscale('log')


# In[ ]:




