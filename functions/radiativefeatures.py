#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Features of radiative profiles

@author: bfildier
"""

import numpy as np
import xarray as xr


class Features():
    """Finds and stores characteristics of the peak radiative cooling"""
    
    def __init__(self,dz_smooth=150):
        """Class constructor
        
        Arguments:
            - dz_smooth: filter width (default, 150m)"""
        
        self.dz_smooth = dz_smooth
        self.qrad_peak = None

    def __str__(self):
        """Override string function to print attributes
        """
        str_out = ''
        for a in dir(self):
            if '__' not in a:
                a_str = str(getattr(self,a))
                if 'array' in str(getattr(self,a).__class__):
                    str_out = str_out+("%s : %s\n"%(a,str(getattr(self,a).__class__)))
                elif 'method' not in a_str:
                    str_out = str_out+("%s = %s\n"%(a,a_str))
                
        return str_out


    def computeQradPeaks(self,data,which='net'):
        
        if self.qrad_peak is not None:
            print("Abort: qrad_peak is already computed")
            pass
        
        self.launch_time = data.launch_time.values
        self.z = data.alt.values
        # dz = np.nanmean(np.diff(self.z))
        # n_smooth = self.dz_smooth/dz
        
        # define
        setattr(self,'i_%s_peak'%which,np.nan*np.zeros((self.launch_time.size,),dtype=int))
        setattr(self,'z_%s_peak'%which,np.nan*np.zeros((self.launch_time.size,)))
        setattr(self,"qrad_%s_peak"%which,np.nan*np.zeros((self.launch_time.size,)))
        setattr(self,"qrad_%s_smooth"%which,np.nan*np.zeros((self.launch_time.size,self.z.size)))
        
        for i_lt in range(data.dims['launch_time']):
            
            if which == 'net':
                data_i = data.q_rad.values[i_lt]
            else:
                data_i = getattr(data,'q_rad_%s'%which).values[i_lt]
            i, qrad_i, qrad_s = self.findPeak(data_i,return_all=True)
            
            getattr(self,'i_%s_peak'%which)[i_lt] = i
            getattr(self,'z_%s_peak'%which)[i_lt] = self.z[i]
            getattr(self,'qrad_%s_peak'%which)[i_lt] = qrad_i
            getattr(self,'qrad_%s_smooth'%which)[i_lt,:] = qrad_s
            
        # convert to int (again..)
        setattr(self,'i_%s_peak'%which,np.array(getattr(self,'i_%s_peak'%which),dtype=int))
        
        setattr(self,'%s_peaks'%which,xr.Dataset({"launch_time":(["launch_time"], self.launch_time),\
                                 "z":(["zlay"],self.z),\
                                 "longitude":(["launch_time","zlay"],data.longitude.values),\
                                 "latitude":(["launch_time","zlay"],data.latitude.values),\
                                 "i_%s_peak"%which:(["launch_time"],getattr(self,'i_%s_peak'%which)),\
                                 "z_%s_peak"%which:(["launch_time"],getattr(self,'z_%s_peak'%which)),\
                                 "qrad_%s_peak"%which:(["launch_time"],getattr(self,'qrad_%s_peak'%which)),\
                                 "qrad_%s_smooth"%which:(["launch_time","zlay"],getattr(self,'qrad_%s_smooth'%which))}))

    def findPeak(self,values,n_smooth_0=15,return_all=False):
        """Returns index and value of radiative cooling peak.
        
        Arguments:
        - values: numpy array
        - n_smooth: width of smoothing window (number of points)
        - return_all: boolean
        
        Returns:
        - """
        
        val_smooth_0 = np.convolve(values,np.repeat([1/n_smooth_0],n_smooth_0),
                                 'same')

        ind = np.nanargmin(val_smooth_0)

        if return_all:
            return ind, val_smooth_0[ind], val_smooth_0
        else:
            return ind, val_smooth_0[ind]
        
    def computePW(self,data,i_z_max=-1,attr_name='pw'):
        """Compute and store precipitable water
        
        Arguments:
        - data: xarray"""

        # qv
        qv = data.specific_humidity.values
        # density
        t_lay = data.temperature.values
        R = 287 # J/kg/K
        p_lay = data.pressure.values
        rho_lay = p_lay/(R*t_lay)
        self.rho = rho_lay
        # dz
        dz = np.diff(data.alt_edges)
        dz = np.append(np.append([dz[0]],dz),[dz[-1]])
        dz_3D = np.repeat(dz[np.newaxis,:],t_lay.shape[0],axis=0)
        # PW
        pw_layers = qv*rho_lay*dz_3D
        
        if i_z_max.__class__ is int:
            setattr(self,attr_name,np.nansum(pw_layers[:,:i_z_max],axis=1))
        elif i_z_max.__class__ is np.ndarray:
            # truncate at height
            n_pw = pw_layers.shape[0]
            pw = np.nan*np.zeros((n_pw,))
            for i_pw in range(n_pw):
                i_z = i_z_max[i_pw]
                pw[i_pw] = np.nansum(pw_layers[i_pw,:i_z])
            setattr(self,attr_name,pw)