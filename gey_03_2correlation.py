# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:18:44 2024

@author: KEHOK, S.Y. lai
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as img
import pandas as pd
import geopandas as gpd
import pickle
import os
from sklearn.cluster import k_means
import geodetic_conversion as gc

# self-developed stuff
from smooties import smooth2d
from gravmag.common import gridder
import earthquake.quake as quake

#---------------------------
#  Input and output folders
#---------------------------
run_corr = True 
block = True
verbose = 1
plot_cult = False

data = './data/'
fault = './fault/SHP/'
pkl = './data/pkl/'
png = './png/'




#---------------------------
#   Runing parameter: (1) two point corr
#---------------------------
fname_clu2 = 'Level3_Clustered_Data.pkl' # Temporary file
with open(pkl + fname_clu2, 'rb') as fid:
    df, n_clu2 = pickle.load(fid)
    df['depth'] = 1000*df['Depth']  # chagne Depth to the depth header name the program needs
    ind = df['Magnitude'] >=0.2
    df = df[ind]



# # Clusters to be lumed together for analysis
clu_list0 = [2]  
clu_list1 = [1, 3, 16]
clu_list2 = [0, 11, 10, 15]


# two point correlation parameters, dr=inter-event offset (m)
dr0, rmin0, rmax0 = 100., 100., 2000.0 
dr1, rmin1, rmax1 = 50., 250., 900.0 





# Reference lines for comparison 
a_list = [0.5, 1.0]


# selected event depth range: for geothermal related events
zmin =  500 # Min hypocenter depth
zmax = 3150 # Max hypocenter depth 
scl = 1e-3 # m to km



key_id1  = 'clu_id3'

ssgf_list0 = []
for jj, idd in enumerate(clu_list0):
    ind = df[key_id1]==idd
    dfc = df[ind]
    ssgf_list0.append(dfc)




ssgf_list1 = []
for jj, idd in enumerate(clu_list1):
    ind = df[key_id1]==idd
    dfc = df[ind]
    ssgf_list1.append(dfc)



ssgf_list2 = []
for jj, idd in enumerate(clu_list2):
    ind = df[key_id1]==idd
    dfc = df[ind]
    ssgf_list2.append(dfc)


###############################################
#
#  Two point correlation analysis starts here
#
###############################################

#--------------------------------------------
#   Power-law correlation: three groups
#--------------------------------------------

# two point correlation parameters, dr=inter-event offset (m)


density = True
if run_corr:
    key_id  = 'dum_id'
    key_id1  = 'clu_id3'
    df0 = pd.concat(ssgf_list0)
    df1 = pd.concat(ssgf_list1)
    df2 = pd.concat(ssgf_list2)
    df0[key_id] = 1
    df1[key_id] = 1  ## add a new id for all cluster correlation
    df2[key_id] = 1
    clu_dum1 = [1]
    # clu_dum2 = [2]

    ind = df2['year'] == 2021  # look at the events in 1 year period 
    df2_2021 = df2[ind]
 
    
    ind = (df0['year'] >= 2007) & (df0['year'] < 2014)  # look at the events in 1 year period 
    df0_2007 = df0[ind]
    
    
#    rmin, rmax: Min and max distance to use in linear regression analysis
     # Default is [rmin, rmax] = [100, 1000] meters
#    zmin, zmax: Min and max hypocenter depth or z-coord to include in analysis
#    Default is [zmin, zmax] = [0, np.inf]
#    dr: Sampling interval of the two-point correlation
#    df: pd.DataFrame. Earthquake data.
#    Hypcenter in columns [key_x, key_y, key_x], default is [key_x, key_y, key_z] = ['x', 'y', 'depth']
#    Clster ID in column key_id
#    clu_list: list of cluster IDs to analyze



# Level 2 correlations
    fig_p1 = quake.power_correl(df1, clu_dum1, dr1, key_id=key_id, 
                            rmin=rmin1,  rmax=rmax1,  zmin=zmin, zmax=zmax,
                            a_list=a_list, density=density, verbose=verbose)



    fig_p2 = quake.power_correl(df0_2007, clu_dum1, dr1, key_id=key_id, 
                            rmin=rmin1,  rmax=rmax1,  zmin=zmin, zmax=zmax,
                            a_list=a_list, density=density, verbose=verbose)
    
    
    fig_p3 = quake.power_correl(df2, clu_dum1, dr1, key_id=key_id, 
                            rmin=rmin1,  rmax=rmax1,  zmin=zmin, zmax=zmax,
                            a_list=a_list, density=density, verbose=verbose)

    fig_p4 = quake.power_correl(df2_2021, clu_dum1, dr1, key_id=key_id, 
                            rmin=rmin1,  rmax=rmax1,  zmin=zmin, zmax=zmax,
                            a_list=a_list, density=density, verbose=verbose)


# Save plots
 # fig, ax = plt.subplots(figsize=(6,6))
    fig_p1.savefig(png + f'clu1316_TwoPoint_Corr_rmax{rmax1:.0f}_zmax{zmax:.0f}.png')
    fig_p2.savefig(png + f'clu25_TwoPoint_Corr_rmax{rmax1:.0f}_zmax{zmax:.0f}.png')
    fig_p3.savefig(png + f'clu0101115_TwoPoint_Corr_rmax{rmax1:.0f}_zmax{zmax:.0f}.png')
    fig_p4.savefig(png + f'2021_clu0101115_TwoPoint_Corr_rmax{rmax1:.0f}_zmax{zmax:.0f}.png')
plt.show(block=block)







# plot the depth distribution of loomed cluster

# 1. cluster 0 10 11 15
figa, ax = plt.subplots(figsize=(7,2.9))
sc = ax.scatter(scl*df2.x, scl*df2.depth, marker='.', s=0.07)
ax.set_xbound(513, 521) 
ax.set_ybound(0, 3.2)
ax.set_xlabel('x')
ax.set_ylabel('depth (km)')
ax.invert_yaxis()
plt.show();
figa.tight_layout(pad=1)
figa.savefig(png + 'Geysers_0_10_11_15.png')
