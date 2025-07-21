# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 22:56:50 2023

@author: kehok

## edited by szu-ying lai
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as img
import pandas as pd
import geopandas as gpd
import pickle
import os

from sklearn.cluster import k_means
import geodetic_conversion as gc

# KetilH stuff
from smooties import smooth2d
import khio.grid as khio
from gravmag.common import gridder
from gravmag.common import MapData
import earthquake.quake as quake

#---------------------------
#  Input and output folders
#---------------------------

block = True

plot_cult = False

data = './data/'
fault = './fault/SHP/'
pkl = './data/pkl/'
shpdir = "../saltonsea/map/geothermal_site/ResourcePotential_Geothermal/"
pltdir = "../saltonsea/map/geothermal_site/operating-geothermal-plants/"
png = './png/'


#-----------------------------------s
#   Cut-off depths
#-----------------------------------

with open(pkl + 'geyser_Cut_off_Depth.pkl', 'rb') as fid:
    eq = pickle.load(fid)  
    
## in depth.pkl file, there is x,y,gx,gy,z
 # inside grd:[depth_max, depth_Pq, depth_mean_std, depth_mean, depth_std]



#----------------------------------------
#   Read some shape files for plotting
#----------------------------------------

scl = 1e-3 # m to km for x and y


# Geothermal area shp
shp = "ResourcePotential_Geothermal.shp"
geothermal = gpd.read_file(shpdir + shp)  ## X,Y is utm format (meter)
print(geothermal.crs) ## 3310
geothermal = geothermal.to_crs({'init': 'epsg:32611'}) # convert to utm



geoplt = "Operating_Updated_07 24 2014_point.shp"
geothermal_plt = gpd.read_file(pltdir + geoplt)  ## X,Y is utm format (meter)
print(geothermal_plt.crs) ## 4326
geothermal_plt = geothermal_plt.to_crs({'init': 'epsg:32611'}) # convert to utm


# fault zone
faultfn = "Qfaults_US_Database.shp"
cafault = gpd.read_file(fault + faultfn)  ## X,Y is utm format (meter)
print(cafault.crs) ## 4326
cafault = cafault.to_crs({'init': 'epsg:32611'}) # convert to utm


    
# Scale meters to km
cafault.geometry = cafault.geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))
geothermal_plt.geometry = geothermal_plt.geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))
geothermal.geometry = geothermal.geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))


#-------------------------------
#  Read well data from Prati
#-------------------------------
fname = './injection/Prati_wells_position.csv'
df_wells = pd.read_csv(fname, sep=",")


# Compute utm coords
lon = np.array(df_wells['Long'])
lat = np.array(df_wells['Lat'])
df_wells['x'], df_wells['y'] = gc.wgs_to_utm(lat, lon, 10, 'N')
    


#-------------------------------------
# Read EQ input
#-------------------------------------

fname = '2005_geysers_Earthquake_Data_with_CluID.csv'
df_all = pd.read_csv(data+fname, sep=",")
n_clu = df_all['clu_id'].max() + 1  # clu_id starts from 0
print(f'### n_clu={n_clu}')





#------------------------------------------------
# computing b-values for entire catalogue: two methods
#------------------------------------------------

# mc, mc2 = 1.9, 6.21
mc, mc2 = 0.9, 4.3
magnitude = np.array(df_all['Magnitude'])

# 1. b-value by line reg
dm = 0.1 # magnitude binning
b, a = quake.reg_b_value(magnitude, mc, mc2, dm)

# 2. b-value by Aki (1965) method
delm = 0.0
b_aki, std_aki = quake.aki_b_value(magnitude, mc, mc2, delm)
a_aki = a - 0.30 # by trial and error, need to change it accordingly

print (f'Regression: b={b:.3f},  a={a:.3f}')
print (f'Aki MLE:    b={b_aki:.3f} +/- {std_aki:.3f}')


#----------------------------------
# Plot Gutenberg-Richter trend
#----------------------------------

suptitle = 'Gutenberg-Richter law: Geysers NSCEC 2005-2021 catalogue'
label, label2 = f'Linear regression, b={b:.3f}', f'Aki (1965) MLE, , b={b_aki:.3f}'

fig = quake.plot_gutenberg_richter(magnitude, mc, mc2, dm, b=b, a=a, label=label,
                                   b2=b_aki, a2=a_aki, label2=label2, suptitle=suptitle)

fig.savefig(png + 'Geysers_Gutenberg_Richter_All.png')








#------------------------------------
# b-values for each cluster
#------------------------------------

# Reset:
sm = 1.0
scl=1e-3
key_x, key_y = 'x', 'y'
mc, mc2  = 0.9, 4.3
ind = (df_all.Magnitude>=mc) & (df_all.Magnitude<=mc2)
df = df_all[ind]
df['b'] = 0.0

# Spatial clustering using kMeans: CLustering in sc_02...py
cent_x = np.zeros(n_clu)
cent_y = np.zeros(n_clu)

figa, ax = plt.subplots(1, figsize=(10,10))
figk, kxs = plt.subplots(4, n_clu//4, figsize=(24,10))
style = 'log'
nbin = 34 # 34 magnitude value interval


# we're going to compute b value for each cluster and saves as a dataframe
df_bval = pd.DataFrame(columns=['cluster', 'nn', 'a', 'b', 'b_mle', 'std_mle'], index=range(0,n_clu))

for idd in df['clu_id'].unique():
    ind = df['clu_id']==idd # filter out events for this cluster
    dfc = df[ind]
    ndd = dfc.shape[0] #number of events in this cluster
    cent_x[idd] = np.nanmean(dfc[key_x]) # get the mean(ignore nan value)
    cent_y[idd] = np.nanmean(dfc[key_y])
    print(f'idd = {idd:2d}: n_eq = {dfc.shape[0]}')
    
    magnitude = np.array(dfc['Magnitude'])
    ax.scatter(scl*dfc['x'], scl*dfc['y'], marker='.', c=idd*np.ones((ndd)), 
               s=sm*magnitude, cmap=cm.tab20b, vmin=0, vmax=n_clu-1)
    
    # b-value by line reg
    dm = 0.1
    b, a = quake.reg_b_value(magnitude, mc, mc2, dm)

    # b-value by Aki (1965) method
    delm = 0.0
    b_mle, std_mle = quake.aki_b_value(magnitude, mc, mc2, delm)
    a_mle = a + 0.1

    # For plotting
    # get the frequency of events per bin
    [Nh, be] = np.histogram(dfc.Magnitude, bins=nbin, range=(mc,mc2))
    m = (be[0:-1] + be[1:])/2
    Ncum = np.cumsum(Nh[::-1])[::-1]
    marr = np.linspace(mc, mc2,101)
    Narr = 10**(a-b*marr)

    # print (f'Regression: idd={idd}, b={b:.3f},  a={a:.3f}')
    # print (f'Aki MLE:               b={b_mle:.3f} +/- {std_mle:.3f}')

    nn = dfc.shape[0]
    df_bval.loc[idd, ['cluster', 'nn', 'a', 'b', 'b_mle', 'std_mle']] = [idd, nn, a, b, b_mle, std_mle]
    df.loc[ind, ['b', 'b_mle', 'std_mle']]  = [b, b_mle, std_mle]
 
    
 # Assign b-value to all EQ in the cluster (for output)
    jnd = df_all['clu_id']==idd
    df_all.loc[jnd, ['b', 'b_mle', 'std_mle']]  = [b, b_mle, std_mle]

    # Mean cut-off by cluster
    depth_mean = np.mean(dfc['Depth'])
    depth_std  = np.std(dfc['Depth'])
    depth_max  = np.max(dfc['Depth'])
    df.loc[ind, 'depth_mean'] = depth_mean
    df.loc[ind, 'depth_std']  = depth_std
    df.loc[ind, 'depth_max']  = depth_max

    kx = kxs.ravel()[idd]
    kx.plot(marr, Narr, 'k-')
    kx.plot(m, Ncum,'r-o')    
    kx.set_yscale('log')
    kx.set_title(f'cluster {idd}: n={dfc.shape[0]}, b_mle={b_mle:.2f}')
    kx.set_xlim(mc, mc2)
    kx.set_ylim(1*10**0, 2*10**3)
    
figk.tight_layout(pad=1)
figk.savefig(png + f'EQ_Gutenberg_Richter_per_Cluster_{style}.png')


ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_title('kmeans clustering')

for jj in range(n_clu):
    ax.text(scl*cent_x[jj], scl*cent_y[jj], f'{jj}')

plt.show();
figa.savefig(png + 'EQ_Clusters.png')


