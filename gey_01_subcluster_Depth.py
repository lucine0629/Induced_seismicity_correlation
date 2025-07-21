# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 22:56:50 2023

@author: kehok
"""
## edited by szuying lai



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import geopandas as gpd
import pickle
from sklearn.cluster import k_means
import os
import csv
import geodetic_conversion as gc
# KetilH stuff
from earthquake.quake import cut_off_depth

#---------------------------
#  Input and output folders
#---------------------------

block = True

data = './data/'
fault = './fault/SHP/'
pkl = './data/pkl/'
pltdir = '../saltonsea/map/geothermal_site/operating-geothermal-plants/'
png = './png/'

#-------------------------------------
#   Read input data
#-------------------------------------

fname = '2005_geysers_Earthquake_Data_with_CluID.csv'
df = pd.read_csv(data+fname, sep=",")
n_clu = df['clu_id'].max() + 1  # clu_id starts from 0
print(f'### n_clu={n_clu}')


#-------------------------------------
#   Read some shp files for plotting 
#-------------------------------------
geoplt = "Operating_Updated_07 24 2014_point.shp"
geothermal_plt = gpd.read_file(pltdir + geoplt)  ## X,Y is utm format (meter)
print(geothermal_plt.crs) ## 4326
geothermal_plt_utm = geothermal_plt.to_crs({'init': 'epsg:32610'}) # convert to utm


# fault zone
faultfn = "Qfaults_US_Database.shp"
cafault = gpd.read_file(fault + faultfn)  ## X,Y is utm format (meter)
print(cafault.crs) ## 4326
cafault_utm = cafault.to_crs({'init': 'epsg:32610'})



####Scale meters in utm coordinates to km
scl = 1e-3
cafault_utm.geometry = cafault.geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))
geothermal_plt_utm.geometry = geothermal_plt.geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))




#-------------------------------
#  Read well data from Prati
#-------------------------------
fname = './injection/Prati_wells_position.csv'
df_wells = pd.read_csv(fname, sep=",")


# Compute utm coords
lon = np.array(df_wells['Long'])
lat = np.array(df_wells['Lat'])
df_wells['x'], df_wells['y'] = gc.wgs_to_utm(lat, lon, 10, 'N')
 










# #------------------------------------
# #  PLot clusters and
# #  cut-off depth per cluster
# #------------------------------------

# Clusters to be lumed together around geysers 
key_id  = 'clu_id'
key_id2 = 'clu_id2'
n_clu0 = 1 
clu_list0 = [0, 1, 2, 3, 4, 5, 6, 8, 9]  
ssgf_list = []
for jj, idd in enumerate(clu_list0):
    ind = df[key_id]==idd
    dfc = df[ind]
    ssgf_list.append(dfc) # For later use



df0 = pd.concat(ssgf_list)
key_dum = 'dum_id'
df0[key_dum] = 0  ## create a new id for loomed cluster




# Subclustering1: Clustering based on depth into 2 clusters
############## sub-clustering 1
df1 = df0.copy()
key_x, key_y = 'Depth', 'x'
n_clu2 = 2
centroid, clu_id2, inertia = k_means(df0[[key_x, key_y]], n_clu2, 
                                       n_init='auto', algorithm='elkan')

df1[key_id2] = clu_id2.astype(int) # change data type to integer
clu_list2 = [jj for jj in range(n_clu2)]


fname_clu2 = 'Level2_Clustered_Data.pkl'
with open(pkl + fname_clu2, 'wb') as fid:
    pickle.dump([df1, n_clu2], fid)
    
    
    
    
    

## then we plot subclustering result in depth view
figd, dx = plt.subplots(figsize=(6,3))
for idd in df['clu_id2'].unique():
    ind = df['clu_id2']==idd
    dfc = df[ind]
    ndd = dfc.shape[0]
    print(f'idd = {idd:2d}: n_eq = {dfc.shape[0]}')  # f is f-string, to pring the string with variables
    
    dx.scatter(scl*dfc.x, dfc.Depth, marker='.', s=0.18, c=idd*np.ones((ndd)), 
               cmap=cm.tab20b, label=f'cluster {idd}',vmin=0, vmax=n_clu-1)
    
dx.set_ybound(0, 8) 
dx.set_xbound(508, 529) 
dx.set_xlabel('Easting (km)')
dx.set_ylabel('Depth (km)')
dx.invert_yaxis()
dx.legend(loc='upper right', fontsize=8, markerscale=4,borderaxespad=0.)
figd.tight_layout(pad=1)
filename = "2005_Geysers_cluster"+"_EQs_xdepth.png"
figd.savefig(png + filename)






figd, dx = plt.subplots(figsize=(6,3))
for idd in df['clu_id2'].unique():
    ind = df['clu_id2']==idd
    dfc = df[ind]
    ndd = dfc.shape[0]
    print(f'idd = {idd:2d}: n_eq = {dfc.shape[0]}')  # f is f-string, to pring the string with variables
    
    dx.scatter(scl*dfc.y, dfc.Depth, marker='.', s=0.18, c=idd*np.ones((ndd)), 
               cmap=cm.tab20b, label=f'cluster {idd}',vmin=0, vmax=n_clu-1)
    
dx.set_ybound(0, 8) 
dx.set_xbound(4286, 4305) 
dx.set_xlabel('Northing (km)')
dx.set_ylabel('Depth (km)')
dx.invert_yaxis()
dx.legend(loc='upper right', fontsize=8, markerscale=4,borderaxespad=0.)
figd.tight_layout(pad=1)
filename = "2005_Geysers_cluster"+"_EQs_ydepth.png"
figd.savefig(png + filename)





    



############### sub-clustering 2: by xy locations
with open(pkl + 'Level2_Clustered_Data.pkl', 'rb') as fid:
    df, n_clu2 = pickle.load(fid)  

# filtering if needed
ymax = 4305000
ind = df['y'] <= ymax
df = df[ind]
kdd=0
df2 = df[df[key_id2]==kdd].copy()

key_id3 = 'clu_id3'
key_x, key_y = 'x', 'y'
n_clu3 = 17
centroid, clu_id3, inertia = k_means(df2[[key_x, key_y]], n_clu3, 
                                       n_init='auto', algorithm='elkan')

df2[key_id3] = clu_id3.astype(int) # change data type to integer




fname_clu2 = 'Level3_Clustered_Data.pkl' # Temporary file
with open(pkl + fname_clu2, 'wb') as fid:
    pickle.dump([df2, n_clu3], fid)
    




## plot subclustering in map view (utm)

fname_clu2 = 'Level3_Clustered_Data.pkl' # Temporary file
with open(pkl + fname_clu2, 'rb') as fid:
    df, n_clu2 = pickle.load(fid)
    df['depth'] = 1000*df['Depth']  # chagne Depth to the depth header name the program needs
    ind = df['Magnitude'] >=0.2
    df = df[ind]


figd, dx = plt.subplots(figsize=(8,7))
for idd in df2['clu_id3'].unique():
    ind = df2['clu_id3']==idd
    dfc = df2[ind]
    ndd = dfc.shape[0]
    print(f'idd = {idd:2d}: n_eq = {dfc.shape[0]}')  # f is f-string, to pring the string with variables
    
    dx.scatter(scl*dfc.x, scl*dfc.y, marker='.', s=0.15, c=idd*np.ones((ndd)), 
               cmap=cm.tab20b, label=f'cluster {idd}',vmin=0, vmax=n_clu-1)
    
for jj in range(n_clu3):
    xc, yc = scl*centroid[jj,0], scl*centroid[jj,1]
    dx.text(xc, yc, f'{jj}')
    
cafault_utm.plot(ax=dx, color='black', linewidth=0.5, label='Quaternary faults')
geothermal_plt_utm.plot(ax=dx, color='r', markersize=12, label='operating plants')
sc = dx.scatter(scl*df_wells['x'], scl*df_wells['y'], marker='^',s=20, color='c', label='Prati injection well')


dx.axis('scaled')
dx.set_xbound(508, 529) 
dx.set_ybound(4286, 4305) 
dx.set_xlabel('Easting (km)')
dx.set_ylabel('Northing (km)')
dx.legend(fontsize=6, markerscale=4,borderaxespad=0.)
figd.tight_layout(pad=1)
plt.show()
filename = "2005_Geysers_subcluster_level3"+"_EQs_utm10n.png"
figd.savefig(png + filename)











