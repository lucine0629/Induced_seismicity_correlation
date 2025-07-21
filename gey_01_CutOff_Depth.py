# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 22:56:50 2023

@author: kehok
"""
## edited by szuying lai



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
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
png = './png/'
shpdir = '../saltonsea/map/geothermal_site/ResourcePotential_Geothermal/'
#-------------------------------------
#   Read input data
#-------------------------------------

# Read earthquake from NCEDC: csv FILE
fname = '1984-2022_ncedc_dd_eq.txt'
df = pd.read_csv(data+fname, sep=",")


# # filter the data to Geysers field
xmin, xmax  = -123.1, -122.5
ymin, ymax  = 38.6, 38.93
ind = (df.Longitude>=xmin) & (df.Longitude<=xmax) & (df.Latitude<=ymax) & (df.Latitude>=ymin)
df = df[ind]

df[['date', 'time']] = df['DateTime'].str.split(' ', n=1, expand=True)
df[['year','month']] = df['date'].str.split('/', n=1, expand=True)
df = df.drop('date', axis=1)
df[['month','day']] = df['month'].str.split('/', n=1, expand=True)
df['year'] = pd.to_numeric(df['year'])

fname_geyser = '1984-2022_geysers_dd_eq.csv'
with open(data+fname_geyser, "wb") as fid:
    df.to_csv(fid, index=False)




# Read earthquake from thurber 2021 (old)
# fname1 = 'thurber/2005_dd.csv'
# fname2 = 'thurber/2011_dd.csv'
# df1 = pd.read_csv(data+fname1, sep=",")
# df2 = pd.read_csv(data+fname2, sep=",")
# thurber_dd = pd.concat([df1, df2])
# thurber_dd['Depth(km)'] = -thurber_dd['Depth(km)'] 


fname = '1984-2022_geysers_dd_eq.csv'
df = pd.read_csv(data+fname, sep=",")




# create new column in utm metre
lon = np.array(df['Longitude'])
lat = np.array(df['Latitude'])
df['x'], df['y'] = gc.wgs_to_utm(lat, lon, 10, 'N') ## epsg32610





#-------------------------------------
#   Read some shp files for plotting 
#-------------------------------------
fname = './injection/Prati_wells_position.csv'
well = pd.read_csv(fname, sep=",")


# Compute utm coords
lon = np.array(well['Long'])
lat = np.array(well['Lat'])
well['x'], well['y'] = gc.wgs_to_utm(lat, lon, 10, 'N')
 






# Geothermal area shp
shp = "ResourcePotential_Geothermal.shp"
geothermal = gpd.read_file(shpdir + shp) 
print(geothermal.crs) ## 3310


pltdir = "../map/geothermal_site/operating-geothermal-plants/"
geoplt = "Operating_Updated_07 24 2014_point.shp"
geothermal_plt = gpd.read_file(pltdir + geoplt) 
print(geothermal_plt.crs) ## 4326


# fault zone
faultfn = "Qfaults_US_Database.shp"
cafault = gpd.read_file(fault + faultfn)
print(cafault.crs) ## 4326



### crs conversion
geothermal = geothermal.to_crs({'init': 'epsg:32610'}) # convert to utm
geothermal_plt = geothermal_plt.to_crs({'init': 'epsg:32610'}) # convert to utm
cafault = cafault.to_crs({'init': 'epsg:32610'})


####Scale meters in utm coordinates to km
scl = 1e-3
geothermal.geometry = geothermal.geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))
geothermal_km = geothermal
geothermal_plt.geometry = geothermal_plt.geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))
cafault.geometry = cafault.geometry.scale(xfact=scl, yfact=scl, zfact=1.0, origin=(0, 0))










##### PLOT eismicity by years
#### magnitude filtering
mag1 = 1.5
ind = df.Magnitude >= mag1
df_15 = df[ind]

mag1 = 1.0
ind = df.Magnitude >= mag1
df_10 = df[ind]

mag1 = 0.5
ind = df.Magnitude >= mag1
df_05 = df[ind]


## eqs counts in years
yrbin = np.arange(1983, 2022)
bin = np.arange(1984, 2022)
eqcount = pd.cut(df['year'], bins=yrbin, labels=bin).value_counts(sort=False)
eqcount15 = pd.cut(df_15['year'], bins=yrbin, labels=bin).value_counts(sort=False)
eqcount10 = pd.cut(df_10['year'], bins=yrbin, labels=bin).value_counts(sort=False)
eqcount05 = pd.cut(df_05['year'], bins=yrbin, labels=bin).value_counts(sort=False)


figc, cx = plt.subplots(figsize=(6,3))
cx.plot(eqcount, label='all earthquakes')
cx.plot(eqcount15, label='Mw >= 1.5')
cx.plot(eqcount10, label='Mw >= 1.0')
cx.plot(eqcount05, label='Mw >= 0.5')
# cx.plot(injection, label='injection volume')  
cx.set_xlabel('year')
cx.set_ylabel('count(per year)')
cx.set_title('Geysers seismicity 1984-2021')
figc.tight_layout(pad=1)
cx.legend()
cx.axis('scaled')
figc.savefig(png + 'Geysers_seismicity.png')
plt.show();










## filter to get data after 2005
year = 2005
ind = df.year >= year
df_2005 = df[ind]



### 1. plot eqs in the geysers field
fig, ax = plt.subplots(figsize=(8,6))
sc = ax.scatter(scl*df_2005.x, scl*df_2005.y, marker='.', s=0.2, c=df_2005.Depth, vmin=0, vmax=5)
geothermal_plt.plot(ax=ax, color='r', markersize=10, label='operating plants')
cafault.plot(ax=ax, color='black', linewidth=0.5, label='Quaternary faults')
ax.axis('scaled')
ax.set_xbound(505, 530) 
ax.set_ybound(4283, 4308) 
ax.set_title('EQs at Geysers after 2005')
ax.set_xlabel('Easting (km)')
ax.set_ylabel('Northing (km)')
ax.figure.colorbar(sc, shrink=0.5, label='Event Depth')
ax.legend()
plt.show();
fig.tight_layout(pad=1)
fig.savefig(png + '2005_geysers_EQs_utm.png')





### 2. plot eqs at depth
figb, bx = plt.subplots(figsize=(7,3.5))
sc = bx.scatter(scl*df_2005.x, df_2005.Depth, marker='.', s=0.07, c=df_2005.Magnitude, vmin=0, vmax=1.0)
ax.set_xbound(505, 530) 
bx.set_ybound(0, 10) 
bx.set_title('Geysers EQs')
bx.set_xlabel('Easting (km)')
bx.set_ylabel('depth (km)')
bx.invert_yaxis()
plt.colorbar(sc, shrink=0.8, label='Magnitude')
figb.tight_layout(pad=1)
figb.savefig(png + '2005_Geysers_EQs_depth.png')







### 3. plot binning eqs at depth to see the major event depth
### -----eqs counts by depth
dpbin = np.arange(0, 7.5, 0.3)
bin = np.arange(0, 7,0.3)
count = pd.cut(df_2005['Depth'], bins=dpbin, labels=bin).value_counts(sort=False)

figb, bx = plt.subplots(figsize=(6,8))
bx.barh(bin, count) 
bx.set_title('geyser EQs at depth', fontsize=18)
bx.set_xlabel('event count', fontsize=18)
bx.set_ylabel('depth (km)', fontsize=18)
bx.tick_params(axis='both', which='major', labelsize=15)
bx.invert_yaxis()
plt.show();
figb.tight_layout(pad=1)
figb.savefig(png + 'geyser_EQs_depth_his.png')






## 4. plot source depth & magnitude
figd, dx = plt.subplots(figsize=(7,8.5))
sc = dx.scatter(df_2005.Magnitude, df_2005.Depth, marker='.', s=0.15)
dx.set_xlabel('Magnitude',fontsize=15)
dx.set_ylabel('depth (km)',fontsize=15)
dx.tick_params(axis='both', which='major', labelsize=15)
dx.set_ybound(0, 7) 
dx.set_xbound(0, 3) 
dx.invert_yaxis()
figd.tight_layout(pad=1)
filename = "depth_magnitude_distribution"
figd.savefig(png + filename)









# #------------------------------------
# #   K-means Clustering
# #------------------------------------
n_clu = 10
        
# # Spatial clustering using kMeans: test differnt numbers of clusters
key_x, key_y = 'x', 'y'

## try x,y,z into clustering
centroid, clu_id, inertia = k_means(df_2005[[key_x, key_y]], n_clu, 
                                    n_init='auto', algorithm='elkan')

df_2005['clu_id'] = clu_id.astype(int)

fname_out = '2005_geysers_Earthquake_Data_with_CluID.csv'
with open(data+fname_out, "wb") as fid:
    df_2005.to_csv(fid, index=False)





# #------------------------------------
# #  PLot clusters and
# #  cut-off depth per cluster
# #------------------------------------

figd, dx = plt.subplots(figsize=(8,5))

style = 'log'
scl = 1e-3
for idd in df_2005['clu_id'].unique():
    ind = df_2005['clu_id']==idd
    dfc = df_2005[ind]
    ndd = dfc.shape[0]
    print(f'idd = {idd:2d}: n_eq = {dfc.shape[0]}')
    dx.scatter(scl*dfc['x'], scl*dfc['y'], marker='.',s=0.2, c=idd*np.ones((ndd)), 
               cmap=cm.tab20b, label=f'cluster {idd}', vmin=0, vmax=n_clu-1)
    


    
    # Mean cut-off per cluster
    depth_mean = np.mean(dfc['Depth'])
    depth_std  = np.std(dfc['Depth'])
    depth_max  = np.max(dfc['Depth'])
    df_2005.loc[ind, 'depth_mean'] = depth_mean
    df_2005.loc[ind, 'depth_std']  = depth_std
    df_2005.loc[ind, 'depth_max']  = depth_max



for jj in range(n_clu):
    xc, yc = centroid[jj,0], centroid[jj,1]
    dx.text(xc, yc, f'{jj}')

cafault.plot(ax=dx, color='black', linewidth=0.4, label='USGS Quaternary faults')
geothermal_plt.plot(ax=dx, color='r', markersize=10, label='operating plants')
dx.set_title('kmeans clustering') 

dx.legend(loc='upper right', fontsize=8, markerscale=2,borderaxespad=0.)
dx.axis('scaled')
dx.set_xbound(500, 530) 
dx.set_ybound(4286, 4308) 
dx.set_xlabel('Easting')
dx.set_ylabel('Northing')
figd.tight_layout(pad=1)
figd.savefig(png + 'GEYSERS_EQ_Clusters.png')
plt.show();






fige, ex = plt.subplots(figsize=(8,4))
cut_off = (df_2005['depth_mean'] + df_2005['depth_std'])
sc = ex.scatter(scl*df_2005.x, scl*df_2005.y, marker='.', s=0.3, 
                c=cut_off, vmin=0.0, vmax=6.0)
cafault.plot(ax=ex, color='black', linewidth=0.5, label='USGS Quaternary faults')
cb = ex.figure.colorbar(sc, ax=ex, shrink=0.8, label='Cutoff depth (km)')

geothermal_plt.plot(ax=ex, color='r', markersize=10, label='operating plants')
sc = ex.scatter(scl*well['x'], scl*well['y'], marker='.',s=6, color='yellow', label='Prati injection well')
ex.axis('scaled')
ex.set_xbound(500, 530) 
ex.set_ybound(4286, 4308) 
ex.set_xlabel('Easting')
ex.set_ylabel('Northing')
ex.set_title('Cut-off depth per cluster')
ex.legend()
fige.tight_layout(pad=1)
plt.show();
fige.savefig(png + 'GEYSERS_EQ_CutOff_Depth_per_Cluster.png')







# #---------------------------------------------------
# #   Compare different Cut-off depth computing methods 
# #---------------------------------------------------

x1, x2 =  np.floor(np.min(df_2005['x'])), np.floor(np.max(df_2005['x'])) + 1e3 # why add another 1km?
y1, y2 =  np.floor(np.min(df_2005['y'])), np.floor(np.max(df_2005['y'])) + 1e3

# # Compute cut-off depth
dx, dy = 500., 500.
nx = int(np.ceil((x2-x1)/dx))
ny = int(np.ceil((y2-y1)/dy))

x = np.linspace(x1,x1+(nx-1)*dx, nx)
y = np.linspace(y1,y1+(ny-1)*dy, ny)

cutoff = cut_off_depth(df_2005, x, y, key_z='Depth', verbose=1)
 # this is to compute the depth on a defined grid. x and y= location in the grid


with open(pkl + 'geyser_Cut_off_Depth.pkl', 'wb') as fid:
    pickle.dump(cutoff, fid) # save as a new file (binary)






#-----------------------------------
#  Plot cut-off depth computed by two methods
#-----------------------------------

fig, axs = plt.subplots(2,2, figsize=(8,6))
for jj in range(3):
    ax = axs.ravel()[jj]
    xp = scl*cutoff.gx.ravel()
    yp = scl*cutoff.gy.ravel()
    zp = cutoff.grd[jj].ravel() 
    sc = ax.scatter(xp, yp, marker='.', s=1, c=zp, vmin=0, vmax=10)
    cb = ax.figure.colorbar(sc, ax=ax)
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_title(f'eq cutoff ({cutoff.label[jj]})')
    ax.axis('scaled')

jj = 2
ax = axs.ravel()[3]
xp = scl*cutoff.gx.ravel()
yp = scl*cutoff.gy.ravel()
zp = cutoff.grd[jj].ravel() - cutoff.grd[0].ravel() 
sc = ax.scatter(xp, yp, marker='.', s=1, c=zp)
cb = ax.figure.colorbar(sc, ax=ax)
ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_title(f'eq cutoff diff {cutoff.label[jj]} - {cutoff.label[0]}')
ax.axis('scaled')

fig.suptitle('Earthquake cut-off depth (3 alternatives)')
fig.tight_layout(pad=1)
fig.savefig(png + 'EQ_CutOff_Depth_Comparison.png')
plt.show(block=block)
    
    