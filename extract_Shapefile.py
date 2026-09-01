# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 16:00:52 2025

@author: Aurélien
"""

import requests
import geopandas as gpd
def extract_shapefile(timestamp):
    url = "https://api.mobilitytwin.brussels/stib/shapefile?timestamp="+timestamp

    r = requests.get(url,  headers={
    'Authorization': 'Bearer 17cafff86b7b9a9bf2efecae7b84e0cb59783c17e76173f4b48d9a837e3b87d679cdcff4b6716f9c60e527238293c2097f70489fe2e6142061f04e043f4fc1d1'
    })
    if r.status_code == 200:
        print("Request achieved")
        
    else:
        print("Error "+str(r.status_code)+" :"+r.reason)
        exit(-1)
    data_timestamp=r.headers['X-Data-Timestamp']
    print("Timestamp: "+data_timestamp)
    data_timestamp=data_timestamp.split('.')[0].replace(':', '-')
    data=r.json()
    return data
data=extract_shapefile("1755208891")#Specify the wanted start time of the GTFS 15/08/25
#Timestamp: 2025-08-14T03:20:01.885264
#data=extract_shapefile("1758146400")#Specify the wanted start time of the GTFS 26/08/25
#Timestamp: 2025-08-24T03:20:02.373183
#%%
gdf = gpd.GeoDataFrame.from_features(data["features"],crs=4326)
print(gdf['date_fin'][0])
#%%
# Plot the GeoDataFrame
ax=gdf.plot(color=gdf['color_hex'])

def lon(p):
    return p['lon']
def lat(p):
    return p['lat']
gdf['points']=gpd.points_from_xy(gdf['geo_point_2d'].apply(lon),gdf['geo_point_2d'].apply(lat))
gdf=gdf.set_geometry('points',crs=4326)
gdf.plot(color=gdf['color_hex'],ax=ax)
#%%
gdf=gdf.set_geometry('geometry',crs=4326)
m = gdf.explore(color=gdf['color_hex'])
gdf=gdf.set_geometry('points',crs=4326)
m = gdf.explore(color=gdf['color_hex'],m=m)
outfp = r"base_map.html"
m.save(outfp)
#%%
from_timestamp=1755208891# 14/08/25 22:01:31
to_timestamp = 1755291600# 15/08/25 23:00:00
#to_timestamp = 1756596942# 31/08/25 01:35:42
prev_timestamp=1755208891
prev_data_timestamp=''
for timestamp in range(from_timestamp,to_timestamp,10):
    url = "https://api.mobilitytwin.brussels/stib/vehicle-position?timestamp="+str(timestamp)
    
    r = requests.get(url,  headers={
    'Authorization': 'Bearer 17cafff86b7b9a9bf2efecae7b84e0cb59783c17e76173f4b48d9a837e3b87d679cdcff4b6716f9c60e527238293c2097f70489fe2e6142061f04e043f4fc1d1'
    })
    if r.status_code == 200:
        #print("Request achieved")
        True
    else:
        print("Error "+str(r.status_code)+" :"+r.reason)
        exit(-1)
    data_timestamp=r.headers['X-Data-Timestamp']
    if(data_timestamp!=prev_data_timestamp):
        print("Timestamp:",prev_timestamp,"to",timestamp-1,"("+str(timestamp-prev_timestamp)+")")
        print("Date:",prev_data_timestamp)
        prev_data_timestamp=data_timestamp
        prev_timestamp=timestamp