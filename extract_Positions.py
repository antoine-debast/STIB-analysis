# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 16:00:52 2025

@author: Aurélien
"""

import requests
import pandas as pd
import geopandas as gpd
def extract_positions(timestamp):
    url = "https://api.mobilitytwin.brussels/stib/vehicle-position?timestamp="+timestamp

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
data=extract_positions("1755208891")#Specify the wanted start time of the GTFS 15/08/25
#%%
gdf = gpd.GeoDataFrame.from_features(data["features"])
# Plot the GeoDataFrame
#gdf.plot()
#extract_gtfs("1756596942")#Specify the wanted start time of the GTFS 31/08/25 01:35:42
#%%
from_timestamp=1755208891# 14/08/25 22:01:31
to_timestamp = 1755291600# 15/08/25 23:00:00
#to_timestamp = 1756596942# 31/08/25 01:35:42
prev_timestamp=1755208891
prev_data_timestamp=''

datasets=[]

for timestamp in range(from_timestamp,to_timestamp,20):
    data, data_timestamp=extract_positions(timestamp)
    if(data_timestamp!=prev_data_timestamp):
        print("Timestamp:",prev_timestamp,"to",timestamp-1,"("+str(timestamp-prev_timestamp)+")")
        print("Date:",prev_data_timestamp)
        
        gdf = gpd.GeoDataFrame.from_features(data["features"], crs='epsg:4326')
        gdf['pointId']=gdf['pointId'].astype('string')
        datasets.append(gdf)
        
        prev_data_timestamp=data_timestamp
        prev_timestamp=timestamp
gdf=gpd.concat(datasets)
print(gdf.memory_usage(deep=True))
gdf.to_pickle('Some_positions.pkl')