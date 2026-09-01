# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 16:24:01 2025

@author: Aurélien
"""
import os
from pathlib import Path
import zipfile
import datetime 
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
#%%
data_dir="data"
data_out=data_dir+"/daily positions/"
Path(data_out).mkdir(parents=True, exist_ok=True)#Crée le dossier
Positions_zip_file="positions.zip"
files=list(list(zipfile.Path(data_dir+'/'+Positions_zip_file).iterdir())[0].iterdir())
files.sort(key=lambda x:x.name)
#print(files[0:10])

prev_datetime=pd.Timestamp('2025-08-15 00:00:00', tz="UTC+02:00")
prev_timestamp=prev_datetime.timestamp()#1755208800# 2025-08-15 00:00:00
gdfs=[]
next_gdfs=[]
for f in files:
    #print(Position_dir.open(f).read())
    df = pd.read_json(f.open(), encoding='')
    gdf = gpd.GeoDataFrame.from_features(df["features"], crs='epsg:4326')

    gdf_lite=gdf.drop('color', axis=1)
    #gdf_lite['direction']=gdf_lite['direction']-1
    #gdf_lite=gdf_lite.astype({'direction': 'category','lineId':'category', 'timestamp':'category'})
    #gdf_lite.dtypes
    gdf_lite['timestamp_dt']=pd.to_datetime(gdf_lite['timestamp'],utc=True, unit='s').dt.tz_convert('UTC+02:00')
    time=gdf_lite['timestamp_dt'].iloc[0]
    
    if((time-prev_datetime)>pd.Timedelta(value=28, unit='hour')):
        if(len(gdfs)>0):
            day_positions=pd.concat(gdfs)
            day_positions.to_pickle(data_out+'day_positions('+str(prev_datetime.date())+').pkl')
            print("\nFrom:",prev_datetime,"to",time)
            gdfs=next_gdfs
            next_gdfs=[]
        prev_datetime=time.normalize()#prev_datetime+pd.Timedelta(value=1, unit='day')
        #prev_timestamp=prev_timestamp+60*60*24
        
    else:
        if((time-prev_datetime)>pd.Timedelta(value=1, unit='day')):
            next_gdfs.append(gdf_lite)
    gdfs.append(gdf_lite)
    
    #print(gdf.columns)
    print('\r',len(gdfs),str(time),end='')
    
day_positions=pd.concat(gdfs)
day_positions.to_pickle(data_out+'day_positions('+str(prev_datetime.date())+').pkl')
#%% Check the correct slicing
for name in os.listdir(data_out):
    # Open file
    sdate=name.replace('day_positions(','').replace(').pkl','')
    print(name)
    datetime_debut=pd.Timestamp(sdate, tz="UTC+02:00")
    datetime_fin=datetime_debut+pd.Timedelta(value=28, unit='hour')
    
    day_position = pd.read_pickle(data_out+name)
    
    min_date= day_position['timestamp_dt'].min()
    max_date= day_position['timestamp_dt'].max()
    print(f"From {min_date} to {max_date}")
    if(min_date<datetime_debut or max_date>datetime_fin):
        print("Problème")
    day_position=day_position.sort_values('timestamp_dt', ascending=True)
    temp=day_position.groupby('uuid').first()