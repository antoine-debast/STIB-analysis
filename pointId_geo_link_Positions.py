# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 17:41:53 2025

@author: Aurélien
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import datetime 
import utils
GTFS_zip_files=['GTFS_2025-08-13T04-20-01.zip', 'GTFS_2025-08-22T04-20-01.zip',
                'GTFS_2025-09-03T04-20-01.zip','GTFS_2025-09-04T04-20-01.zip','GTFS_2025-09-15T04-20-01.zip']
GTFS_dates=['2025-08-13','2025-08-22','2025-09-03','2025-09-04','2025-09-15']
data_dir="data"

data_out="data/"
Path(data_out).mkdir(parents=True, exist_ok=True)
Positions_dir="data/daily positions/"
sdate_debut="2025-08-15"#"2025-09-01"#
sdate_fin="2025-09-15"
date_fin=pd.Timestamp(sdate_fin, tz='UTC+02:00')#pd.to_datetime(sdate_fin)
date=pd.Timestamp(sdate_debut, tz='UTC+02:00')
Id_geo_keys=[]
geometries=[]
pointIds=[]
while date<=date_fin:
#%%
    sdate=str(date.date())
    print("-------------------"+sdate+"-------------------")
#%%Load the positions
    fname='day_positions('+str(sdate)+').pkl'
    #%%
    try:
        #%%Extraction VehiclePositions et pre-processing
        day_positions=pd.read_pickle(Positions_dir+fname)
        day_positions=day_positions.sort_values('timestamp')
        day_positions=day_positions.reset_index(drop=True)
        day_positions['pointId']=day_positions['pointId'].astype('string')
        day_positions['lineId']=day_positions['lineId'].astype('string')
        day_positions['direction']=~(day_positions['direction'].astype('int64')-1).astype('boolean')
        day_positions['geometry']=day_positions['geometry'].set_crs("EPSG:4326")
        
        #%%Correspondance pointId/geometry
        Id_geo_key=day_positions.query("distanceFromPoint==0").drop_duplicates(['pointId','geometry','lineId'])
        Id_geo_key=Id_geo_key[['pointId','geometry','lineId']]
        print("Geometry liés à plus d'un pointId")
        print(Id_geo_key.groupby(['geometry','lineId']).nunique().sort_values('pointId').query('pointId>1'))
        print("pointIds liés à plus d'une Geometry")
        print(Id_geo_key.groupby(['pointId','lineId']).nunique().sort_values('geometry').query('geometry>1'))
        print(len(Id_geo_key))
        Id_geo_keys.append(Id_geo_key)
        #%%
        geometries.append(day_positions.drop_duplicates(['geometry','lineId','direction']))
        pointIds.append(day_positions.drop_duplicates(['pointId','lineId','direction']))
    except(FileNotFoundError):
        print(sdate,"manquant. On passe")
    date=date+pd.Timedelta(value=1, unit='day')
Id_geo_key_all=pd.concat(Id_geo_keys)
Id_geo_key_all=Id_geo_key_all.drop_duplicates(['pointId','geometry','lineId'])
Id_geo_key_all.to_pickle(data_out+'Id_geo_key.pkl')
#%%
geometries_df=pd.concat(geometries).drop_duplicates(['geometry','lineId','direction'])
pointIds_df=pd.concat(pointIds).drop_duplicates(['pointId','lineId','direction'])
#%%
print(f"Ids in the key: {len(Id_geo_key_all.drop_duplicates(['pointId']))} on {len(pointIds_df.drop_duplicates('pointId'))}")
print(f"Geometries in the key: {len(Id_geo_key_all.drop_duplicates(['geometry']))} on {len(geometries_df.drop_duplicates('geometry'))}")
geometries_notfound_df=geometries_df[~geometries_df['geometry'].isin(Id_geo_key_all['geometry'])]
pointIds_notfound_df=pointIds_df[~pointIds_df['pointId'].isin(Id_geo_key_all['pointId'])]
#%%GTFS
GTFS_i=1
active_GTFS_zip_file=GTFS_zip_files[GTFS_i]
geo_stops, stop_times, trips, lines, routes, calendar, calendar_dates = utils.read_gtfs(active_GTFS_zip_file,data_dir)
#%%Essaye de trouver les geometry manquants grâce au GTFS
geo_stops['stop_id2']=geo_stops['stop_id'].str.strip('ABCDEFGH').str.lstrip(to_strip='0')
geo_stops_sameId=geo_stops.query("stop_id2.isin(@geometries_notfound_df['pointId'])")
geo_stops_candidat=geo_stops.query("~stop_id2.isin(@geometries_notfound_df['pointId'])")
merge_geo=geometries_notfound_df.to_crs('epsg:3857').sjoin_nearest(geo_stops_candidat.to_crs('epsg:3857'),#max_distance=100,
                                                               how='left',distance_col='merge_dist')
merge_geo=merge_geo.to_crs('EPSG:4326').sort_values('merge_dist')
merge_geo=merge_geo.drop_duplicates('stop_id')
merge_geo=merge_geo.query("merge_dist<200")
m=geo_stops_sameId.explore( color='red')
m=merge_geo.explore(column='merge_dist', cmap='tab20', m=m)
m.save("key_map.html")
#%%Essaye de trouver les pointId manquants grâce au GTFS
merge_id=pointIds_notfound_df.merge(geo_stops, left_on='pointId',
                                               right_on='stop_id',how='left')
merge_id=merge_id.set_geometry('geometry_y')
merge_id_geo2=merge_id.to_crs('epsg:3857').sjoin_nearest(geometries_df.to_crs('epsg:3857'),#max_distance=100,
                                                               how='right',distance_col='merge_dist')
merge_id_geo2=merge_id_geo2.to_crs('EPSG:4326').sort_values('merge_dist')
#merge_id_geo2=merge_id_geo2.query("(lineId_left==lineId_right)&(direction_left==direction_right)")
merge_id_geo2=merge_id_geo2.drop_duplicates(['pointId_left','lineId_left','direction_left'])
m=merge_id_geo2.explore(column='merge_dist')
m.save("key_map_Id.html")
#merge_id['sp_error']=merge_id['geometry_x'].to_crs('epsg:3857').distance(merge_id['geometry_y'].to_crs('epsg:3857'))
#TODO: Finir