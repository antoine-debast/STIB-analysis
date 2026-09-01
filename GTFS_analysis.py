# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 11:43:45 2025

@author: Aurélien
"""
import pandas as pd
import geopandas as gpd
import datetime 
import utils
import matplotlib.pyplot as plt

all_stops_times=pd.read_pickle('stop_times(all).pkl')
all_stops_times['arrival_time'].max()#'27:58:00'

del all_stops_times
#%%
GTFS_zip_files=['GTFS_2025-08-13T04-20-01.zip', 'GTFS_2025-08-22T04-20-01.zip',
                'GTFS_2025-09-03T04-20-01.zip','GTFS_2025-09-04T04-20-01.zip','GTFS_2025-09-15T04-20-01.zip']
GTFS_dates=['2025-08-13','2025-08-22','2025-09-03','2025-09-04','2025-09-15']
data_dir="data"

sdate_debut="2025-08-12"
sdate_fin="2025-09-15"
date_fin=pd.to_datetime(sdate_fin)
date=pd.to_datetime(sdate_debut)

GTFS_i=0
active_GTFS_zip_file=GTFS_zip_files[GTFS_i]
for GTFS_i in range(len(GTFS_zip_files)):
    date=pd.to_datetime(GTFS_dates[GTFS_i])
    active_GTFS_zip_file=GTFS_zip_files[GTFS_i]
    geo_stops, stop_times, trips, lines, routes, calendar, calendar_dates = utils.read_gtfs(active_GTFS_zip_file,data_dir)
    #%%Stops exploration
    stop_names_unique=geo_stops.groupby('stop_name').nunique()
    print(stop_names_unique.value_counts('stop_id'))
    stop_names_unique.plot.hist(y=['stop_id'], bins=13, grid=True, logy=True)
    #%% Convert the time columns of stop_times to datetime
    stop_times=stop_times.assign(arrival_datetime=stop_times['arrival_time'].str.split(':').apply(lambda x:date+datetime.timedelta(hours=int(x[0]), minutes=int(x[1]), seconds=int(x[2]))),
                             departure_datetime=stop_times['departure_time'].str.split(':').apply(lambda x:date+datetime.timedelta(hours=int(x[0]), minutes=int(x[1]), seconds=int(x[2]))))
    #%%
    diff_times=stop_times[stop_times['departure_datetime']!=stop_times['arrival_datetime']].reset_index(drop=True)
    diff_times['delta']=(diff_times['departure_datetime']-diff_times['arrival_datetime'])
    plt.figure()
    diff_times['delta'].dt.seconds.sort_values().reset_index(drop=True).plot(grid=True)
    plt.title('Distribution du delta: '+active_GTFS_zip_file)
    plt.ylabel('Différence entre departure_time et arrival_time [s]')
    print(diff_times['delta'].min(),diff_times['delta'].max())
