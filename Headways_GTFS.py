# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 12:25:57 2025

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
out_dir='data/figures/headways_GTFS/'
Path(out_dir).mkdir(parents=True, exist_ok=True)

sdate_debut="2025-09-02"
sdate_fin="2025-09-02"
date_fin=pd.to_datetime(sdate_fin)
date=pd.to_datetime(sdate_debut)

GTFS_i=1
active_GTFS_zip_file=GTFS_zip_files[GTFS_i]
geo_stops, stop_times, trips, lines, routes, calendar, calendar_dates = utils.read_gtfs(active_GTFS_zip_file,data_dir)
while date<=date_fin:
    sdate=str(date.date())
    print("-------------------"+sdate+"-------------------")
    date=pd.to_datetime(sdate)#datetime.datetime.fromisoformat(sdate)
    if(sdate>GTFS_dates[GTFS_i+1]):
        GTFS_i=GTFS_i+1
        active_GTFS_zip_file=GTFS_zip_files[GTFS_i]
        geo_stops, stop_times, trips, lines, routes, calendar, calendar_dates = utils.read_gtfs(active_GTFS_zip_file,data_dir)
    active_stop_times, active_trips =utils.get_active_stop_times(sdate, trips, stop_times, calendar, calendar_dates)
    #%% Convert the time columns of active_stop_times to datetime
    active_stop_times=active_stop_times.assign(arrival_datetime=active_stop_times['arrival_time'].str.split(':').apply(lambda x:date+datetime.timedelta(hours=int(x[0]), minutes=int(x[1]), seconds=int(x[2]))),
                             departure_datetime=active_stop_times['departure_time'].str.split(':').apply(lambda x:date+datetime.timedelta(hours=int(x[0]), minutes=int(x[1]), seconds=int(x[2]))))
    #%% Fusionne les stop_times actifs et les routes
    act=active_trips[['route_id', 'trip_id', 'direction_id', 'shape_id']].merge(active_stop_times, on='trip_id', how='outer')
    r=routes[['route_id', 'route_short_name','route_type']]
    r.assign(route_short_name=r['route_short_name'].astype('string'))
    act=act.merge(r, on='route_id', how='outer', indicator=True)
    print("Résultat de la fusion entre les stop_times actifs(horaire du jour) et les routes(lignes)")
    print(act.drop_duplicates('route_id')['_merge'].value_counts())
    s=act.query("_merge=='right_only'")
    act=act[act['_merge']!="right_only"]
    act=act.drop('_merge',axis=1)
    #%% Fusionne les stop_times actifs et les stops
    act=geo_stops[['stop_id', 'stop_name', 'geometry']].merge(act, on='stop_id', how='right')
    #%%Un même arrêt == même stop_id ou même stop_name ?
    stop_key='stop_name'#'stop_id'#
    act=act.sort_values([stop_key,'route_id','direction_id','arrival_datetime'])
    d=act[[stop_key,'route_id','direction_id', 'departure_datetime']].groupby([stop_key,'route_id','direction_id']).diff()
    act=act.assign(delta=(act['departure_datetime']-act['arrival_datetime']))
    act=act.assign(headway=d['departure_datetime']-act['delta'])
    act=act.assign(headway_s=act['headway'].dt.seconds)
    #%%Summary
    aggr=act.groupby(['route_short_name','direction_id',stop_key])['headway'].agg(['min','mean','median','max','count'])
    #%% Box plots
    ymax=1800
    
    act.query("route_type=='0'").plot.box(title='Tram lines',ylabel='headway [s]',column='headway_s',by='route_short_name',figsize=(10,6), ylim=(0,ymax), grid=True)#[stop_key,'route_id','direction_id']
    plt.hlines(12*60,1,act.query("route_type=='0'")['route_short_name'].nunique(), colors="tab:red")
    act.query("route_type=='1'").plot.box(title='Métro lines',ylabel='headway [s]',column='headway_s',by='route_short_name',figsize=(8,6), ylim=(0,ymax), grid=True)#[stop_key,'route_id','direction_id']
    plt.hlines(12*60,1,act.query("route_type=='1'")['route_short_name'].nunique(), colors="tab:red")
    act.query("route_type=='3'").plot.box(title='Bus lines',ylabel='headway [s]',column='headway_s',by='route_short_name',figsize=(25,6), ylim=(0,ymax), grid=True)#[stop_key,'route_id','direction_id']
    plt.hlines(12*60,1,act.query("route_type=='3'")['route_short_name'].nunique(), colors="tab:red")
    #%% Histograms
    uniques=act.drop_duplicates(['route_id'])
    #uniques=uniques.query("(route_short_name=='69')")
    for (_,s) in uniques.iterrows():
        sampled_route=s['route_id']
        sample_route_name=s['route_short_name']
        sample=act.query("(route_id==@sampled_route)")
        ax=sample[sample['direction_id']].plot.kde(y='headway_s',label='Aller',bw_method=0.05,title='Ligne '+sample_route_name)
        sample[~sample['direction_id']].plot.kde(y='headway_s',label='Retour',bw_method=0.05,title='Ligne '+sample_route_name,ax=ax)
        if(plt.xlim()[1]>3600):
            plt.xlim((0,3600))
        plt.vlines(12*60, 0,0.001, colors="tab:red")
    #%% Plot the evolution of headway at a random stop for a random line and direction
    uniques=act.drop_duplicates([stop_key,'route_id','direction_id'])
    uniques=uniques.groupby(['route_id','direction_id']).sample(3)
    #uniques=uniques.query("route_short_name=='7'")
    uniques=uniques.sort_values(['route_id','direction_id','stop_sequence'])
    for (_,s) in uniques.iterrows():
        sampled_stop=s[stop_key]
        sampled_dir=s['direction_id']
        sampled_route=s['route_id']
        sampled_stop=act[(act[stop_key]==sampled_stop)&(act['direction_id']==sampled_dir)&(act['route_id']==sampled_route)]
        
        sampled_stop.plot.scatter(x='departure_datetime',y='headway_s', 
                                  xlim=(act['departure_datetime'].min(),act['departure_datetime'].max()),
                                  figsize=(15,6),
                                  grid=True)
        plt.hlines(12*60,sampled_stop['departure_datetime'].min(),sampled_stop['departure_datetime'].max(), colors="tab:red")
        plt.title('Arrêt:'+s['stop_name']+' Line: '+str(s['route_short_name'])+' Direction: '+str(sampled_dir))
        plt.savefig(f"{out_dir}Line {str(s['route_short_name'])} {str(sampled_dir)} {s['stop_name']} ({sdate}).jpg")
        plt.show()
    #%%
    plt.figure()
    act['headway'].dt.seconds.plot.hist(logy=True, bins=200)
    plt.title("Histogramme des headways")
    
    plt.figure()
    act['headway'].dt.seconds.plot.hist(logy=True, bins=800, xlim=(0,7200), grid=True)
    plt.title("Histogramme des headways (zoomé)")
    
    plt.figure()
    sub_hour=act[act['headway'].dt.seconds<=3600]
    sub_hour['headway'].dt.seconds.plot.hist(logy=True, bins=360, xlim=(0,3600), grid=True)
    plt.vlines(12*60, 1,100000, colors="tab:red")
    plt.title("Histogramme des headways (zoomé)")
    #%% Outliers
    over_hour=act[act['headway'].dt.seconds>3600].sort_values(['route_id','direction_id','stop_sequence'])
    temp=over_hour[['stop_name','route_short_name','direction_id','headway']].groupby(['route_short_name','direction_id']).agg(['nunique','count','min','max'])
    sub_minit=act[act['headway'].dt.seconds<=60].sort_values(['route_id','direction_id','stop_sequence'])
    
    #%%
    date=date+pd.Timedelta(value=1, unit='day')