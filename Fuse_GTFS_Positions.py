# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 15:49:17 2025

@author: Aurélien
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import datetime 
import utils
import gc
GTFS_zip_files=['GTFS_2025-08-13T04-20-01.zip', 'GTFS_2025-08-22T04-20-01.zip',
                'GTFS_2025-09-03T04-20-01.zip','GTFS_2025-09-04T04-20-01.zip','GTFS_2025-09-15T04-20-01.zip']
GTFS_dates=['2025-08-13','2025-08-22','2025-09-03','2025-09-04','2025-09-15']
GTFS_dir="data"
data_out="data/fusion result/"
Path(data_out).mkdir(parents=True, exist_ok=True)
Positions_dir="data/daily positions/"
sdate_debut="2025-08-15"#"2025-09-01"#
sdate_fin="2025-09-15"
date_fin=pd.Timestamp(sdate_fin, tz='UTC+02:00')#pd.to_datetime(sdate_fin)
date=pd.Timestamp(sdate_debut, tz='UTC+02:00')#pd.to_datetime(sdate_debut)

#Lines in GTFS but not in the VehiclePositions
excluded_lines=['10','20','50','60','80','96',
                'N04','N05','N06','N08','N09','N10']
yesterday_uuids=[]

GTFS_i=0
while(sdate_debut>=GTFS_dates[GTFS_i+1]):
    GTFS_i=GTFS_i+1
active_GTFS_zip_file=GTFS_zip_files[GTFS_i]
geo_stops, stop_times, trips, lines, routes, calendar, calendar_dates = utils.read_gtfs(active_GTFS_zip_file,GTFS_dir)
#%%Evaluation Function
def evaluate_fusion(merge_res, at_stops_positions):
    #La fusion ne garde que les lignes dans le GTFS
    #Les lignes dans VehiclePositions non fusionnées sont jetées
    #Pourcentage de ligne fusionnées
    GTFS_lines_count=merge_res['route_short_name'].count()
    Both_lines_count=merge_res['lineId'].count()
    print(f"Rows du GTFS trouvées: {Both_lines_count} sur {GTFS_lines_count} ({(Both_lines_count/GTFS_lines_count)*100}%)")
    
    #Lignes de VehiclePositions dupliquées
    found_in_Positions=merge_res.dropna(how='all', subset=['timestamp_dt_mean', 'lineId', 'direction', 'uuid'])
    found_in_Positions_unique=found_in_Positions.drop_duplicates(['timestamp_dt_mean','lineId','direction','uuid'])
    Positions_duplicated=found_in_Positions[found_in_Positions.duplicated(['timestamp_dt_mean','lineId','direction','uuid'])]
    print(f"Rows excessives/dupliquées: {len(Positions_duplicated)}")
    print("Rows uniques de VehiclePositions:",len(found_in_Positions_unique))
    
    print(f"Rows du GTFS trouvées sans dupliqués: {Both_lines_count-len(Positions_duplicated)} sur {GTFS_lines_count} ({((Both_lines_count-len(Positions_duplicated))/GTFS_lines_count)*100}%)")

    
    #Pourcentages de ligne de VehiclePositions non fusionnées
    temp=found_in_Positions_unique.merge(at_stops_positions, on=['timestamp_dt_mean','lineId','direction','uuid'], how='right', indicator=True)
    merge_res_not_Pos=temp[temp['_merge']=='right_only'].dropna(how='all',axis=1)
    print(f"Rows de VehiclePositions non fusionnées:{len(merge_res_not_Pos)} sur {len(at_stops_positions)}")
    #Vérification: 1 trip = 1 uuid
    per_trip_nu=merge_res.groupby('trip_id').nunique()
    #print(per_trip_nu['uuid'].value_counts())
    per_uuid_nu=merge_res.groupby('uuid').nunique()
    #print(per_uuid_nu['trip_id'].value_counts())
    temp=merged2.groupby(['trip_id', 'uuid','route_short_name','direction_id'])[['stop_id']].count()
    temp=temp.reset_index().set_index('trip_id')
    temp['sum']=temp.groupby('trip_id')['stop_id'].sum()
    temp['frac']=temp['stop_id']/temp['sum']
    te=temp.groupby(['trip_id','route_short_name','direction_id']).max()
    if(False):
        te['frac'].plot.hist(bins=50,title="Proportion du uuid majoritaire pour chaque trip")
    #Conversion 
    merge_res=merge_res.assign(delay_s=merge_res['delay'].dt.total_seconds())
    merge_res=merge_res.assign(good_pos=(merge_res['distanceFromPoint_post']==0))
    
    #Distribution des retards
    if(False):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.semilogy(merge_res['delay_s'].abs().sort_values().reset_index(drop=True))
        ax.grid(True)
        ax.set_title(f"Fonction de distibution du Retard ({sdate})")
        plt.show()
        plt.figure()
        merge_res['delay_s'].plot.hist(log=True, bins=1000, title=f'Retard en secondes {sdate}', grid=True)
        plt.show()
        plt.figure()
        merge_res['delay_s'].plot.hist(log=True, bins=1000, title=f'Retard en secondes {sdate}(zoomé)', grid=True, xlim=(-3600,3600))
        plt.show()
        
        #Erreur spatiale
        plt.figure()
        merge_res.plot.hist(column=['sp_error'],log=True, bins=200, title=f'Erreur spatiale en mètres {sdate}', grid=True)
        plt.show()
    
    # plt.figure()
    # merge_res.plot.scatter(x='delay_s', y='sp_error')
    # plt.show()
    
    #Lien entre 'distanceFromPoint' et l'erreur spatiale
    #Clairement, quand 'distanceFromPoint' == 0, l'erreur spatiale est moindre
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.loglog(merge_res['distanceFromPoint']+0.5, merge_res['sp_error']+0.0001,'x')
    # ax.set_xlabel('distanceFromPoint')
    # ax.set_ylabel('Erreur spatiale')
    # plt.show()
    
    #Influence de la position du stop dans la ligne
    merge_res=merge_res.assign(extrem_stop=1)
    merge_res.loc[merge_res.groupby('trip_id')['stop_sequence'].idxmin(),'extrem_stop']=0
    merge_res.loc[merge_res.groupby('trip_id')['stop_sequence'].idxmax(),'extrem_stop']=2
    print("Pourcentage de fusion réussie selon que le stop soit le départ, au milieu ou la fin de la ligne")
    print((merge_res.groupby('extrem_stop').count()['delay_s']/merge_res.groupby('extrem_stop').count()['trip_id'])*100)
    # plt.figure()
    # merge_res.plot.scatter(x='delay_s', y='sp_error', c='extrem_stop', cmap='tab10')
    # plt.show()
    # plt.figure(figsize=(15,27))
    # merge_res.plot.hist(column=['sp_error'],by='extrem_stop',log=True, bins=100, title='Erreur spatiale en mètres ?', grid=True, density=True)
    # plt.show()
    # plt.figure(figsize=(15,27))
    # merge_res.plot.hist(column=['delay_s'],by='extrem_stop',log=True, bins=1000, title='Retard en secondes', grid=True,xlim=(-5000,5000), density=True)
    # plt.show()
#%%Vizualization function for the stops
def Visualize(prefix_titles, dfs_GTFS, dfs_Positions):
    if(len(dfs_GTFS)!=len(dfs_Positions)):
        print("Erreur !")
        return
    lines=dfs_GTFS[0][['route_short_name','direction_id']].drop_duplicates().values
    for line in lines:
        lineId=line[0]
        dir=line[1]
        ref_stop_geo=dfs_GTFS[0].query("(route_short_name==@lineId)&(direction_id==@dir)")['stop_geo'].to_crs('epsg:3857').unique()[0]
        ref_name=dfs_GTFS[0].query("(route_short_name==@lineId)&(direction_id==@dir)")['stop_name'].unique()[0]
        for i in range(len(dfs_GTFS)):
            #Avant fusion
            candidates_Positions=dfs_Positions[i].query("(lineId==@lineId)&(direction==@dir)")
            candidates_GTFS=dfs_GTFS[i].query("(route_short_name==@lineId)&(direction_id==@dir)")
            
            candidates_GTFS=candidates_GTFS.sort_values('stop_sequence', ascending=False)
            candidates_GTFS['dist_to_0']=candidates_GTFS['stop_geo'].to_crs('epsg:3857').distance(ref_stop_geo)
            candidates_Positions['dist_to_0']=candidates_Positions['geometry'].to_crs('epsg:3857').distance(ref_stop_geo)     
            ax=candidates_GTFS.plot.scatter(x='departure_datetime', y='dist_to_0', c='red')
            candidates_Positions.plot.scatter(x='timestamp_dt_mean', y='dist_to_0', ax=ax,marker='x',
                               title=prefix_titles[i]+lineId+' '+str(dir)+' '+ref_name, figsize=(18,12), grid=True)
            #ax.vlines(group['departure_datetime'],0,1000, colors="tab:red")
            plt.show()
#%%
while date<=date_fin:
#%%
    sdate=str(date.date())
    print("-------------------"+sdate+"-------------------")
    #date=pd.to_datetime(sdate)#datetime.datetime.fromisoformat(sdate)
    if(sdate>=GTFS_dates[GTFS_i+1]):
        GTFS_i=GTFS_i+1
        active_GTFS_zip_file=GTFS_zip_files[GTFS_i]
        geo_stops, stop_times, trips, lines, routes, calendar, calendar_dates = utils.read_gtfs(active_GTFS_zip_file,GTFS_dir)
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
        #day_positions['datetime']=day_positions['timestamp'].apply(lambda x:datetime.datetime.fromtimestamp(x))
        #day_positions['timestamp_dt']=pd.to_datetime(1755208800,utc=True, unit='s').tz_convert('UTC+02:00')
        #day_positions['direction']=(day_positions['direction'].astype('int64')-1).astype('boolean')
        day_positions['direction']=~(day_positions['direction'].astype('int64')-1).astype('boolean')
        day_positions['geometry']=day_positions['geometry'].set_crs("EPSG:4326")
        #%%
        active_stop_times, active_trips =utils.get_active_stop_times(sdate, trips, stop_times, calendar, calendar_dates)
        #%% Convert the time columns of active_stop_times to datetime
        #active_stop_times=active_stop_times.assign(arrival_datetime=active_stop_times['arrival_time'].str.split(':').apply(lambda x:date+datetime.timedelta(hours=int(x[0]), minutes=int(x[1]), seconds=int(x[2]))),
        #                         departure_datetime=active_stop_times['departure_time'].str.split(':').apply(lambda x:date+datetime.timedelta(hours=int(x[0]), minutes=int(x[1]), seconds=int(x[2]))))
        active_stop_times=active_stop_times.assign(arrival_datetime=date+pd.to_timedelta(active_stop_times['arrival_time']),
                                                   departure_datetime=date+pd.to_timedelta(active_stop_times['departure_time']))
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
        act=act.assign(route_short_name=act['route_short_name'].astype('string'))
        #Fusion des stop_times actifs et geo_stops
        gstops=geo_stops[['stop_id', 'stop_name', 'location_type', 'parent_station', 'geometry']]
        gstops=gstops.rename(columns={'geometry':'stop_geo'})
        act=gstops.merge(act, on='stop_id', how='outer', indicator=True)
        print("Résultat de la fusion entre les arrêts et les stop_times actifs(horaire du jour)")
        print(act['_merge'].value_counts())#484 stops not in the active stop_times
        act=act[act['_merge']!="left_only"]
        act=act.drop('_merge',axis=1)
        act=act.assign(stop_id=act['stop_id'].astype('string'))
        del r, gstops,active_stop_times, active_trips
        gc.collect()
        #%% Retire les lignes qui ne sont pas dans VehiclePositions
        act_route_notinPositions=act.query("route_short_name.isin(@excluded_lines)")
        print(f"Excluding the routes not in VehiclePositions: {act_route_notinPositions.shape[0]} on {act.shape[0]} lines dropped")
        act=act.query("~route_short_name.isin(@excluded_lines)")
        #%% Se débarasse des colonnes inutiles
        act=act.drop(['source','arrival_time','departure_time'],axis=1)

        #%%Défausse les véhicules qui commence à circuler la veille
        #Il y a des véhicules qui commencent à circuler le lendemain mais qui sont compté dans le GTFS du jour même
        # #le lendemain
        tomorrow=date+pd.Timedelta(value=1, unit='day')
        # tomorrow_positions=day_positions.query('timestamp_dt>=@tomorrow')
        today_positions=day_positions.query('(timestamp_dt>=@date)&(timestamp_dt<@tomorrow)')
        today_uuids=today_positions['uuid'].drop_duplicates().values
        inter_uuids=np.intersect1d(today_uuids, yesterday_uuids, assume_unique=False)
        
        print(f"Nombre de véhicules du jour démarrant la veille : {len(inter_uuids)}")
        print(f"Nombre de lignes du matin défaussées: {day_positions['uuid'].isin(yesterday_uuids).value_counts()}")
        day_positions=day_positions[~day_positions['uuid'].isin(yesterday_uuids)]
        yesterday_uuids=today_uuids
        del tomorrow, today_positions, inter_uuids, today_uuids
        #%% Retire les lignes de VehiclePositions avant 4h26 car elles sont dans l'horaire de la veille
        date_tresh=date+pd.Timedelta(value=4, unit='hour')
        day_positions=day_positions.query('timestamp_dt>@date_tresh')
        #%%Keep only the positions with the earliest timestamp for each stop, line, direction and vehicle
        at_stops_positions=utils.get_at_stop_positions(day_positions)    
        #%%Vérification du nombre d'arrêts en commun
        # points=day_positions.loc[day_positions.groupby(['geometry','pointId'], as_index=False).idxmin()['distanceFromPoint'],:]
        # # points_0=points.query("distanceFromPoint==0")
        # # points_1=points.query("distanceFromPoint>0")
        
        # act_stops=act.drop_duplicates('stop_id')[['stop_id', 'stop_name', 'location_type', 'parent_station', 'stop_geo']]
        
        # stop_merged=act_stops.merge(points,left_on='stop_id', right_on='pointId', how='outer', indicator=True)
        # print(stop_merged['_merge'].value_counts())
        # merged_both=stop_merged[(stop_merged['_merge']=='both')].drop("_merge", axis=1)
        # points_only=stop_merged[(stop_merged['_merge']=='right_only')].dropna(axis=1,how='all').drop("_merge", axis=1)
        # points_only=points_only.set_geometry('geometry')
        # GTFS_only=stop_merged[(stop_merged['_merge']=='left_only')].dropna(axis=1,how='all').drop("_merge", axis=1)
        # GTFS_only=GTFS_only.set_geometry('stop_geo')
        # #%%Vérification du nombre d'arrêts en commun en retirant les suffixes
        # GTFS_only['stop_id2']=GTFS_only['stop_id'].str.strip('ABCDEFGH').str.lstrip(to_strip='0')
        # stop_merged2=GTFS_only.merge(points_only,left_on='stop_id2', right_on='pointId', how='outer', indicator=True)
        # print(stop_merged2['_merge'].value_counts())
        # merged_both2=stop_merged2[(stop_merged2['_merge']=='both')].drop("_merge", axis=1)
        # points_only2=stop_merged2[(stop_merged2['_merge']=='right_only')].dropna(axis=1,how='all').drop("_merge", axis=1)
        # points_only2=points_only2.set_geometry('geometry')
        # GTFS_only2=stop_merged2[(stop_merged2['_merge']=='left_only')].dropna(axis=1,how='all').drop("_merge", axis=1)
        # GTFS_only2=GTFS_only2.set_geometry('stop_geo')
        
        # stop_merge_res=pd.concat([merged_both, merged_both2])
        # stop_merge_res=stop_merge_res.loc[stop_merge_res.drop(['geometry','stop_geo'],axis=1).groupby(['pointId'], as_index=False).idxmin()['distanceFromPoint'],:]
        # stop_merge_res=stop_merge_res.reset_index(drop=True)
        # #%%Vérification de la fusion des arrêts
        # stop_merge_res['sp_error']=stop_merge_res['stop_geo'].to_crs('epsg:3857').distance(stop_merge_res['geometry'].to_crs('epsg:3857'))
        # stop_merge_res.loc[(stop_merge_res['distanceFromPoint'])!=0, 'sp_error']=-stop_merge_res.loc[(stop_merge_res['distanceFromPoint'])!=0, 'sp_error']
        # plt.figure()
        # stop_merge_res['sp_error'].plot.hist(log=True, bins=100, title='Erreur spatiale en mètres ?', grid=True)
        # #%%Vérification du nombre d'arrêts en commun en prenant les arrêts les plus proches
        # #TODO:
        # stop_merged3=GTFS_only2.to_crs('epsg:3857').sjoin_nearest(points_only2.to_crs('epsg:3857'), how='right',max_distance=10000,distance_col='merge_dist')
        # stop_merged3_alt=geo_stops.to_crs('epsg:3857').sjoin_nearest(points_only2.to_crs('epsg:3857'), how='right',distance_col='merge_dist')
        # stop_merged3_only=geo_stops.to_crs('epsg:3857').sjoin_nearest(at_stops_positions_minDist.to_crs('epsg:3857'), how='right',distance_col='merge_dist')
        #%%
        print("Nombre de combinaisons uniques de ligne/direction/arrêt")
        print("Dans GTFS:", act.drop_duplicates(['route_short_name', 'direction_id','stop_id','stop_geo']).shape[0])
        print("Dans Positions:", at_stops_positions.drop_duplicates(['lineId', 'direction','pointId','geometry']).shape[0])
               
        #%%First try to merge
        if(False):
            act=act.sort_values('departure_datetime')
            at_stops_positions=at_stops_positions.sort_values('timestamp_dt_mean')
            merged=pd.merge_asof(act, at_stops_positions,
                    left_on='departure_datetime', right_on='timestamp_dt_mean',
                    left_by=['route_short_name', 'direction_id','stop_id'], right_by=['lineId', 'direction','pointId'],
                    #tolerance=pd.Timedelta("1h"),
                    #allow_exact_matches=False
                    direction='nearest'
                    )
            merged=merged.assign(delay=merged['timestamp_dt_mean']-merged['departure_datetime'])
            merged['sp_error']=merged['stop_geo'].to_crs('epsg:3857').distance(merged['geometry'].to_crs('epsg:3857'))
            merged.loc[merged['distanceFromPoint']!=0,'sp_error']=pd.NA
            
            #evaluate_fusion(merged, at_stops_positions)
            
            #Drop the lines of at_stops_positions fused more than once
            #Keep only the one with the lowest absolute delay
            merged['delay_abs']=merged['delay'].abs()
            merged=merged.sort_values('delay_abs')
            found_in_Positions=merged.dropna(how='all', subset=['pointId', 'lineId', 'direction', 'uuid'])
            not_merged_GTFS=merged[merged[['pointId', 'lineId', 'direction', 'uuid']].isna().any(axis=1)]
            merged_dupli=found_in_Positions[found_in_Positions.duplicated(subset=['pointId','lineId','direction','uuid'],keep=False)]
            t=merged_dupli.groupby(['pointId','lineId','direction','uuid']).nunique()
            found_in_Positions=found_in_Positions.mask(found_in_Positions.duplicated(subset=['pointId','lineId','direction','uuid'],keep='first'),
                                    other=lambda x: x[act.columns]).infer_objects(copy=False)
            merged=pd.concat([found_in_Positions,not_merged_GTFS])
        
        #%%Second try to merge from zero
        print("Deuxième tentative (stop_id tronqués)")
        act['stop_id2']=act['stop_id'].str.strip('ABCDEFGH').str.lstrip(to_strip='0')
        m=act.groupby('stop_id2').nunique()
        m[m['stop_id']>1].shape[0]#46 débuts de stop_id correspondent à 2 stop_id différents Mais, ils sont associés à des routes ou directions différentes
        m[(m['stop_id']>1)&(m['route_id']==1)&(m['direction_id']==1)].shape[0]#0
        if(m[(m['stop_id']>1)&(m['route_id']==1)&(m['direction_id']==1)].shape[0]!=0):
            print("Des stop_ids différents ont été amalgamés !!!")
        
        act=act.sort_values('departure_datetime')
        at_stops_positions=at_stops_positions.sort_values('timestamp_dt_mean')
        #TODO: Et si on essayait plutôt de fusionner chaque ligne de VehiclePositions avec une ligne du GTFS au lieu de l'inverse ?
        merged2=pd.merge_asof(act, at_stops_positions,
                left_on='departure_datetime', right_on='timestamp_dt_mean',
                left_by=['route_short_name', 'direction_id','stop_id2'], right_by=['lineId', 'direction','pointId'],
                #tolerance=pd.Timedelta("1h"),
                #allow_exact_matches=False
                direction='nearest'#'forward'#TODO: Décider
                )
        merged2=merged2.assign(delay=merged2['timestamp_dt_mean']-merged2['departure_datetime'])
        merged2['sp_error']=merged2['stop_geo'].to_crs('epsg:3857').distance(merged2['geometry'].to_crs('epsg:3857'))

        evaluate_fusion(merged2, at_stops_positions) 

        print("Dropping the lines incoherent with the next and previous ones...")        
        #Drop the lines incoherent with the previous and next ones
        merged2=merged2.sort_values(['trip_id','departure_datetime'])
        merged2.insert(21,'prev_next_geometry' ,merged2.groupby('trip_id')['next_geometry'].shift(periods=1))
        merged2.insert(23,'next_prev_pointId',merged2.groupby('trip_id')['previous_pointId'].shift(periods=-1))
        merged2.insert(19,'prev_uuid',merged2.groupby('trip_id')['uuid'].shift(periods=1))
        merged2.insert(21,'next_uuid',merged2.groupby('trip_id')['uuid'].shift(periods=-1))

        #TODO: Mettre dan sla fonction evaluate_fuison
        merged2_bad_nextGeo=merged2.eval("(prev_next_geometry!=geometry)&(~geometry.isna())&(~prev_next_geometry.isna())")
        merged2_bad_prevId=merged2.eval("(next_prev_pointId!=pointId)&(~pointId.isna())&(~next_prev_pointId.isna())")
        merged2_bad_uuid=merged2.eval("((prev_uuid!=uuid)&(next_uuid!=uuid))&"+#((prev_uuid==next_uuid)|(stop_sequence==1))&"+
                                                 "((~prev_uuid.isna())|(stop_sequence==1))&(~next_uuid.isna())&(~uuid.isna())")
        
        merged2.insert(25,'bad_nextGeo',merged2_bad_nextGeo)
        merged2.insert(25,'bad_prevId',merged2_bad_prevId)
        merged2.insert(25,'bad_uuid',merged2_bad_uuid)
        
        print(f"{merged2[merged2['bad_uuid']].shape[0]} lines dropped")
        merged2=merged2.mask(merged2['bad_uuid'],
                                other=lambda x: x[act.columns]).infer_objects(copy=False)
        merged2=merged2.drop(['bad_nextGeo','bad_prevId','bad_uuid'], axis=1)
        merged2=merged2.drop(['prev_next_geometry','next_prev_pointId','prev_uuid', 'next_uuid'], axis=1)
        
        #Drop the lines of at_stops_positions fused more than once
        #Keep only the one with the lowest absolute delay
        print("Dropping the duplicated lines...")
        merged2['delay_abs']=merged2['delay'].abs()
        merged2=merged2.sort_values('delay_abs')
        
        #Lines found in both GTFS and VehiclePositions
        merged2_both=merged2.dropna(how='all', subset=['timestamp_dt_mean', 'lineId', 'direction', 'uuid'])
        #Lines found only in GTFS
        not_merged2_GTFS=merged2[merged2[['timestamp_dt_mean', 'lineId', 'direction', 'uuid']].isna().any(axis=1)]
        #Lines found in both GTFS and VehiclePositions with values from VehiclePositions duplicated
        merged2_both_dupliPos=merged2_both[merged2_both.duplicated(subset=['timestamp_dt_mean','lineId','direction','uuid'],keep=False)]
        t=merged2_both_dupliPos.groupby(['timestamp_dt_mean','lineId','direction','uuid']).nunique()
        #Actually drop the lines of at_stops_positions fused more than once
        merged2_both=merged2_both.mask(merged2_both.duplicated(subset=['timestamp_dt_mean','lineId','direction','uuid'],keep='first'),
                                other=lambda x: x[act.columns]).infer_objects(copy=False)
        merged2_bis=pd.concat([merged2_both,not_merged2_GTFS])
        
        merged2_bis_both=merged2_bis.dropna(how='all', subset=['timestamp_dt_mean', 'lineId', 'direction', 'uuid'])
        merged2_bis_not_GTFS=merged2_bis[merged2_bis[['timestamp_dt_mean', 'lineId', 'direction', 'uuid']].isna().any(axis=1)].dropna(axis=1,how='all')
        found_in_Positions_unique2=merged2_both.dropna(subset=['timestamp_dt_mean', 'lineId', 'direction', 'uuid'], how='any').drop_duplicates(['timestamp_dt_mean','lineId','direction','uuid'])
        temp=found_in_Positions_unique2.merge(at_stops_positions, on=['timestamp_dt_mean','lineId','direction','uuid'], how='right', indicator=True, suffixes=('_x', None))
        merged2_bis_not_Positions=temp[temp['_merge']=='right_only'].drop('_merge', axis=1).dropna(axis=1, how='all')
        merged2_bis_not_GTFS=merged2_bis_not_GTFS.set_geometry('stop_geo')
        
        evaluate_fusion(merged2_bis, at_stops_positions)
        print(f"Merged              : {merged2_bis_both.shape[0]}")
        print(f"Not merged GTFS     : {merged2_bis_not_GTFS.shape[0]}")
        print(f"Not merged Positions:  {merged2_bis_not_Positions.shape[0]}")
        
        #%%Visualize the fusion results for each line and direction
        merged2_bis_both=merged2_bis.dropna(subset=['uuid'], how='all')
        # Visualize(["Avant fusion ","Après ","Pas fusion "],
        #           [act,merged2_bis_both,merged2_bis_not_GTFS],
        #           [at_stops_positions,merged2_bis_both,merged2_bis_not_Positions])
        #%% Merge iteratively based on next_geometry and previous_point_ID
        merge3_before=merged2_bis.copy()
        merge3_before_not_Positions=merged2_bis_not_Positions.copy()
        
        merge3_s=[]
        merge_i=1
        continue_merge=True
        while continue_merge:
            print(f"-----------------------------Iteration {merge_i}----------------------------")
            merge3_before_both=merge3_before.dropna(subset=['uuid'], how='all')
            merge3_before=merge3_before.sort_values(['trip_id','departure_datetime'])
            if('prev_next_geometry' in merge3_before):
                merge3_before['prev_next_geometry']=merge3_before.groupby('trip_id')['next_geometry'].shift(periods=1)
                merge3_before['next_prev_pointId']=merge3_before.groupby('trip_id')['previous_pointId'].shift(periods=-1)
                merge3_before['prev_uuid']=merge3_before.groupby('trip_id')['uuid'].shift(periods=1)
                merge3_before['next_uuid']=merge3_before.groupby('trip_id')['uuid'].shift(periods=-1)
    
            else:
                merge3_before.insert(21,'prev_next_geometry' ,merge3_before.groupby('trip_id')['next_geometry'].shift(periods=1))
                merge3_before.insert(23,'next_prev_pointId',merge3_before.groupby('trip_id')['previous_pointId'].shift(periods=-1))
                merge3_before.insert(19,'prev_uuid',merge3_before.groupby('trip_id')['uuid'].shift(periods=1))
                merge3_before.insert(21,'next_uuid',merge3_before.groupby('trip_id')['uuid'].shift(periods=-1))
    
            merge3_before_weird_uuid=merge3_before.query("((prev_uuid!=uuid)|(next_uuid!=uuid))&"+
                                                     "(~prev_uuid.isna())&(~next_uuid.isna())")
             
            #Lignes de merge3_before non fusionnés venant de GTFS
            merge3_before_not_GTFS=merge3_before[merge3_before[['timestamp_dt_mean', 'lineId', 'direction', 'uuid']].isna().any(axis=1)].dropna(axis=1,how='all')
            merge3_before_not_GTFS_with_npId=merge3_before_not_GTFS.dropna(subset='next_prev_pointId')
            merge3_before_not_GTFS_with_pnGeo=merge3_before_not_GTFS.dropna(subset='prev_next_geometry')
            merge3_before_not_GTFS_with_pnGeo_npID=merge3_before_not_GTFS.dropna(subset=['prev_next_geometry','next_prev_pointId'], how='all')
            merge3_before_not_GTFS_sans_pnGeoEtnpID=merge3_before_not_GTFS[merge3_before_not_GTFS[['prev_next_geometry','next_prev_pointId']].isna().all(axis=1)]
            print(f"Lines not merged mergeable: {merge3_before_not_GTFS_with_pnGeo_npID.shape[0]} on {merge3_before_not_GTFS.shape[0]}")
            #Merge on prev_next_geometry
            print("Merge on prev_next_geometry")
            merged3_prevGeo = merge3_before_not_GTFS_with_pnGeo.merge(merge3_before_not_Positions,
                                                           left_on=['route_short_name', 'direction_id','prev_next_geometry', 'prev_uuid'],
                                                           right_on=['lineId', 'direction','geometry','uuid'],
                                                           how='outer', indicator=True)
            merged3_prevGeo['_merge']=merged3_prevGeo['_merge'].cat.rename_categories({'left_only':'GTFS_only', 'right_only':'Positions_only'})
            print(merged3_prevGeo['_merge'].value_counts())
            merged3_prevGeo_both=merged3_prevGeo.query("(_merge=='both')").drop('_merge', axis=1)
            merged3_prevGeo_bothEtGTFS=merged3_prevGeo.query("(_merge=='both')|(_merge=='GTFS_only')")
            merged3_prevGeo=merged3_prevGeo.drop('_merge', axis=1)
            merged3_prevGeo_bothEtGTFS=merged3_prevGeo_bothEtGTFS.drop('_merge', axis=1)
    
            print("Merge on next_prev_pointId")
            merged3_nextId = merge3_before_not_GTFS_with_npId.merge(merge3_before_not_Positions,
                                                           left_on=['route_short_name', 'direction_id','next_prev_pointId','next_uuid'],
                                                           right_on=['lineId', 'direction','pointId','uuid'],
                                                           how='outer', indicator=True)
            merged3_nextId['_merge']=merged3_nextId['_merge'].cat.rename_categories({'left_only':'GTFS_only', 'right_only':'Positions_only'})
            print(merged3_nextId['_merge'].value_counts())
            merged3_nextId_both=merged3_nextId.query("(_merge=='both')").drop('_merge', axis=1)
            merged3_nextId_bothEtGTFS=merged3_nextId.query("(_merge=='both')|(_merge=='GTFS_only')")
            merged3_nextId=merged3_nextId.drop('_merge', axis=1)
            merged3_nextId_bothEtGTFS=merged3_nextId_bothEtGTFS.drop('_merge', axis=1)
            
            merged3_nextId_both=merged3_nextId_both.assign(delay=merged3_nextId_both['timestamp_dt_mean']-merged3_nextId_both['departure_datetime'])
            merged3_prevGeo_both=merged3_prevGeo_both.assign(delay=merged3_prevGeo_both['timestamp_dt_mean']-merged3_prevGeo_both['departure_datetime'])
            merged3_all = merged3_nextId_both.merge(merged3_prevGeo_both,
                            on=['route_short_name', 'direction_id','trip_id','stop_id'],
                            how='outer', indicator=True)
            print(merged3_all['_merge'].value_counts())
            merged3_all_both=merged3_all.query("_merge=='both'")
            merged3_all_both=merged3_all_both.sort_index(axis=1)
            merged3_all_both_bad=merged3_all_both.query("(uuid_x!=uuid_y)|(pointId_x!=pointId_y)")
            print(f"Incohérences: {merged3_all_both_bad.shape[0]}")
            
            merged3=pd.concat([merged3_prevGeo_bothEtGTFS, merged3_nextId_bothEtGTFS])
            
            merged3=merged3.assign(delay=merged3['timestamp_dt_mean']-merged3['departure_datetime'])
            merged3['sp_error']=merged3['stop_geo'].to_crs('epsg:3857').distance(merged3['geometry'].to_crs('epsg:3857'))
            #If lines are duplicated on GTFS values, keep the one with the smallest absolute delay
            #TODO: Keep only positive delay ?
            merged3['delay_abs']=merged3['delay'].abs()
            merged3=merged3.sort_values('delay_abs')
            merged3_dropped=merged3[merged3.duplicated(['route_short_name', 'direction_id','trip_id','stop_id'], keep='first')]
            merged3_dropped=merged3_dropped[merge3_before_not_GTFS.columns]
            merged3=merged3.drop_duplicates(['route_short_name', 'direction_id','trip_id','stop_id'], keep='first')
            
            evaluate_fusion(merged3, merge3_before_not_Positions)
            
            
            
            #Drop the lines of at_stops_positions fused more than once
            #Keep only the one with the lowest absolute delay
            print("Dropping the duplicated lines...")
            merged3_both=merged3.dropna(how='all', subset=['timestamp_dt_mean', 'lineId', 'direction', 'uuid'])
            not_merged3_GTFS=merged3[merged3[['timestamp_dt_mean', 'lineId', 'direction', 'uuid']].isna().any(axis=1)].dropna(how='all',axis=1)
            merged3_dupliPos=merged3_both[merged3_both.duplicated(subset=['timestamp_dt_mean','lineId','direction','uuid'],keep=False)]
            t=merged3_dupliPos.groupby(['timestamp_dt_mean','lineId','direction','uuid']).nunique()
            merged3_both=merged3_both.mask(merged3_both.duplicated(subset=['timestamp_dt_mean','lineId','direction','uuid'],keep='first'),
                                    other=lambda x: x[act.columns]).infer_objects(copy=False)
            merged3_bis=pd.concat([merged3_both,not_merged3_GTFS, merged3_dropped,merge3_before_not_GTFS_sans_pnGeoEtnpID])
            #Je rajoute également merge3_before_not_GTFS_sans_pnGeoEtnpID avec les lignes non fusionnables
            
            merged3_bis_not_GTFS=merged3_bis[merged3_bis[['timestamp_dt_mean', 'lineId', 'direction', 'uuid']].isna().any(axis=1)]
            merged3_both_uniquePos=merged3_both.drop_duplicates(['timestamp_dt_mean','lineId','direction','uuid'])
            temp=merged3_both_uniquePos.merge(merge3_before_not_Positions, on=['timestamp_dt_mean','lineId','direction','uuid'], how='right', indicator=True, suffixes=('_x', None))
            merged3_bis_not_Positions=temp[temp['_merge']=='right_only'].drop('_merge', axis=1)
            merged3_bis_not_GTFS=merged3_bis_not_GTFS.set_geometry('stop_geo')
            
              
            
            #Visualize the fusion results for each line and direction
            merged3_bis_both=merged3_bis.dropna(subset=['uuid'], how='all')
            if((merged3_bis_both.shape[0]==0)|merge_i>1):
                continue_merge=False
            if(False):#TODO: Supprimer
                Visualize([f"Avant fusion {merge_i} ",f"Après {merge_i} ",f"Pas fusion {merge_i} "],
                          [merge3_before_not_GTFS,merged3_bis_both,merged3_bis_not_GTFS],
                          [merge3_before_not_Positions,merged3_bis_both,merged3_bis_not_Positions])
            
            #Remets les lignes fusionnées précedement dans merged3_bis
            merged3_bis=pd.concat([merged3_bis,merge3_before_both])
            evaluate_fusion(merged3_bis, merge3_before_not_Positions)
            merged3_bis_both=merged3_bis.dropna(subset=['uuid'], how='all')
            
            merged3_bis.sort_values(['trip_id','departure_datetime'])
            merged3_bis['prev_next_geometry']=merged3_bis.groupby('trip_id')['next_geometry'].shift(periods=1)
            merged3_bis['next_prev_pointId']=merged3_bis.groupby('trip_id')['previous_pointId'].shift(periods=-1)
            merged3_bis['prev_uuid']=merged3_bis.groupby('trip_id')['uuid'].shift(periods=1)
            merged3_bis['next_uuid']=merged3_bis.groupby('trip_id')['uuid'].shift(periods=-1)
            
            merged3_bis_not_GTFS=merged3_bis[merged3_bis[['timestamp_dt_mean', 'lineId', 'direction', 'uuid']].isna().any(axis=1)]
            
            print(f"Merged              : {merged3_bis_both.shape[0]}")
            print(f"Not merged GTFS     : {merged3_bis_not_GTFS.shape[0]}")
            print(f"Not merged Positions:  {merged3_bis_not_Positions.shape[0]}")
            
            #Set up for the next iteration
            merge_i+=1
            merge3_before=merged3_bis.copy()
            merge3_before_not_Positions=merged3_bis_not_Positions.copy().dropna(how='all',axis=1)
            
            merged3_bis_not_GTFS_with_pnGeo_npID=merged3_bis_not_GTFS.dropna(subset=['prev_next_geometry','next_prev_pointId'], how='all')
            print(f"Lines not merged mergeable: {merged3_bis_not_GTFS_with_pnGeo_npID.shape[0]} on {merged3_bis_not_GTFS.shape[0]}")
            if(merged3_bis_not_GTFS_with_pnGeo_npID.shape[0]==0):
                continue_merge=False
        print(f"Lines of VehiclePositions still not merged: {len(merged3_bis_not_Positions)}")
        # #%%Third try
        # #merged2_bis_not_Positions_0=merged2_bis_not_Positions.query("distanceFromPoint==0")
        # #merged2_bis_not_Positions_1=merged2_bis_not_Positions.query("distanceFromPoint!=0")

        # merged3=merged2_bis_not_GTFS.merge(merged2_bis_not_Positions, 
        #                            left_on=['route_short_name', 'direction_id'],
        #                            right_on=['lineId', 'direction'],
        #                            how='outer', indicator=True)
        # print(merged3['_merge'].value_counts())
        
        # merged3=merged3.assign(delay=merged3['timestamp_dt_mean']-merged3['departure_datetime'])
        # merged3['sp_error']=merged3['stop_geo'].to_crs('epsg:3857').distance(merged3['geometry'].to_crs('epsg:3857'))
        
        # merged3_g=merged3.groupby(['route_short_name','direction_id','point_id'])
        # for name, group in merged3_g:
        #     print(name)
        #     group=group.assign(delay=group['timestamp_dt_mean']-group['departure_datetime'])
        #     group['delay_s']=group['delay'].dt.total_seconds()
        #     group['sp_error']=group['stop_geo'].to_crs('epsg:3857').distance(group['geometry'].to_crs('epsg:3857'))
        #     group.plot.scatter(x='delay_s',y='sp_error',title=name[0]+' '+name[1]+' '+str(name[2]))
        #     group.plot.scatter(x='departure_datetime',y=0,c='red')
        #     plt.show()
        # #%%Third try third method
        # merged2_bis_not_GTFS=merged2_bis_not_GTFS.set_geometry('stop_geo')
        # merged2_bis_not_GTFS_g=merged2_bis_not_GTFS.groupby(['route_short_name','direction_id'])
        # merged3_arr=[]
        # merged3_not_GTFS_arr=[]
        # merged3_not_Pos_arr=[]
        # for name, candidates_GTFS in merged2_bis_not_GTFS_g:
        #     lineId=name[0]
        #     dir=name[1]
        #     candidates_Positions=merged2_bis_not_Positions.query("(lineId==@lineId)&(direction==@dir)")
        #     print(f"Ligne {lineId} {dir}: {candidates_Positions.shape[0]} pour {candidates_GTFS.shape[0]}")
        #     correspondances=candidates_GTFS.merge(candidates_Positions, 
        #                           left_on=['route_short_name', 'direction_id'],
        #                           right_on=['lineId', 'direction'],
        #                           how='outer', indicator=True)
        #     correspondances=correspondances.sort_values('stop_sequence', ascending=False)
        #     stop_id=correspondances.iloc[0,:].loc['stop_id']
        #     group=correspondances.query("stop_id==@stop_id")
        #     group=group.assign(dist_to_0=group['stop_geo'].to_crs('epsg:3857').distance(group['geometry'].to_crs('epsg:3857')))
        #     candidates_GTFS['dist_to_0']=candidates_GTFS['stop_geo'].to_crs('epsg:3857').distance(group['stop_geo'].to_crs('epsg:3857').unique()[0])


        #     ax=candidates_GTFS.plot.scatter(x='departure_datetime', y='dist_to_0', c='red')
        #     group.plot.scatter(x='timestamp_dt_mean', y='dist_to_0', ax=ax,marker='x',
        #                        title=lineId+' '+str(dir)+' '+group['stop_name'].unique()[0], figsize=(9,6), grid=True)
        #     #ax.vlines(group['departure_datetime'],0,1000, colors="tab:red")
        #     plt.show()
        #     if(candidates_Positions.shape[0]!=0):
        #         #Clé de correspondance entre les géométries
        #         correspondances['sp_error']=correspondances['stop_geo'].to_crs('epsg:3857').distance(correspondances['geometry'].to_crs('epsg:3857'))
        #         geo_key=correspondances[['stop_geo', 'geometry','sp_error']].drop_duplicates(['stop_geo', 'geometry'])
        #         geo_key=geo_key.sort_values('sp_error')
        #         geo_key_Pos=geo_key.drop_duplicates(['geometry'], keep='first')#Pour chaque arrêt dans VehiclePos, le plus proche dans GTFS
        #         geo_key_GTFS=geo_key.drop_duplicates(['stop_geo'], keep='first')#Pour chaque arrêt dans VehiclePos, le plus proche dans GTFS

        #         geo_key_filtr=geo_key_Pos.query("sp_error<100")
        #         print(f"Premier filtre sur distance<100: {geo_key_filtr.shape[0]} arrêts restants sur {geo_key_Pos.shape[0]} dans VehiclePositions")
        #         print(f"{geo_key_filtr['stop_geo'].nunique()} sur {geo_key_GTFS.shape[0]} dans GTFS")
                
        #         correspondances2=candidates_GTFS.merge(geo_key_filtr, 
        #                               on='stop_geo',
        #                               how='outer', indicator=True)
        #         print(correspondances2['_merge'].value_counts())
        #         correspondances2=correspondances2.drop('_merge', axis=1)
        #         correspondances2=correspondances2.merge(candidates_Positions, 
        #                               on='geometry',
        #                               how='outer', indicator=True)
        #         print(correspondances2['_merge'].value_counts())
        #         correspondances2=correspondances2.drop('_merge', axis=1)
                
        #         correspondances2=correspondances2.assign(delay=correspondances2['timestamp_dt_mean']-correspondances2['departure_datetime'])
        #         correspondances2=correspondances2.assign(delay_abs=correspondances2['delay'].abs())
        #         correspondances2=correspondances2.sort_values('delay_abs', ascending=True)
        #         correspondances2_filter=correspondances2.drop_duplicates(['trip_id', 'stop_id'], keep='first')
        #         print(f"Ne garde que le retard absolu minimum: {correspondances2_filter.shape[0]} sur {correspondances2_filter.shape[0]}")
                
        #         #Retire les lignes de VehiclePositions dupliquées
        #         print("Dropping the duplicated lines...")
        #         found_in_Positions3=correspondances2_filter.dropna(how='all', subset=['timestamp_dt_mean', 'lineId', 'direction', 'uuid'])
        #         not_merged3_GTFS=correspondances2_filter[correspondances2_filter[['timestamp_dt_mean', 'lineId', 'direction', 'uuid']].isna().any(axis=1)]
        #         merged3_dupli=found_in_Positions3[found_in_Positions3.duplicated(subset=['timestamp_dt_mean','lineId','direction','uuid'],keep=False)]
        #         t=merged3_dupli.groupby(['timestamp_dt_mean','lineId','direction','uuid']).nunique()
        #         found_in_Positions3=found_in_Positions3.mask(found_in_Positions3.duplicated(subset=['timestamp_dt_mean','lineId','direction','uuid'],keep='first'),
        #                                 other=lambda x: x[act.columns]).infer_objects(copy=False)
        #         merged3=pd.concat([found_in_Positions3,not_merged3_GTFS])
                
        #         merged3_not_GTFS=merged3[merged3[['timestamp_dt_mean', 'lineId', 'direction', 'uuid']].isna().any(axis=1)].dropna(axis=1,how='all')
        #         found_in_Positions_unique3=found_in_Positions3.drop_duplicates(['timestamp_dt_mean','lineId','direction','uuid'])
        #         temp=found_in_Positions_unique3.merge(candidates_Positions, on=['timestamp_dt_mean','lineId','direction','uuid'], how='right', indicator=True, suffixes=('_x', None))
        #         merged3_not_Positions=temp[temp['_merge']=='right_only'].drop('_merge', axis=1).dropna(how='all',axis=1)
        #         print(f"{merged3['timestamp_dt_mean'].count()} lignes de VehiclePositions restantes")
                
        #         print(f"Résultat: {merged3['timestamp_dt_mean'].count()} trouvés sur {candidates_GTFS['stop_id'].count()}")
                
        #         merged3=merged3.sort_values('stop_sequence', ascending=False)
        #         merged3['dist_to_0_GTFS']=merged3['stop_geo'].to_crs('epsg:3857').distance(merged3['stop_geo'].to_crs('epsg:3857').unique()[0])
        #         merged3['dist_to_0_Pos']=merged3['geometry'].to_crs('epsg:3857').distance(merged3['stop_geo'].to_crs('epsg:3857').unique()[0])
        #         ax=merged3.plot.scatter(x='departure_datetime', y='dist_to_0_GTFS', c='red')
        #         merged3.plot.scatter(x='timestamp_dt_mean', y='dist_to_0_Pos', ax=ax,marker='x',
        #                            title=lineId+' '+str(dir)+' '+group['stop_name'].unique()[0], figsize=(9,6), grid=True)
                
        #         merged3.plot.scatter(x='dist_to_0_Pos', y='dist_to_0_GTFS', c='red')
        #         merged3['delay_s']=merged3['delay'].dt.total_seconds()
        #         merged3.plot.scatter(x='timestamp_dt_mean', y='delay_s', c='red')
        #         plt.show()
                
        #         merged3_arr.append(merged3)
        #         merged3_not_GTFS_arr.append(merged3_not_GTFS)
        #         merged3_not_Pos_arr.append(merged3_not_Positions)
        #     else:
        #         merged3_arr.append(candidates_GTFS)
        #         merged3_not_GTFS_arr.append(candidates_GTFS)
            
        # merged3=pd.concat(merged3_arr)
        # merged3_not_Pos=pd.concat(merged3_not_Pos_arr)
        # merged3_not_GTFS=pd.concat(merged3_not_GTFS_arr)
        # #%%Third try second method
        # merged2_bis_not_GTFS=merged2_bis_not_GTFS.set_geometry('stop_geo')
        # merged3=merged2_bis_not_GTFS.to_crs('epsg:3857').sjoin_nearest(merged2_bis_not_Positions.to_crs('epsg:3857'), how='left',distance_col='sp_error')
        
        
        # #Drop the rows where 'lineId'/'route_short_name', 'direction'/'direction_id' are not equal
        # #merged3=merged3.mask((merged3['lineId']!=merged3['route_short_name'])|(merged3['direction']!=merged3['direction_id']),
        # #                       other=lambda x: x[act.columns])
        # merged3=merged3.query("(lineId==route_short_name)&(direction==direction_id)")
        # merged3=merged3.assign(delay=merged3['timestamp_dt_mean']-merged3['departure_datetime'])
        # merged3=merged3.assign(delay_abs=merged3['delay'].abs())
        # merged3=merged3.sort_values('delay_abs')
        # print("Dropping the duplicated lines...")
        # merged3['delay_abs']=merged3['delay'].abs()
        # merged3=merged3.sort_values('delay_abs')
        # found_in_Positions3=merged3.dropna(how='all', subset=['pointId', 'lineId', 'direction', 'uuid'])
        # not_merged3_GTFS=merged3[merged3[['pointId', 'lineId', 'direction', 'uuid']].isna().any(axis=1)]
        # merged3_dupli=found_in_Positions3[found_in_Positions3.duplicated(subset=['pointId','lineId','direction','uuid'],keep=False)]
        # t=merged3_dupli.groupby(['pointId','lineId','direction','uuid']).nunique()
        # found_in_Positions3=found_in_Positions3.mask(found_in_Positions3.duplicated(subset=['pointId','lineId','direction','uuid'],keep='first'),
        #                         other=lambda x: x[act.columns]).infer_objects(copy=False)
        # merged3_bis=pd.concat([found_in_Positions3,not_merged3_GTFS])
        
        # merged3_bis_not_GTFS=merged3_bis[merged3_bis[['pointId', 'lineId', 'direction', 'uuid']].isna().any(axis=1)].dropna(axis=1,how='all')
        # found_in_Positions_unique2=found_in_Positions3.drop_duplicates(['pointId','lineId','direction','uuid'])
        # temp=found_in_Positions_unique2.merge(at_stops_positions, on=['pointId','lineId','direction','uuid'], how='right', indicator=True, suffixes=('_x', None))
        # merged3_bis_not_Positions=temp[temp['_merge']=='right_only'].dropna(how='all',axis=1).drop('_merge', axis=1)

        # evaluate_fusion(merged3_bis, merged2_bis_not_Positions)
        #%%Concatenating the results of the different merges
        final_merge=merged3_bis#pd.concat([merged2_bis])
        final_merge_not_GTFS=final_merge[final_merge[['lineId', 'direction', 'uuid']].isna().any(axis=1)].dropna(axis=1,how='all')
        final_merge_not_GTFS=pd.concat([final_merge_not_GTFS,act_route_notinPositions])
        final_merge_both=final_merge.dropna(subset=['uuid'], how='all')
        final_merge_both_uniquePos=final_merge_both.drop_duplicates(['timestamp_dt_mean','lineId','direction','uuid'])
        temp=final_merge_both_uniquePos.merge(at_stops_positions, on=['timestamp_dt_mean','lineId','direction','uuid'], how='right', indicator=True, suffixes=('_x', None))
        final_merge_not_Positions=temp[temp['_merge']=='right_only'].drop('_merge', axis=1)
        final_merge_not_Positions=final_merge_not_Positions.dropna(axis=1, how='all')
        
        #Mets aussi les lignes non fusionées dans final_merge
        final_merge=pd.concat([final_merge_not_Positions, final_merge]).reset_index(drop=True)
        print("Final result:")
        print(f"Lignes de GTFS non trouvées: {len(final_merge_not_GTFS)} sur {len(act)} ({len(act_route_notinPositions)} introuvables)")
        print(f"Lignes de VePo non trouvées: {len(final_merge_not_Positions)} sur {len(at_stops_positions)}")        
        
        #%%Save
        final_merge.to_pickle(data_out+'Fused_GTFS_VehiclePositions('+sdate+').pkl')
    except(FileNotFoundError):
        print(sdate,"manquant. On passe")
        yesterday_uuids=[]
#%%
    date=date+pd.Timedelta(value=1, unit='day')