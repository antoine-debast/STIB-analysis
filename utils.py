# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 10:15:04 2025

@author: Aurélien
"""
import pandas as pd
import geopandas as gpd
from shapely import LineString
import zipfile
GTFS_zip_files=['GTFS_2025-08-13T04-20-01.zip', 'GTFS_2025-08-22T04-20-01.zip',
                'GTFS_2025-09-03T04-20-01.zip','GTFS_2025-09-04T04-20-01.zip','GTFS_2025-09-15T04-20-01.zip']
GTFS_dates=['2025-08-13','2025-08-22','2025-09-03','2025-09-04','2025-09-15']
def stops_to_geo(stops):
    geometry=gpd.points_from_xy(stops['stop_lon'], stops['stop_lat'], z=None, crs='epsg:4326')
    geo_stops=gpd.GeoDataFrame(data=stops, geometry=geometry)
    return geo_stops
def shapes_to_geo(shapes):
    geometry=gpd.points_from_xy(shapes['shape_pt_lon'], shapes['shape_pt_lat'], z=None, crs='epsg:4326')
    geo_shapes=gpd.GeoDataFrame(data=shapes, geometry=geometry)
    #Sort the lines by id and Sequence so they are ordered
    geo_shapes_sorted = geo_shapes.sort_values(by=['shape_id', 'shape_pt_sequence'])
    
    #Group the points by 'route'
    lines = geo_shapes_sorted.groupby('shape_id', observed=True).apply(lambda x: LineString(x.geometry.tolist()),include_groups=False)
    lines=lines.reset_index()
    lines = gpd.GeoDataFrame(data=lines['shape_id'], geometry=lines[0], crs=geo_shapes.crs)
    return lines
def read_gtfs(GTFS_zip_file,data_dir):
    with zipfile.ZipFile(data_dir+'/'+GTFS_zip_file) as GTFS_dir:
        #print(GTFS_dir.namelist())
        #GTFS_dir.extractall(GTFS_zip_file.split('.')[0]+"/",members=[])
        print("Feed infos: "+GTFS_zip_file)
        infos = pd.read_csv(GTFS_dir.open("feed_info.txt"),
                            usecols=["feed_start_date", "feed_end_date","feed_version"],
                            parse_dates=["feed_start_date", "feed_end_date"], date_format="%Y%m%d")
        print("Version:",infos['feed_version'][0])
        print("From",infos['feed_start_date'][0],"to",infos['feed_end_date'][0])

        print("\rLoading stops", end='')
        stops = pd.read_csv(GTFS_dir.open("stops.txt"), usecols=['stop_id', 'stop_name', 'stop_lat',
               'stop_lon', 'location_type', 'parent_station'])
        print("\rLoading stop_times", end='')
        stop_times = pd.read_csv(GTFS_dir.open("stop_times.txt"),
                                 usecols=['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence',
                                        'pickup_type', 'drop_off_type'],
                                 dtype={'stop_id':'string',
                                        'pickup_type':'boolean',
                                        'drop_off_type':'boolean',
                                        'stop_sequence':'uint8',
                                        'trip_id':'string'},
                                parse_dates=["arrival_time", "departure_time"], date_format="%H:%M")
        print("\rLoading trips         ", end='')
        trips = pd.read_csv(GTFS_dir.open("trips.txt"), dtype={'direction_id':'boolean', 'trip_id':'string'})
        print("\rLoading shapes        ", end='')
        shapes = pd.read_csv(GTFS_dir.open("shapes.txt"), dtype={'shape_id':'category',
                                                            'shape_pt_sequence':'uint32'})
        print("\rLoading routes        ", end='')
        routes = pd.read_csv(GTFS_dir.open("routes.txt"), usecols=['route_id','route_short_name','route_long_name','route_type'],
                            dtype={'route_type':'category'})
        #print("Loading agency")
        #agency = pd.read_csv(pd.read_csv(GTFS_dir.open("stops.txt")))
        print("\rLoading calendar", end='')
        calendar = pd.read_csv(GTFS_dir.open("calendar.txt"),
                               dtype={'monday':'boolean',
                                    'tuesday':'boolean',
                                    'wednesday':'boolean',
                                    'thursday':'boolean',
                                    'friday':'boolean',
                                    'saturday':'boolean',
                                    'sunday':'boolean'},
                              parse_dates=["start_date", "end_date"], date_format="%Y%m%d")
        print("\rLoading calendar_dates", end='')
        calendar_dates = pd.read_csv(GTFS_dir.open("calendar_dates.txt"),
                                     dtype={'exception_type':'category'},
                                     parse_dates=["date"], date_format="%Y%m%d")
        
        print("\rFiles read              ")
        
        geo_stops=stops_to_geo(stops).drop(columns=['stop_lat','stop_lon'])
        lines=shapes_to_geo(shapes)
        
        geo_stops['source']=GTFS_zip_file
        stop_times['source']=GTFS_zip_file
        trips['source']=GTFS_zip_file
        lines['source']=GTFS_zip_file
        routes['source']=GTFS_zip_file
        calendar['source']=GTFS_zip_file
        calendar_dates['source']=GTFS_zip_file
        
        return geo_stops, stop_times, trips, lines, routes, calendar, calendar_dates
#%%
def get_active_services(date, calendar, calendar_dates):
    week_day=pd.Timestamp(date).day_name().lower()
    active_services=calendar.query("(start_date<=@date)&(@date<=end_date)")
    active_services=active_services[active_services[week_day]]
    
    exceptions_ajoute=calendar_dates.query("(exception_type=='1')&(date==@date)")
    exceptions_ajoute=exceptions_ajoute['service_id']
    exceptions_retire=calendar_dates.query("(exception_type=='2')&(date==@date)")
    exceptions_retire=exceptions_retire['service_id']
    
    added_services=calendar[calendar['service_id'].isin(exceptions_ajoute)]
    if(added_services.index.intersection(active_services.index).size!=0):
        print("Services d'exceptions déjà présents")
    active_services=active_services[~active_services['service_id'].isin(exceptions_retire)]
    active_services=pd.concat([active_services,added_services])
    
    return active_services
def get_active_stop_times(date, trips, stop_times, calendar, calendar_dates):
    #date=pd.Timestamp(date)
    
    active_services=get_active_services(date, calendar, calendar_dates)
    
    active_trips=trips[trips['service_id'].isin(active_services['service_id'])]
    active_trips_id=active_trips['trip_id']
    active_stop_times=stop_times[stop_times['trip_id'].isin(active_trips_id)]
    return active_stop_times, active_trips
#%% VehiclePositions
def get_at_stop_positions(day_positions):
    print("Loading Id_geo_key...")
    
    try:
        Id_geo_key=pd.read_pickle('data/Id_geo_key.pkl')
    except FileNotFoundError:
        print("Id_geo_key not found")
        print("Generating it...")
        import pointId_geo_link_Positions#TODO: Pas sûr
    
    #TODO: résoudre les dupliqués
    dupliId_geo_key=Id_geo_key[Id_geo_key.duplicated(['geometry','lineId'], keep=False)]
    if(dupliId_geo_key.shape[0]>0):
        print("Attention! Des geometry sont associées à plusieurs pointId")
        print(dupliId_geo_key)
    #Keep only the positions with the earliest timestamp for each stop, line, direction and vehicle
    #TODO: Vérifier que ca correspond bien au moments où les véhicules sont aux arrêts
    #TODO: Gérer les véhicules qui restent trop longtemps à un arrêt
    day_positions=day_positions.sort_values('timestamp_dt')
    post_stops_positions_0=day_positions.query("distanceFromPoint==0").drop_duplicates(['pointId','lineId','direction','uuid'], keep='last')
    post_stops_positions_1=day_positions.query("distanceFromPoint!=0").drop_duplicates(['pointId','lineId','direction','uuid'], keep='first')
    #post_stops_positions_minDist=day_positions.loc[day_positions.drop('geometry',axis=1).groupby(['pointId','lineId','direction','uuid']).idxmin()['distanceFromPoint'],:]
    
    
    #ante_stops_positions=day_positions.sort_values('timestamp_dt').groupby(['geometry','lineId','direction','uuid']).last()
    #ante_stops_positions=ante_stops_positions.reset_index()
    #ante_stops_positions['geometry']=ante_stops_positions['geometry'].set_crs("EPSG:4326")
    ante_stops_positions_0=day_positions.query("distanceFromPoint==0").drop_duplicates(['geometry','lineId','direction','uuid'], keep='first')
    ante_stops_positions_1=day_positions.query("distanceFromPoint!=0").drop_duplicates(['geometry','lineId','direction','uuid'], keep='last')
    # tempa=post_stops_positions[(post_stops_positions['distanceFromPoint']!=post_stops_positions_minDist['distanceFromPoint'])]
    # tempb=post_stops_positions_minDist[(post_stops_positions['distanceFromPoint']!=post_stops_positions_minDist['distanceFromPoint'])]
    # tempb['error']=(tempb['timestamp_dt']-tempa['timestamp_dt'])
    # tempa['error']=(tempb['timestamp_dt']-tempa['timestamp_dt'])
    
    # problem_stops=(tempa.value_counts("pointId")/post_stops_positions.value_counts("pointId")).dropna(axis=0).sort_values(ascending=False)
    # problem_lines=(tempa.value_counts("lineId")/post_stops_positions.value_counts("lineId")).dropna(axis=0).sort_values(ascending=False)
    # problem_geo=(tempa.value_counts("geometry")/post_stops_positions.value_counts("geometry")).dropna(axis=0).sort_values(ascending=False)
    
    ante_stops_positions_1=ante_stops_positions_1.merge(Id_geo_key,on=['geometry','lineId'], how='outer', indicator=True)
    print(ante_stops_positions_1['_merge'].value_counts())
    post_stops_positions_1=post_stops_positions_1.merge(Id_geo_key,on=['pointId','lineId'], how='outer', indicator=True)
    print(post_stops_positions_1['_merge'].value_counts())
    
    ante_stops_positions_1_not_merged=ante_stops_positions_1.query("_merge=='left_only'")
    post_stops_positions_1_not_merged=post_stops_positions_1.query("_merge=='left_only'")
    ante_stops_positions_1=ante_stops_positions_1.query("_merge!='right_only'").drop('_merge', axis=1)
    post_stops_positions_1=post_stops_positions_1.query("_merge!='right_only'").drop('_merge', axis=1)
    
    ante_stops_positions_1=ante_stops_positions_1.sort_values(['uuid','timestamp_dt'])[['lineId', 'direction','geometry','pointId_x','pointId_y',
                                                                                    'distanceFromPoint', 'uuid', 'distance', 'timestamp_dt','id']]
    post_stops_positions_1=post_stops_positions_1.sort_values(['uuid','timestamp_dt'])[['lineId', 'direction','geometry_x','geometry_y','pointId',
                                                                                    'distanceFromPoint', 'uuid', 'distance', 'timestamp_dt','id']]
    
    ante_stops_positions=ante_stops_positions_0.merge(ante_stops_positions_1, how='outer', indicator=True,
                                                      suffixes=('_0','_1'),
                                                      left_on=['geometry','lineId','direction','uuid'],
                                                      right_on=['geometry','lineId','direction','uuid'])
    print("Ante merge results:")
    print(ante_stops_positions['_merge'].value_counts())
    ante_stops_positions=ante_stops_positions.rename({'pointId_x':'previous_pointId',
                                                      'pointId_y':'pointId_1',
                                                      'pointId':'pointId_0'}, axis=1)

    
    post_stops_positions=post_stops_positions_0.merge(post_stops_positions_1, how='outer', indicator=True,
                                                      suffixes=('_0','_1'),
                                                      left_on=['pointId','lineId','direction','uuid'],
                                                      right_on=['pointId','lineId','direction','uuid'])
    print("Post merge results:")
    print(post_stops_positions['_merge'].value_counts())
    post_stops_positions=post_stops_positions.rename({'geometry_x':'next_geometry',
                                                      'geometry_y':'geometry_1',
                                                      'geometry':'geometry_0'}, axis=1)
    
    ante_stops_positions['pointId']=ante_stops_positions['pointId_0'].where(pd.notna(ante_stops_positions['pointId_0']), ante_stops_positions['pointId_1'])
    ante_stops_positions['distanceFromPoint']=ante_stops_positions['distanceFromPoint_0'].where(pd.notna(ante_stops_positions['distanceFromPoint_0']), ante_stops_positions['distanceFromPoint_1'])
    ante_stops_positions['timestamp_dt']=ante_stops_positions['timestamp_dt_0'].where(pd.notna(ante_stops_positions['distanceFromPoint_0']), ante_stops_positions['timestamp_dt_1'])
    ante_stops_positions=ante_stops_positions[['lineId', 'direction', 'uuid',
                                               'geometry', 'pointId', 'previous_pointId',
                                               'timestamp_dt','distanceFromPoint',
                                               'timestamp_dt_0', 'timestamp_dt_1',
                                               'pointId_0', 'pointId_1',
                                               'distanceFromPoint_1',
                                               'distance_0','distance_1','id_0','id_1',]]
    ante_stops_positions_bad=ante_stops_positions.query("(pointId_0!=pointId_1)&~pointId_0.isna()&~pointId_1.isna()")
    if(len(ante_stops_positions_bad)!=0):
        print("Fusion of ante_stops_positions_0 and ante_stops_positions_1 incorrecte !!!!")
        #TODO: Trouvez pourquoi elles sont incorrectes (voir dans Id_geo_key ?)
        print("Exclusion des lignes incorrectes...")
        ante_stops_positions=ante_stops_positions.query("(pointId_0==pointId_1)|pointId_0.isna()|pointId_1.isna()")
        

    post_stops_positions['geometry']=post_stops_positions['geometry_0'].where(pd.notna(post_stops_positions['geometry_0']), post_stops_positions['geometry_1'])
    post_stops_positions['distanceFromPoint']=post_stops_positions['distanceFromPoint_0'].where(pd.notna(post_stops_positions['distanceFromPoint_0']), post_stops_positions['distanceFromPoint_1'])
    post_stops_positions['timestamp_dt']=post_stops_positions['timestamp_dt_0'].where(pd.notna(post_stops_positions['distanceFromPoint_0']), post_stops_positions['timestamp_dt_1'])
    post_stops_positions=post_stops_positions[['lineId', 'direction', 'uuid',
                                               'geometry', 'pointId', 'next_geometry',
                                               'timestamp_dt','distanceFromPoint',
                                               'timestamp_dt_0', 'timestamp_dt_1',
                                               'geometry_0', 'geometry_1',
                                               'distanceFromPoint_1',
                                               'distance_0','distance_1','id_0','id_1',]]
    #Check the good fusion of post_stops_positions_0 and post_stops_positions_1
    post_stops_positions_bad=post_stops_positions.query("(geometry_0!=geometry_1)&~geometry_0.isna()&~geometry_1.isna()")
    if(len(post_stops_positions_bad)!=0):
        print("Fusion of post_stops_positions_0 and post_stops_positions_1 incorrecte !!!!")
        print("Exclusion des lignes incorrectes...")
        post_stops_positions=post_stops_positions.query("(geometry_0==geometry_1)|geometry_0.isna()|geometry_1.isna()")

        
    at_stops_positions=ante_stops_positions.merge(post_stops_positions, how='outer', indicator=True,
                                                  suffixes=('_pre','_post'),
                                                  left_on=['pointId','geometry','lineId','direction','uuid'],
                                                  right_on=['pointId','geometry','lineId','direction','uuid'])
    print(at_stops_positions['_merge'].value_counts())
    at_stops_positions=at_stops_positions.assign(timestamp_dt_mean=at_stops_positions[['timestamp_dt_pre','timestamp_dt_post']].mean(axis=1))
    at_stops_positions=at_stops_positions[['lineId', 'direction', 'uuid',
                                           'geometry','next_geometry',
                                           'pointId','previous_pointId',
                                           'distanceFromPoint_pre','distanceFromPoint_post',
                                           'timestamp_dt_pre','timestamp_dt_mean','timestamp_dt_post',
                                           'pointId_0', 'pointId_1',
                                           'geometry_0', 'geometry_1',
                                           'timestamp_dt_0_pre', 'timestamp_dt_1_pre','timestamp_dt_0_post', 'timestamp_dt_1_post',
                                           'distanceFromPoint_1_pre','distanceFromPoint_1_post',
                                           'distance_0_pre', 'distance_1_pre','distance_0_post', 'distance_1_post', 
                                           'id_0_pre', 'id_1_pre','id_0_post', 'id_1_post']]

    #at_stops_positions['geometry']=at_stops_positions['geometry'].where(pd.notna(at_stops_positions['geometry']), at_stops_positions['geometry_y'])
    #at_stops_positions['pointId']=at_stops_positions['pointId'].where(pd.notna(at_stops_positions['pointId']), at_stops_positions['pointId_y'])
    at_stops_positions=at_stops_positions.sort_values(['uuid','timestamp_dt_mean'])
    #Check if 'timestamp_dt_pre' is before 'timestamp_dt_post' as it is should
    at_stops_positions_bad=at_stops_positions.query("timestamp_dt_pre>timestamp_dt_post")
    if(at_stops_positions_bad.shape[0]>0):
        print("Attention! Des lignes ont des temps inversés")
        print(at_stops_positions_bad)
    
    at_stops_positions_big=at_stops_positions.copy()
    at_stops_positions=at_stops_positions[['lineId', 'direction', 'uuid', 'geometry', 'next_geometry', 'pointId',
           'previous_pointId', 'distanceFromPoint_pre', 'distanceFromPoint_post',
           'timestamp_dt_pre', 'timestamp_dt_mean', 'timestamp_dt_post']]
    return at_stops_positions
#%%
if __name__=="__main__":
    date="2025-09-03"
    data_dir="data"
    for i in range(1,len(GTFS_dates)):
        if(GTFS_dates[i]>date):
            active_GTFS_zip_file=GTFS_zip_files[i-1]
            break
    _, stop_times, trips, _, _, calendar, calendar_dates=read_gtfs(active_GTFS_zip_file,data_dir)
    #%%
    active_stop_times=get_active_stop_times(date, trips, stop_times, calendar, calendar_dates)