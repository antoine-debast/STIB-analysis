# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 16:48:04 2025

@author: Aurélien
"""
#import gtfs_kit as gk

#fname='GTFS_2025-08-30T04-20-01.zip'

#feed = gk.read_feed(fname, 'm')

#%% Open the zip file 
import zipfile
import pandas as pd
import geopandas as gpd
from shapely import LineString
#%%
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
    lines = geo_shapes_sorted.groupby('shape_id', observed=True).apply(lambda x: LineString(x.geometry.tolist()))
    lines=lines.reset_index()
    lines = gpd.GeoDataFrame(data=lines['shape_id'], geometry=lines[0], crs=geo_shapes.crs)
    return lines

#%%
data_dir='data'
GTFS_zip_files=['GTFS_2025-08-13T04-20-01.zip', 'GTFS_2025-08-22T04-20-01.zip',
                'GTFS_2025-09-03T04-20-01.zip','GTFS_2025-09-04T04-20-01.zip','GTFS_2025-09-15T04-20-01.zip']

dico={'stops':[],'stop_times':[],'trips':[],'shapes':[],'routes':[],'calendar':[],'calendar_dates':[]}

for GTFS_zip_file in GTFS_zip_files:
    with zipfile.ZipFile(data_dir+'/'+GTFS_zip_file) as GTFS_dir:
        print(GTFS_dir.namelist())
        #GTFS_dir.extractall(GTFS_zip_file.split('.')[0]+"/",members=[])
        print("Feed infos")
        infos = pd.read_csv(GTFS_dir.open("feed_info.txt"),
                            usecols=["feed_start_date", "feed_end_date","feed_version"],
                            parse_dates=["feed_start_date", "feed_end_date"], date_format="%Y%m%d")
        print("Version:",infos['feed_version'][0])
        print("From",infos['feed_start_date'][0],"to",infos['feed_end_date'][0])

        print("Loading stops")
        stops = pd.read_csv(GTFS_dir.open("stops.txt"), usecols=['stop_id', 'stop_name', 'stop_lat',
               'stop_lon', 'location_type', 'parent_station'])
        print("Loading stop_times")
        stop_times = pd.read_csv(GTFS_dir.open("stop_times.txt"),
                                 usecols=['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence',
                                        'pickup_type', 'drop_off_type'],
                                 dtype={'stop_id':'string',
                                        'pickup_type':'boolean',
                                        'drop_off_type':'boolean',
                                        'stop_sequence':'uint8'},
                                parse_dates=["arrival_time", "departure_time"], date_format="%H:%M")
        print("Loading trips")
        trips = pd.read_csv(GTFS_dir.open("trips.txt"), dtype={'direction_id':'boolean'})
        print("Loading shapes")
        shapes = pd.read_csv(GTFS_dir.open("shapes.txt"), dtype={'shape_id':'category',
                                                            'shape_pt_sequence':'uint32'})
        print("Loading routes")
        routes = pd.read_csv(GTFS_dir.open("routes.txt"), usecols=['route_id','route_short_name','route_long_name','route_type'],
                            dtype={'route_type':'category'})
        #print("Loading agency")
        #agency = pd.read_csv(pd.read_csv(GTFS_dir.open("stops.txt")))
        print("Loading calendar")
        calendar = pd.read_csv(GTFS_dir.open("calendar.txt"),
                               dtype={'monday':'boolean',
                                    'tuesday':'boolean',
                                    'wednesday':'boolean',
                                    'thursday':'boolean',
                                    'friday':'boolean',
                                    'saturday':'boolean',
                                    'sunday':'boolean'},
                              parse_dates=["start_date", "end_date"], date_format="%Y%m%d")
        print("Loading calendar_dates")
        calendar_dates = pd.read_csv(GTFS_dir.open("calendar_dates.txt"),
                                     dtype={'exception_type':'category'},
                                     parse_dates=["date"], date_format="%Y%m%d")
        
        print("Files read")
        
        geo_stops=stops_to_geo(stops).drop(columns=['stop_lat','stop_lon'])
        lines=shapes_to_geo(shapes)
        
        geo_stops['source']=GTFS_zip_file
        stop_times['source']=GTFS_zip_file
        trips['source']=GTFS_zip_file
        lines['source']=GTFS_zip_file
        routes['source']=GTFS_zip_file
        calendar['source']=GTFS_zip_file
        calendar_dates['source']=GTFS_zip_file
        dico['stops'].append(geo_stops)
        dico['stop_times'].append(stop_times)
        dico['trips'].append(trips)
        dico['shapes'].append(lines)
        dico['routes'].append(routes)
        dico['calendar'].append(calendar)
        dico['calendar_dates'].append(calendar_dates)
#%% Fuse the GTFS's and try to clean the duplicates
#%% Stops
all_stops=pd.concat(dico['stops'])
all_stops=all_stops.drop_duplicates(['stop_id', 'stop_name', 'location_type',
       'parent_station', 'geometry'], keep='first')#Drop exact duplicates: 13.910 -> 3.405 lignes
all_stops.to_pickle("geo_stops(all).pkl")
#2739 points différents
#2885 ids différentes
all_stops['dupli_geo']=all_stops.duplicated(['geometry'], keep=False)#1088 duplicates
all_stops['dupli_id']=all_stops.duplicated(['stop_id'], keep=False)#520 duplicates
all_stops.duplicated(['stop_name', 'location_type', 'geometry'], keep=False).value_counts()#997 duplicates
#%%Stop_times
all_stop_times=pd.concat(dico['stop_times'])
all_stop_times=all_stop_times.drop_duplicates(['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence',
       'pickup_type', 'drop_off_type'], keep='first')#Drop exact duplicates: 9.191.679 -> 3.220.721 lignes
all_stop_times.to_pickle("stop_times(all).pkl")
all_stop_times.duplicated(['trip_id','arrival_time','departure_time'], keep=False).value_counts()#0 duplicates
all_stop_times.duplicated(['trip_id','stop_id'], keep='first').value_counts()#0 duplicates
#%%Trips
all_trips=pd.concat(dico['trips'])
all_trips=all_trips.drop_duplicates(['route_id', 'service_id', 'trip_id', 'trip_headsign', 'direction_id',
       'block_id', 'shape_id'], keep='first')#Drop exact duplicates: 452.067 -> 240.911  lignes
all_trips.to_pickle("trips(all).pkl")
#all_trips.duplicated(['trip_id'], keep=False).value_counts()#151.519 duplicates
all_trips['dupli_id']=all_trips.duplicated(['trip_id'], keep=False)
all_trips.duplicated(['trip_id','route_id'], keep=False).value_counts()#0 duplicates
#%%Shapes/Lines
all_lines=pd.concat(dico['shapes'])
all_lines=all_lines.drop_duplicates(['shape_id', 'geometry'], keep='first')#Drop exact duplicates: 3.902 -> 1.038  lignes
all_lines.to_pickle("lines(all).pkl")
all_lines.duplicated(['shape_id'], keep=False).value_counts()#44 duplicates
all_lines.duplicated(['geometry'], keep=False).value_counts()#229 duplicates
all_lines['dupli_id']=all_lines.duplicated(['shape_id'], keep=False)
#%%Routes
all_routes=pd.concat(dico['routes'])
all_routes=all_routes.drop_duplicates(['route_id', 'route_short_name', 
                'route_long_name', 'route_type'], keep='first')#Drop exact duplicates: 448 -> 197  lignes
all_routes.to_pickle("routes(all).pkl")
#90 ids
all_routes.duplicated(['route_id'], keep='first').value_counts()#191 duplicates
all_routes['dupli_id']=all_routes.duplicated(['route_id'], keep=False)
#%% Calendar
all_cal=pd.concat(dico['calendar'])
all_cal=all_cal.drop_duplicates(['service_id', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
       'saturday', 'sunday', 'start_date', 'end_date'], keep='first')#Drop exact duplicates: 2189 -> 1615 lignes
all_cal.to_pickle("calendar(all).pkl")
all_cal['dupli']=all_cal.duplicated(['service_id'], keep=False)
#%% Calendar_dates
all_dates=pd.concat(dico['calendar_dates'])
all_dates=all_dates.drop_duplicates(['service_id', 'date', 'exception_type'], keep='first')#Drop exact duplicates: 1.882 -> 1.026  lignes
all_dates.to_pickle("calendar_dates(all).pkl")
#371 ids
all_dates.duplicated(['service_id','date'], keep=False).value_counts()#0 duplicates
all_dates['dupli_id']=all_dates.duplicated(['service_id'], keep=False)
