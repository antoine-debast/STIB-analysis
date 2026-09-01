# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 12:25:57 2025
Modified to perform Time Window Clustering Analysis

@author: Aurélien
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
import utils
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Filter warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- CONFIGURATION ---
GTFS_zip_files=['GTFS_2025-08-13T04-20-01.zip', 'GTFS_2025-08-22T04-20-01.zip',
                'GTFS_2025-09-03T04-20-01.zip','GTFS_2025-09-04T04-20-01.zip','GTFS_2025-09-15T04-20-01.zip']
GTFS_dates=['2025-08-13','2025-08-22','2025-09-03','2025-09-04','2025-09-15']
data_dir="data"

sdate_debut="2025-09-02"
sdate_fin="2025-09-02" # Process one day as requested
date_fin=pd.to_datetime(sdate_fin)
date=pd.to_datetime(sdate_debut)

GTFS_i=1
active_GTFS_zip_file=GTFS_zip_files[GTFS_i]

# --- CLUSTERING FUNCTION DEFINITION ---
def perform_clustering_analysis(df, transport_mode_name, route_types, n_clusters):
    """
    Performs K-Means clustering on hourly headway statistics (Mean vs Std)
    to identify homogeneous time windows.
    """
    print(f"\n>>> Analyzing {transport_mode_name}...")
    
    # 1. Filter Data for specific route types
    # Ensure route_type is string to match the format in 'r' dataframe
    subset = df[df['route_type'].astype(str).isin([str(r) for r in route_types])].copy()
    
    if subset.empty:
        print(f"No data found for {transport_mode_name} (Types: {route_types})")
        return

    # 2. Feature Engineering: Group by Hour
    subset['hour'] = subset['departure_datetime'].dt.hour
    
    # Calculate Mean (Frequency) and Std (Regularity) per hour
    hourly_stats = subset.groupby('hour')['headway_s'].agg(
        mean_headway='mean', 
        std_headway='std', 
        count='count'
    ).reset_index()
    
    # Filter out hours with very little data (e.g. 3 AM start ups)
    hourly_stats = hourly_stats[hourly_stats['count'] > 50].dropna()

    if hourly_stats.empty:
        print("Not enough data to cluster.")
        return

    # 3. Clustering (K-Means)
    X = hourly_stats[['mean_headway', 'std_headway']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    hourly_stats['cluster'] = kmeans.fit_predict(X_scaled)

    # 4. Sort clusters logically (0 = Best frequency, N = Worst)
    cluster_mapping = hourly_stats.groupby('cluster')['mean_headway'].mean().sort_values().index
    mapping_dict = {old: new for new, old in enumerate(cluster_mapping)}
    hourly_stats['cluster_sorted'] = hourly_stats['cluster'].map(mapping_dict)

    # 5. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Plot A: Scatter (Profile Definition)
    sns.scatterplot(
        data=hourly_stats, x='mean_headway', y='std_headway', 
        hue='cluster_sorted', palette='viridis', s=150, ax=axes[0], legend='full'
    )
    # Add hour labels
# %%
# %%
# %%
    for line in range(0, hourly_stats.shape[0]):
# %%
# %%
# %%
        axes[0].text(
            hourly_stats.mean_headway.iloc[line]+2, 
            hourly_stats.std_headway.iloc[line], 
            str(int(hourly_stats.hour.iloc[line])), 
            horizontalalignment='left', size='medium', color='black', weight='bold'
        )
    axes[0].set_title(f'{transport_mode_name}: Service Profiles')
    axes[0].set_xlabel('Mean Headway (s)')
    axes[0].set_ylabel('Std Dev (s)')
    axes[0].grid(True, linestyle='--')

    # Plot B: Timeline (The Time Groups)
    sns.barplot(
        data=hourly_stats, x='hour', y='mean_headway', 
        hue='cluster_sorted', palette='viridis', dodge=False, ax=axes[1]
    )
    axes[1].set_title(f'{transport_mode_name}: Segmentation Timeline')
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Avg Headway (s)')
    axes[1].grid(axis='y')

    plt.tight_layout()
    plt.show()

# --- MAIN DATA PROCESSING LOOP ---

# Initial Read
geo_stops, stop_times, trips, lines, routes, calendar, calendar_dates = utils.read_gtfs(active_GTFS_zip_file,data_dir)

while date<=date_fin:
    sdate=str(date.date())
    print("-------------------"+sdate+"-------------------")
    date_dt=pd.to_datetime(sdate) # Renamed to avoid confusion with loop var
    
    # Check if we need to switch GTFS file
    if(sdate>GTFS_dates[GTFS_i+1]):
        GTFS_i=GTFS_i+1
        active_GTFS_zip_file=GTFS_zip_files[GTFS_i]
        geo_stops, stop_times, trips, lines, routes, calendar, calendar_dates = utils.read_gtfs(active_GTFS_zip_file,data_dir)
    
    active_stop_times, active_trips = utils.get_active_stop_times(sdate, trips, stop_times, calendar, calendar_dates)
    
    # Convert time columns
    # Note: date variable in lambda refers to current loop date
    active_stop_times=active_stop_times.assign(
        arrival_datetime=active_stop_times['arrival_time'].str.split(':').apply(
            lambda x:date_dt+datetime.timedelta(hours=int(x[0]), minutes=int(x[1]), seconds=int(x[2]))),
        departure_datetime=active_stop_times['departure_time'].str.split(':').apply(
            lambda x:date_dt+datetime.timedelta(hours=int(x[0]), minutes=int(x[1]), seconds=int(x[2]))))
    
    # Merge stop_times and routes
    act=active_trips[['route_id', 'trip_id', 'direction_id', 'shape_id']].merge(active_stop_times, on='trip_id', how='outer')
    r=routes[['route_id', 'route_short_name','route_type']]
    # Ensure route_short_name is string
    r = r.assign(route_short_name=r['route_short_name'].astype('string'))
    
    act=act.merge(r, on='route_id', how='outer', indicator=True)
    print("Merge Results (Stop Times + Routes):")
    print(act.drop_duplicates('route_id')['_merge'].value_counts())
    
    # Filter valid merges
    act=act[act['_merge']!="right_only"]
    act=act.drop('_merge',axis=1)
    
    # Merge stops
    act=geo_stops[['stop_id', 'stop_name', 'geometry']].merge(act, on='stop_id', how='right')
    
    # Calculate Headways
    stop_key='stop_name'
    act=act.sort_values([stop_key,'route_id','direction_id','arrival_datetime'])
    d=act[[stop_key,'route_id','direction_id', 'departure_datetime']].groupby([stop_key,'route_id','direction_id']).diff()
    act=act.assign(delta=(act['departure_datetime']-act['arrival_datetime']))
    act=act.assign(headway=d['departure_datetime']-act['delta'])
    act=act.assign(headway_s=act['headway'].dt.seconds)
    
    # --- CLUSTERING EXECUTION ---
    
    # 1. Metro Clustering (Route Type 1)
    # Using 6 clusters to capture the sharp "Step" transitions in Metro schedules
    perform_clustering_analysis(
        act, 
        transport_mode_name="Metro (Line 1, 2, 5, 6)", 
        route_types=['1'], 
        n_clusters=6
    )

    # 2. Surface Clustering (Tram=0, Bus=3)
    # Using 5 clusters to capture variance-based transitions (Peak, Off-peak, Evening, Night, Early)
    perform_clustering_analysis(
        act, 
        transport_mode_name="Surface (Bus & Tram)", 
        route_types=['0', '3'], 
        n_clusters=5
    )

    # Increment Date
    date=date+pd.Timedelta(value=1, unit='day')