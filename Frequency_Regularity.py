# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime 
import utils
import warnings
import os

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")

GTFS_zip_files=['GTFS_2025-08-13T04-20-01.zip', 'GTFS_2025-08-22T04-20-01.zip',
                'GTFS_2025-09-03T04-20-01.zip','GTFS_2025-09-04T04-20-01.zip','GTFS_2025-09-15T04-20-01.zip']
GTFS_dates=['2025-08-13','2025-08-22','2025-09-03','2025-09-04','2025-09-15']
data_dir="data"

# Define the date range you want to analyze
sdate_debut="2025-09-01 "
sdate_fin="2025-09-16"
date_fin=pd.to_datetime(sdate_fin)
date=pd.to_datetime(sdate_debut)

GTFS_i=1
active_GTFS_zip_file=GTFS_zip_files[GTFS_i]

# =============================================================================
#  SECTION 1: PROFILING LOGIC (METRO VS SURFACE)
# =============================================================================

def get_surface_profile(hour_float):
    """Profile A: Bus & Tram"""
    if hour_float < 7.0: return "1. Morning Ramp-up"
    elif 7.0 <= hour_float < 9.5: return "2. AM Rush"
    elif 9.5 <= hour_float < 15.5: return "3. Day Base"
    elif 15.5 <= hour_float < 18.5: return "4. School/PM Rush"
    elif 18.5 <= hour_float < 21.0: return "5. Evening Slide"
    else: return "6. Night"

def get_metro_profile(hour_float):
    """Profile B: Metro"""
    if hour_float < 6.5: return "1. Early Service"
    elif 6.5 <= hour_float < 9.5: return "2. AM Peak"
    elif 9.5 <= hour_float < 16.0: return "3. Day Base"
    elif 16.0 <= hour_float < 19.0: return "4. PM Peak"
    elif 19.0 <= hour_float < 22.0: return "5. Evening"
    else: return "6. Late Night"

# =============================================================================
#  SECTION 2: MAIN PROCESSING LOOP
# =============================================================================

# Container for all results
classification_results = []

geo_stops, stop_times, trips, lines, routes, calendar, calendar_dates = utils.read_gtfs(active_GTFS_zip_file, data_dir)
all_lines = sorted(routes['route_short_name'].dropna().astype(str).unique())

while date <= date_fin:
    sdate = str(date.date())
    print(f"--- Processing Schedule for {sdate} ---")
    
    # Switch GTFS file if needed
    if sdate > GTFS_dates[GTFS_i+1]:
        GTFS_i += 1
        active_GTFS_zip_file = GTFS_zip_files[GTFS_i]
        geo_stops, stop_times, trips, lines, routes, calendar, calendar_dates = utils.read_gtfs(active_GTFS_zip_file, data_dir)
    
    active_stop_times, active_trips = utils.get_active_stop_times(sdate, trips, stop_times, calendar, calendar_dates)
    
    # Time Conversion
    active_stop_times = active_stop_times.assign(
        arrival_datetime=active_stop_times['arrival_time'].str.split(':').apply(
            lambda x: date + datetime.timedelta(hours=int(x[0]), minutes=int(x[1]), seconds=int(x[2]))),
        departure_datetime=active_stop_times['departure_time'].str.split(':').apply(
            lambda x: date + datetime.timedelta(hours=int(x[0]), minutes=int(x[1]), seconds=int(x[2])))
    )

    # Merging
    act = active_trips[['route_id','trip_id','direction_id']].merge(active_stop_times, on='trip_id')
    r = routes[['route_id','route_short_name','route_type']]
    r = r.assign(route_short_name=r['route_short_name'].astype('string'))
    act = act.merge(r, on='route_id')
    act = geo_stops[['stop_id','stop_name']].merge(act, on='stop_id', how='right')

    # Calculate Headway
    stop_key = 'stop_name'
    act = act.sort_values([stop_key, 'route_id', 'direction_id', 'arrival_datetime'])
    d = act[[stop_key, 'route_id', 'direction_id', 'departure_datetime']].groupby([stop_key, 'route_id', 'direction_id']).diff()
    act['delta'] = act['departure_datetime'] - act['arrival_datetime']
    act['headway'] = d['departure_datetime'] - act['delta']
    act['headway_s'] = act['headway'].dt.seconds

    # Filter Valid & Convert
    act = act.dropna(subset=['headway_s'])
    act = act[act['headway_s'] < 7200] 
    
    # Calculate Float Hour for Profiling
    act['hour'] = act['departure_datetime'].dt.hour
    act['minute'] = act['departure_datetime'].dt.minute
    act['hour_float'] = act['hour'] + (act['minute'] / 60.0)
    act['headway_min'] = act['headway_s'] / 60.0

    # LOOP LINES
    for TARGET_LINE in all_lines:
        act_line = act[act['route_short_name'] == TARGET_LINE].copy()
        if act_line.empty: continue
        if len(act_line) < 10: continue

        # Identify Mode and Apply Profile
        profile_type = "Unknown"
        
        # Metro (Type 1)
        if '1' in act_line['route_type'].astype(str).values:
            profile_type = "Metro"
            act_line['Time_Group'] = act_line['hour_float'].apply(get_metro_profile)
        
        # Surface (Type 0, 3)
        elif '3' in act_line['route_type'].astype(str).values or '0' in act_line['route_type'].astype(str).values:
            profile_type = "Surface"
            act_line['Time_Group'] = act_line['hour_float'].apply(get_surface_profile)
        
        else:
            continue # Skip non-relevant modes if any

        # Aggregation: Calculate Mean Headway per Time Group
        # We group by Time_Group to determine the dominant frequency
        group_stats = act_line.groupby('Time_Group')['headway_min'].agg(['mean', 'std', 'count']).reset_index()

        # CLASSIFICATION LOGIC
        for idx, row in group_stats.iterrows():
            time_group = row['Time_Group']
            mean_headway = row['mean']
            
            # The 12-Minute Rule
            qos_type = "Regularity" if mean_headway < 12.0 else "Punctuality"
            metric_used = "EWT" if qos_type == "Regularity" else "Delay"

            # Store Result
            classification_results.append({
                'Date': sdate,
                'Line': TARGET_LINE,
                'Mode': profile_type,
                'Time_Group': time_group,
                'Scheduled_Headway_Min': round(mean_headway, 2),
                'QoS_Method': qos_type,
                'Metric_to_Use': metric_used,
                'Sample_Count': row['count']
            })

    date = date + pd.Timedelta(days=1)

# =============================================================================
#  SECTION 3: EXPORT
# =============================================================================

# Convert to DataFrame
df_results = pd.DataFrame(classification_results)

# Sort for cleaner reading
df_results = df_results.sort_values(['Date', 'Line', 'Time_Group'])

# Save to CSV
output_filename = "service_classification_master.csv"
df_results.to_csv(output_filename, index=False)

print(f"\nSUCCESS! Classification complete.")
print(f"Results saved to: {output_filename}")
print("\nSample Output:")
print(df_results.head(10).to_string(index=False))
