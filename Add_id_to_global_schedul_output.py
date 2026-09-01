# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
import utils  # Assumes your custom utils file is in the same folder
import warnings

# Sklearn Imports
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

N_TIER=7

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")
sns.set(style="white", context="talk") 

GTFS_zip_files=['GTFS_2025-08-13T04-20-01.zip', 'GTFS_2025-08-22T04-20-01.zip',
                'GTFS_2025-09-03T04-20-01.zip','GTFS_2025-09-04T04-20-01.zip','GTFS_2025-09-15T04-20-01.zip']
GTFS_dates=['2025-08-13','2025-08-22','2025-09-03','2025-09-04','2025-09-15']
data_dir="data"

sdate_debut="2025-09-02"
sdate_fin="2025-09-02"
date_fin=pd.to_datetime(sdate_fin)
start_date_obj=pd.to_datetime(sdate_debut)

GTFS_i=1
active_GTFS_zip_file=GTFS_zip_files[GTFS_i]

# =============================================================================
#  HELPER FUNCTIONS
# =============================================================================

def get_data_for_date(target_date, gtfs_file):
    """
    Helper to load and merge GTFS data for a specific date.
    Returns the 'act' dataframe containing realized trips.
    """
    sdate = str(target_date.date())
    geo_stops, stop_times, trips, lines, routes, calendar, calendar_dates = utils.read_gtfs(gtfs_file, data_dir)
    
    active_stop_times, active_trips = utils.get_active_stop_times(sdate, trips, stop_times, calendar, calendar_dates)
    
    
    # Time Conversion
    active_stop_times=active_stop_times.assign(arrival_datetime=date+pd.to_timedelta(active_stop_times['arrival_time']),
                                               departure_datetime=date+pd.to_timedelta(active_stop_times['departure_time']))

    # Merge
    act = active_trips[['route_id','trip_id','direction_id']].merge(active_stop_times, on='trip_id')
    r = routes[['route_id','route_short_name','route_type']]
    r = r.assign(route_short_name=r['route_short_name'].astype('string'))
    act = act.merge(r, on='route_id')
    act = geo_stops[['stop_id','stop_name']].merge(act, on='stop_id', how='right')

    # Calculate Headway
    stop_key = 'stop_id'
    act = act.sort_values([stop_key, 'route_id', 'direction_id', 'arrival_datetime'])
    d = act[[stop_key, 'route_id', 'direction_id', 'departure_datetime']].groupby([stop_key, 'route_id', 'direction_id']).diff()
    act['delta'] = act['departure_datetime'] - act['arrival_datetime']
    act['headway'] = d['departure_datetime'] - act['delta']
    act['headway_s'] = act['headway'].dt.seconds

    # Filter Valid
    act = act.dropna(subset=['headway_s'])
    act = act[act['headway_s'] < 5400] 
    
    return act, routes

# =============================================================================
#  PASS 1: CALIBRATION (DYNAMIC BOUNDARY CALCULATION)
# =============================================================================
print("\n=== STARTING PASS 1: CALIBRATION ===")
print("Scanning network data to learn natural Service Tiers...")

raw_headways_collection = []
date = start_date_obj

# Short loop just to gather stats
while date <= date_fin:
    act, routes = get_data_for_date(date, active_GTFS_zip_file)
    all_lines = sorted(routes['route_short_name'].dropna().astype(str).unique())
    
    for TARGET_LINE in all_lines:
        act_line = act[act['route_short_name'] == TARGET_LINE].copy()
        if act_line.empty or len(act_line) < 100: continue

        # Simple aggregation to get mean headways
        act_line['hour'] = act_line['departure_datetime'].dt.hour
        hourly = act_line.groupby('hour')['headway_s'].mean().reset_index()
        hourly = hourly[hourly['headway_s'] > 0]
        
        # Store minutes
        raw_headways_collection.extend((hourly['headway_s'] / 60.0).tolist())
    
    date = date + pd.Timedelta(days=1)

# --- COMPUTE BOUNDARIES ---
print(f"   -> Collected {len(raw_headways_collection)} hourly data points.")
print(f"   -> Running K-Means (k={N_TIER})...")

X = np.array(raw_headways_collection).reshape(-1, 1)
kmeans = KMeans(n_clusters=N_TIER, random_state=42, n_init=10).fit(X)
centroids = sorted(kmeans.cluster_centers_.flatten())
GLOBAL_BOUNDARIES = [ (centroids[i] + centroids[i+1])/2 for i in range(len(centroids)-1) ]

print("\n*** DYNAMIC BOUNDARIES LEARNED ***")
print(f"   {GLOBAL_BOUNDARIES}")
print("==================================\n")


# =============================================================================
#%%  PASS 2: EXECUTION (USING LEARNED BOUNDARIES)
# =============================================================================

def get_tier_id(minutes):
    # Uses the GLOBAL_BOUNDARIES computed in Pass 1
    return np.digitize(minutes, GLOBAL_BOUNDARIES) + 1

def get_tier_label(tier_id):
    labels = {
        1: "High Freq (Metro/Tram)",
        2: "Peak / Frequent Bus",
        3: "Standard (15 min)",
        4: "Evening (20 min)",
        5: "Sparse (30 min)",
        6: "Hourly / Rural",
        7: "Rare"
    }
    return labels.get(tier_id, "Unknown")

def extract_and_print_time_groups(hourly_stats, line_name):
    hourly_stats = hourly_stats.sort_values('hour').reset_index(drop=True)
    hourly_stats['tier_id'] = hourly_stats['mean_headway'].apply(lambda x: get_tier_id(x / 60.0))
    
    time_groups = []
    if hourly_stats.empty: return None
    
    current_start = hourly_stats.loc[0, 'hour']
    current_tier = hourly_stats.loc[0, 'tier_id']
    accumulated_headway = hourly_stats.loc[0, 'mean_headway']
    count_hours = 1
    
    for i in range(1, len(hourly_stats)):
        row = hourly_stats.loc[i]
        if row['tier_id'] == current_tier:
            accumulated_headway += row['mean_headway']
            count_hours += 1
        else:
            avg_hw_min = (accumulated_headway / count_hours) / 60.0
            prev_end = hourly_stats.loc[i-1, 'hour'] + 1
            time_groups.append({
                'Line': line_name,
                'Start_Time': current_start,
                'End_Time': prev_end,
                'Tier': current_tier,
                'Description': get_tier_label(current_tier),
                'Avg_Headway_Min': round(avg_hw_min, 1)
            })
            current_start = row['hour']
            current_tier = row['tier_id']
            accumulated_headway = row['mean_headway']
            count_hours = 1

    last_end = hourly_stats.iloc[-1]['hour'] + 1
    avg_hw_min = (accumulated_headway / count_hours) / 60.0
    time_groups.append({
        'Line': line_name,
        'Start_Time': current_start,
        'End_Time': last_end,
        'Tier': current_tier,
        'Description': get_tier_label(current_tier),
        'Avg_Headway_Min': round(avg_hw_min, 1)
    })
    
    groups_df = pd.DataFrame(time_groups)
    print(f"--- SCHEDULE BLOCKS FOR LINE {line_name} ---")
    print(groups_df[['Start_Time', 'End_Time', 'Description', 'Avg_Headway_Min']].to_string(index=False))
    return groups_df

def plot_segmentation_like_reference(raw_data, hourly_stats, line_name, final_schedule):
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.vlines(x=raw_data['hour_float'], ymin=0, ymax=raw_data['headway_min'], 
              color='grey', alpha=0.4, linewidth=1, label='Individual Headway')

    hourly_stats = hourly_stats.sort_values('hour')
    x_steps = []
    y_steps = []
    for i, row in hourly_stats.iterrows():
        val = row['mean_headway'] / 60.0
        x_steps.extend([row['hour'], row['hour'] + 1])
        y_steps.extend([val, val])

    ax.plot(x_steps, y_steps, color='#D32F2F', linewidth=3, label='Hourly Average')
    ax.set_title(f"Time Group Definition: Line {line_name}", fontsize=16, fontweight='bold', loc='left')
    ax.set_ylabel("Interval (minutes)")
    ax.set_xlabel("Time (Hour)")
    ax.set_xticks(range(0, 25))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25)])
    ax.set_xlim(5, 25)
    ax.set_ylim(0, 40) 
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def perform_segmentation(act_line, line_name, current_date_str):
    #subset = act_line[act_line['route_type'].astype(str).isin([str(r) for r in route_types])].copy()
    if act_line.empty: return None

    act_line['hour'] = act_line['departure_datetime'].dt.hour
    act_line['minute'] = act_line['departure_datetime'].dt.minute
    act_line['hour_float'] = act_line['hour'] + (act_line['minute'] / 60.0)
    act_line['headway_min'] = act_line['headway_s'] / 60.0

    hourly_stats = act_line.groupby('hour')['headway_s'].agg(
        mean_headway='mean', count='count'
    ).reset_index()
    
    hourly_stats = hourly_stats.dropna()
    hourly_stats = hourly_stats[hourly_stats['count'] > 5] 
    hourly_stats = hourly_stats[hourly_stats['mean_headway'] > 0]

    if hourly_stats.empty: return None
    
    print(f"   -> Analyzing Line {line_name}...")
    final_schedule = extract_and_print_time_groups(hourly_stats, line_name)
    plot_segmentation_like_reference(act_line, hourly_stats, line_name, final_schedule)
    
    if final_schedule is not None and not final_schedule.empty:
        final_schedule.insert(0, 'Date', current_date_str)
        return final_schedule
    return None

# --- MAIN EXECUTION LOOP ---
all_schedules_list = []
date = start_date_obj # Reset date for Pass 2

while date <= date_fin:
    sdate = str(date.date())
    print(f"--- Processing {sdate} ---")
    
    act, routes = get_data_for_date(date, active_GTFS_zip_file)
    all_lines = sorted(routes['route_short_name'].dropna().astype(str).unique())
    
    for TARGET_LINE in all_lines:
        act_line = act[act['route_short_name'] == TARGET_LINE].copy()
        if act_line.empty or len(act_line) < 100: continue 

        result_df = None
        #if '1' in act_line['route_type'].astype(str).values:
        result_df = perform_segmentation(act_line, line_name=TARGET_LINE, current_date_str=sdate)
        #elif '3' in act_line['route_type'].astype(str).values or '0' in act_line['route_type'].astype(str).values:
        #    result_df = perform_segmentation(act_line, ['0','3'], line_name=TARGET_LINE, current_date_str=sdate)
            
        if result_df is not None:
            all_schedules_list.append(result_df)

    date = date + pd.Timedelta(days=1)

if all_schedules_list:
    master_df = pd.concat(all_schedules_list, ignore_index=True)
    filename = "GLOBAL_SCHEDULE_OUTPUT.csv"
    master_df.to_csv(filename, index=False, sep=',')
    print(f"\nSUCCESS: All lines processed. Saved to {filename}")
else:
    print("\nWARNING: No valid schedules were generated.")