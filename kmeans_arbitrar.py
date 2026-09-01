# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
import utils
import warnings

# Sklearn Imports
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

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
date=pd.to_datetime(sdate_debut)

GTFS_i=1
active_GTFS_zip_file=GTFS_zip_files[GTFS_i]

# Container for final results
all_schedules_list = [] 

# =============================================================================
#  ALGORITHM: TIER DEFINITIONS (Derived from K-Means k=6)
# =============================================================================

def get_tier_id(minutes):
    """
    Quantizes a continuous headway (minutes) into a discrete Tier ID (1-6).
    """
    if minutes <= 8.0:  return 1  # High Freq (Metro/Tram)
    if minutes <= 12.0: return 2  # Frequent / Peak Bus
    if minutes <= 17.0: return 3  # Standard (15 min)
    if minutes <= 23.0: return 4  # Evening (20 min)
    if minutes <= 40.0: return 5  # Sparse (30 min)
    return 6                      # Hourly / Rural

def get_tier_label(tier_id):
    """Returns human-readable label for the Tier."""
    labels = {
        1: "High Freq (Metro/Tram)",
        2: "Peak / Frequent Bus",
        3: "Standard (15 min)",
        4: "Evening (20 min)",
        5: "Sparse (30 min)",
        6: "Hourly / Rural"
    }
    return labels.get(tier_id, "Unknown")

def extract_and_print_time_groups(hourly_stats, line_name):
    """
    ALGORITHM STEP 2: GROUPING
    Collapses hourly stats into broader Time Groups based on the Tier ID.
    Merges adjacent hours if they belong to the same Tier.
    """
    # 1. Prepare Data: Sort by time and assign Tier IDs
    df = hourly_stats.sort_values('hour').reset_index(drop=True)
    df['tier_id'] = df['mean_headway'].apply(lambda x: get_tier_id(x / 60.0))
    
    time_groups = []
    if df.empty: return None
    
    # 2. Initialize the first block
    current_start = df.loc[0, 'hour']
    current_tier = df.loc[0, 'tier_id']
    
    # Variables to calculate weighted average for the resulting block
    accumulated_headway = df.loc[0, 'mean_headway']
    count_hours = 1
    
    # 3. Iterate through hours to group them
    for i in range(1, len(df)):
        row = df.loc[i]
        
        # LOGIC: If the Tier is the same, extend the current time block
        if row['tier_id'] == current_tier:
            accumulated_headway += row['mean_headway']
            count_hours += 1
            
        # LOGIC: If Tier changes, save the block and start a new one
        else:
            # Save previous group
            avg_hw_min = (accumulated_headway / count_hours) / 60.0
            prev_end = df.loc[i-1, 'hour'] + 1 # End of the hour block
            
            time_groups.append({
                'Line': line_name,
                'Start_Time': f"{int(current_start):02d}:00",
                'End_Time': f"{int(prev_end):02d}:00",
                'Tier': current_tier,
                'Description': get_tier_label(current_tier),
                'Avg_Headway_Min': round(avg_hw_min, 1)
            })
            
            # Start new group
            current_start = row['hour']
            current_tier = row['tier_id']
            accumulated_headway = row['mean_headway']
            count_hours = 1

    # 4. Save the final pending group
    last_end = df.iloc[-1]['hour'] + 1
    avg_hw_min = (accumulated_headway / count_hours) / 60.0
    time_groups.append({
        'Line': line_name,
        'Start_Time': f"{int(current_start):02d}:00",
        'End_Time': f"{int(last_end):02d}:00",
        'Tier': current_tier,
        'Description': get_tier_label(current_tier),
        'Avg_Headway_Min': round(avg_hw_min, 1)
    })
    
    # 5. Output to Console
    groups_df = pd.DataFrame(time_groups)
    print(f"\n--- SCHEDULE BLOCKS FOR LINE {line_name} ---")
    print(groups_df[['Start_Time', 'End_Time', 'Description', 'Avg_Headway_Min']].to_string(index=False))
    
    return groups_df

# =============================================================================
#  VISUALIZATION FUNCTION
# =============================================================================
def plot_segmentation_like_reference(raw_data, hourly_stats, line_name):
    """
    Replicates the 'Red Step Line' chart based on the realized headways.
    """
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # 1. Plot Raw Data (Vertical Grey lines)
    ax.vlines(
        x=raw_data['hour_float'], 
        ymin=0, 
        ymax=raw_data['headway_min'], 
        color='grey', 
        alpha=0.4, 
        linewidth=1,
        label='Individual Headway'
    )

    # 2. Prepare the Red Step Line (Average per hour)
    hourly_stats = hourly_stats.sort_values('hour')
    x_steps = []
    y_steps = []
    
    for i, row in hourly_stats.iterrows():
        val = row['mean_headway'] / 60.0
        x_steps.extend([row['hour'], row['hour'] + 1])
        y_steps.extend([val, val])

    # 3. Plot the Red Step Line
    ax.plot(x_steps, y_steps, color='#D32F2F', linewidth=3, label='Hourly Average')
    
    ax.set_title(f"Time Group Definition: Line {line_name}", fontsize=16, fontweight='bold', loc='left')
    ax.set_ylabel("Interval (minutes)")
    ax.set_xlabel("Time (Hour)")
    
    ax.set_xticks(range(0, 25))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25)])
    ax.set_xlim(5, 25)
    ax.set_ylim(0, 40) 
    
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
#  CORE ANALYSIS FUNCTION
# =============================================================================

def perform_segmentation(df, route_types, line_name, current_date_str):
    # Filter by route type
    subset = df[df['route_type'].astype(str).isin([str(r) for r in route_types])].copy()
    if subset.empty: return None

    print(f"   -> Analyzing Line {line_name}...")

    # Feature Engineering
    subset['hour'] = subset['departure_datetime'].dt.hour
    subset['minute'] = subset['departure_datetime'].dt.minute
    subset['hour_float'] = subset['hour'] + (subset['minute'] / 60.0)
    subset['headway_min'] = subset['headway_s'] / 60.0

    # Aggregation (Stats per Hour)
    hourly_stats = subset.groupby('hour')['headway_s'].agg(
        mean_headway='mean', std_headway='std', count='count'
    ).reset_index()
    
    # Data Cleaning
    hourly_stats = hourly_stats.dropna()
    hourly_stats = hourly_stats[hourly_stats['count'] > 5] 
    hourly_stats = hourly_stats[hourly_stats['mean_headway'] > 0]

    if hourly_stats.empty: return None
    
    # --- EXECUTE LOGIC ---
    plot_segmentation_like_reference(subset, hourly_stats, line_name)
    final_schedule = extract_and_print_time_groups(hourly_stats, line_name)
    
    # Add Date column
    if final_schedule is not None and not final_schedule.empty:
        final_schedule.insert(0, 'Date', current_date_str)
        return final_schedule
    
    return None

# =============================================================================
#  MAIN LOOP
# =============================================================================

geo_stops, stop_times, trips, lines, routes, calendar, calendar_dates = utils.read_gtfs(active_GTFS_zip_file, data_dir)
all_lines = sorted(routes['route_short_name'].dropna().astype(str).unique())

while date <= date_fin:
    sdate = str(date.date())
    print(f"--- Processing {sdate} ---")
    
    # Handle GTFS file switching if dates cross boundaries
    if sdate > GTFS_dates[GTFS_i+1]:
        GTFS_i += 1
        active_GTFS_zip_file = GTFS_zip_files[GTFS_i]
        geo_stops, stop_times, trips, lines, routes, calendar, calendar_dates = utils.read_gtfs(active_GTFS_zip_file, data_dir)
    
    active_stop_times, active_trips = utils.get_active_stop_times(sdate, trips, stop_times, calendar, calendar_dates)
    
    # Convert Time strings to datetime objects
    active_stop_times = active_stop_times.assign(
        arrival_datetime=active_stop_times['arrival_time'].str.split(':').apply(
            lambda x: date + datetime.timedelta(hours=int(x[0]), minutes=int(x[1]), seconds=int(x[2]))),
        departure_datetime=active_stop_times['departure_time'].str.split(':').apply(
            lambda x: date + datetime.timedelta(hours=int(x[0]), minutes=int(x[1]), seconds=int(x[2])))
    )

    # Merge Data
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

    # Filter Valid Headways
    act = act.dropna(subset=['headway_s'])
    act = act[act['headway_s'] < 5400] 

    # Process Each Line
    for TARGET_LINE in all_lines:
        act_line = act[act['route_short_name'] == TARGET_LINE].copy()
        if act_line.empty: continue
        if len(act_line) < 100: continue 

        result_df = None
        
        # Route Types (Metro vs Surface)
        if '1' in act_line['route_type'].astype(str).values:
            result_df = perform_segmentation(act_line, ['1'], line_name=TARGET_LINE, current_date_str=sdate)
        elif '3' in act_line['route_type'].astype(str).values or '0' in act_line['route_type'].astype(str).values:
            result_df = perform_segmentation(act_line, ['0','3'], line_name=TARGET_LINE, current_date_str=sdate)
            
        if result_df is not None:
            all_schedules_list.append(result_df)

    date = date + pd.Timedelta(days=1)

# =============================================================================
#  SAVE FINAL OUTPUT
# =============================================================================
if all_schedules_list:
    master_df = pd.concat(all_schedules_list, ignore_index=True)
    filename = "GLOBAL_SCHEDULE_OUTPUT.csv"
    master_df.to_csv(filename, index=False)
    print(f"\nSUCCESS: All lines processed. Saved to {filename}")
else:
    print("\nWARNING: No valid schedules were generated.")