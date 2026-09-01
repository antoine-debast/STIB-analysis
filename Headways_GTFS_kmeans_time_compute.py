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
sns.set(style="white", context="talk") # Clean white background like the picture

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

# =============================================================================
#  NEW VISUALIZATION FUNCTION (MATCHING YOUR PICTURE)
# =============================================================================
def plot_segmentation_like_reference(raw_data, hourly_stats, line_name):
    """
    Replicates the 'Red Step Line' chart.
    - Raw Data: Vertical Grey lines (Interval between 2 vehicles)
    - Time Groups: Red Step Line (Average of the group)
    """
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # 1. Plot Raw Data (The Grey Bars)
    # We use vlines to look exactly like the reference picture
    ax.vlines(
        x=raw_data['hour_float'], 
        ymin=0, 
        ymax=raw_data['headway_min'], 
        color='grey', 
        alpha=0.4, 
        linewidth=1,
        label='Individual Headway'
    )

    # 2. Prepare the Red Step Line (The Time Groups)
    # We group by the computed 'Regime' to find the average headway for that block
    hourly_stats = hourly_stats.sort_values('hour')
    
    # Create the step data
    x_steps = []
    y_steps = []
    
    # Iterate to build the step line
    for i, row in hourly_stats.iterrows():
        # Get the mean headway of this specific cluster/regime to represent the "Schedule"
        # (We use the cluster centroid approximation or the actual hourly mean)
        regime_mean = row['mean_headway'] / 60.0 # Convert to minutes
        
        # Start of hour
        x_steps.append(row['hour'])
        y_steps.append(regime_mean)
        # End of hour
        x_steps.append(row['hour'] + 1)
        y_steps.append(regime_mean)

    # 3. Plot the Red Step Line
    ax.plot(x_steps, y_steps, color='#D32F2F', linewidth=3, label='Defined Time Group')
    
    # 4. Fill the area under the red line slightly (optional, adds readability)
    # ax.fill_between(x_steps, y_steps, color='#D32F2F', alpha=0.05)

    # Formatting
    ax.set_title(f"Time Group Definition: Line {line_name} (Based on Realized Headways)", fontsize=16, fontweight='bold', loc='left')
    ax.set_ylabel("Interval between two vehicles (minutes)")
    ax.set_xlabel("Time (Hour of Day)")
    
    # Set X-axis to show hours clearly (05:00, 06:00...)
    ax.set_xticks(range(0, 25))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25)])
    ax.set_xlim(5, 24) # Assuming service starts ~5AM
    
    # Set Y-axis limit (zoom in to relevant headways, e.g., 0 to 30 mins)
    ax.set_ylim(0, 30) 
    
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
#  CLUSTERING ENGINE
# =============================================================================

def perform_segmentation(df, route_types, n_clusters, line_name):
    # Filter
    subset = df[df['route_type'].astype(str).isin([str(r) for r in route_types])].copy()
    if subset.empty: return

    print(f"   -> Analyzing Line {line_name}...")

    # Feature Engineering
    subset['hour'] = subset['departure_datetime'].dt.hour
    
    # Create 'hour_float' for precise plotting (e.g. 10:30 = 10.5)
    subset['minute'] = subset['departure_datetime'].dt.minute
    subset['hour_float'] = subset['hour'] + (subset['minute'] / 60.0)
    subset['headway_min'] = subset['headway_s'] / 60.0

    # Aggregation
    hourly_stats = subset.groupby('hour')['headway_s'].agg(
        mean_headway='mean', std_headway='std', count='count'
    ).reset_index()
    
    # Data Hardening
    hourly_stats = hourly_stats.dropna()
    hourly_stats = hourly_stats[hourly_stats['count'] > 5] # Allow slightly smaller groups
    hourly_stats = hourly_stats[hourly_stats['mean_headway'] > 0]

    if hourly_stats.empty: return

    # --- GMM CLUSTERING ---
    # We weight 'log_mean' higher because your picture defines groups by FREQUENCY (Height of the bar)
    hourly_stats['log_mean'] = np.log(hourly_stats['mean_headway'])
    hourly_stats['log_std'] = np.log(hourly_stats['std_headway'] + 1) # +1 to avoid log(0)
    
    X = hourly_stats[['log_mean', 'log_std']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit GMM
    try:
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42, n_init=10)
        hourly_stats['cluster'] = gmm.fit_predict(X_scaled)
    except:
        return

    # Sort Clusters by Frequency (Mean Headway)
    # Cluster 0 = Shortest Headway (Highest Frequency)
    cluster_means = hourly_stats.groupby('cluster')['mean_headway'].mean().sort_values()
    remap = {old: new for new, old in enumerate(cluster_means.index)}
    hourly_stats['cluster_sorted'] = hourly_stats['cluster'].map(remap)
    
    # --- VISUALIZE ---
    plot_segmentation_like_reference(subset, hourly_stats, line_name)

# =============================================================================
#  MAIN LOOP
# =============================================================================

geo_stops, stop_times, trips, lines, routes, calendar, calendar_dates = utils.read_gtfs(active_GTFS_zip_file,data_dir)
all_lines = sorted(routes['route_short_name'].dropna().astype(str).unique())

while date <= date_fin:
    sdate = str(date.date())
    print(f"--- Processing {sdate} ---")
    
    # (GTFS Loading Logic omitted for brevity, assuming loaded from previous steps)
    if sdate > GTFS_dates[GTFS_i+1]:
        GTFS_i += 1
        active_GTFS_zip_file = GTFS_zip_files[GTFS_i]
        geo_stops, stop_times, trips, lines, routes, calendar, calendar_dates = utils.read_gtfs(active_GTFS_zip_file,data_dir)
    
    active_stop_times, active_trips = utils.get_active_stop_times(sdate, trips, stop_times, calendar, calendar_dates)
    
    # Convert Time
    active_stop_times = active_stop_times.assign(
        arrival_datetime=active_stop_times['arrival_time'].str.split(':').apply(
            lambda x: date + datetime.timedelta(hours=int(x[0]), minutes=int(x[1]), seconds=int(x[2]))),
        departure_datetime=active_stop_times['departure_time'].str.split(':').apply(
            lambda x: date + datetime.timedelta(hours=int(x[0]), minutes=int(x[1]), seconds=int(x[2])))
    )

    # Merge
    act = active_trips[['route_id','trip_id','direction_id']].merge(active_stop_times, on='trip_id')
    r = routes[['route_id','route_short_name','route_type']]
    r = r.assign(route_short_name=r['route_short_name'].astype('string'))
    act = act.merge(r, on='route_id')
    act = geo_stops[['stop_id','stop_name']].merge(act, on='stop_id', how='right')

    # Calculate Headway (Your Logic)
    stop_key = 'stop_name'
    act = act.sort_values([stop_key, 'route_id', 'direction_id', 'arrival_datetime'])
    d = act[[stop_key, 'route_id', 'direction_id', 'departure_datetime']].groupby([stop_key, 'route_id', 'direction_id']).diff()
    act['delta'] = act['departure_datetime'] - act['arrival_datetime']
    act['headway'] = d['departure_datetime'] - act['delta']
    act['headway_s'] = act['headway'].dt.seconds

    # Filter Valid
    act = act.dropna(subset=['headway_s'])
    act = act[act['headway_s'] < 5400] # Remove gaps > 1.5 hours

    # LOOP LINES
    for TARGET_LINE in all_lines:
        act_line = act[act['route_short_name'] == TARGET_LINE].copy()
        if act_line.empty: continue
        if len(act_line) < 100: continue # Skip lines with too little data

        # Process Metro (Type 1)
        if '1' in act_line['route_type'].astype(str).values:
             # Use 4 clusters to force broader groups (Peak, Off-Peak, Night, Transition)
            perform_segmentation(act_line, ['1'], n_clusters=4, line_name=TARGET_LINE)
        
        # Process Surface (Type 0, 3)
        if '3' in act_line['route_type'].astype(str).values or '0' in act_line['route_type'].astype(str).values:
            perform_segmentation(act_line, ['0','3'], n_clusters=4, line_name=TARGET_LINE)

    date = date + pd.Timedelta(days=1)