# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
import utils
import warnings

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

# =============================================================================
#  SECTION 1: THE MANUAL PROFILING LOGIC (Hard-coded Rules)
# =============================================================================

def get_surface_profile(hour_float):
    """
    Profile A: Bus & Tram
    Based on specific cuts: 07:00, 09:30, 15:30, 18:30, 21:00
    """
    if hour_float < 7.0:
        return "1. Morning Ramp-up"
    elif 7.0 <= hour_float < 9.5: # 9:30 is 9.5
        return "2. AM Rush"
    elif 9.5 <= hour_float < 15.5: # 15:30 is 15.5
        return "3. Day Base"
    elif 15.5 <= hour_float < 18.5: # 18:30 is 18.5
        return "4. School/PM Rush"
    elif 18.5 <= hour_float < 21.0:
        return "5. Evening Slide"
    else:
        return "6. Night"

def get_metro_profile(hour_float):
    """
    Profile B: Metro
    Based on specific cuts: 06:30, 09:30, 16:00, 19:00, 22:00
    """
    if hour_float < 6.5: # 6:30 is 6.5
        return "1. Early Service"
    elif 6.5 <= hour_float < 9.5:
        return "2. AM Peak"
    elif 9.5 <= hour_float < 16.0:
        return "3. Day Base"
    elif 16.0 <= hour_float < 19.0:
        return "4. PM Peak"
    elif 19.0 <= hour_float < 22.0:
        return "5. Evening"
    else:
        return "6. Late Night"

def get_profile_coordinates(mode_name):
    """Returns the X-axis start/stop coords for plotting the red lines"""
    if mode_name == "Surface":
        return {
            "1. Morning Ramp-up": (5.0, 7.0),
            "2. AM Rush": (7.0, 9.5),
            "3. Day Base": (9.5, 15.5),
            "4. School/PM Rush": (15.5, 18.5),
            "5. Evening Slide": (18.5, 21.0),
            "6. Night": (21.0, 25.0)
        }
    else: # Metro
        return {
            "1. Early Service": (5.0, 6.5),
            "2. AM Peak": (6.5, 9.5),
            "3. Day Base": (9.5, 16.0),
            "4. PM Peak": (16.0, 19.0),
            "5. Evening": (19.0, 22.0),
            "6. Late Night": (22.0, 25.0)
        }

# =============================================================================
#  SECTION 2: CLASSIFICATION & VISUALIZATION
# =============================================================================

def get_los_color(cv):
    """Returns color based on Coefficient of Variation (Reliability)"""
    if cv <= 0.25:
        return '#2E7D32' # Strong Green (Excellent)
    elif cv <= 0.75:
        return '#F57F17' # Orange (Acceptable/Variable)
    else:
        return '#C62828' # Red (Unstable)

def get_los_grade(cv):
    if cv <= 0.25: return "A (Excellent)"
    if cv <= 0.75: return "B (Normal)"
    return "C (Unstable)"

def classify_and_plot(raw_data, line_name, mode_type):
    """
    1. Aggregates data by Time Group.
    2. Calculates Reliability (Cv).
    3. Plots with Color-Coded Performance lines.
    """
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # 1. Plot Raw Data (Grey Bars)
    ax.vlines(
        x=raw_data['hour_float'], 
        ymin=0, 
        ymax=raw_data['headway_min'], 
        color='grey', 
        alpha=0.2, 
        linewidth=1
    )

    # 2. Calculate Stats per Manual Group
    # We calculate Mean (Frequency) and Std (Regularity)
    group_stats = raw_data.groupby('Time_Group')['headway_min'].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate Coefficient of Variation (Cv)
    group_stats['Cv'] = group_stats['std'] / group_stats['mean']
    
    # 3. Draw the Color-Coded Profile Lines
    coords = get_profile_coordinates(mode_type)
    
    print(f"\n--- PERFORMANCE REPORT: LINE {line_name} ---")
    print(f"{'Time Group':<20} | {'Mean (m)':<10} | {'Cv':<10} | {'Grade':<15}")
    print("-" * 65)

    for index, row in group_stats.iterrows():
        group_name = row['Time_Group']
        mean_val = row['mean']
        cv = row['Cv']
        
        # Print stats to console
        print(f"{group_name:<20} | {mean_val:.2f}       | {cv:.2f}       | {get_los_grade(cv)}")

        if group_name in coords:
            start_x, end_x = coords[group_name]
            
            # Determine Color based on Reliability
            line_color = get_los_color(cv)
            
            # Draw Horizontal Line (The Mean)
            ax.hlines(y=mean_val, xmin=start_x, xmax=end_x, 
                      color=line_color, linewidth=5, zorder=5)
            
            # Add Text Label
            label_text = f"{group_name}\n{mean_val:.1f}m ({get_los_grade(cv)[0]})"
            ax.text(x=start_x + 0.2, y=mean_val + 1.5, s=label_text, 
                    color=line_color, fontweight='bold', fontsize=11, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # 4. Formatting
    ax.set_title(f"Service Performance Classification: Line {line_name}", fontsize=18, fontweight='bold', loc='left')
    ax.set_ylabel("Headway (minutes)")
    ax.set_xlabel("Time of Day")
    
    # Legend for Grades
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#2E7D32', lw=4, label='Grade A: Reliable (Cv < 0.25)'),
        Line2D([0], [0], color='#F57F17', lw=4, label='Grade B: Variable (0.25 < Cv < 0.75)'),
        Line2D([0], [0], color='#C62828', lw=4, label='Grade C: Unstable (Cv > 0.75)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_xticks(range(5, 26))
    ax.set_xticklabels([f"{h:02d}:00" if h < 24 else "01:00" for h in range(5, 26)])
    ax.set_xlim(5, 25)
    
    y_max_limit = raw_data['headway_min'].quantile(0.99) * 1.2
    ax.set_ylim(0, 30 if y_max_limit < 30 else y_max_limit)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
#  SECTION 3: MAIN LOOP
# =============================================================================

geo_stops, stop_times, trips, lines, routes, calendar, calendar_dates = utils.read_gtfs(active_GTFS_zip_file,data_dir)
all_lines = sorted(routes['route_short_name'].dropna().astype(str).unique())

while date <= date_fin:
    sdate = str(date.date())
    print(f"--- Processing {sdate} ---")
    
    if sdate > GTFS_dates[GTFS_i+1]:
        GTFS_i += 1
        active_GTFS_zip_file = GTFS_zip_files[GTFS_i]
        geo_stops, stop_times, trips, lines, routes, calendar, calendar_dates = utils.read_gtfs(active_GTFS_zip_file,data_dir)
    
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

    # Calculate Headway (Your Logic)
    stop_key = 'stop_name'
    act = act.sort_values([stop_key, 'route_id', 'direction_id', 'arrival_datetime'])
    d = act[[stop_key, 'route_id', 'direction_id', 'departure_datetime']].groupby([stop_key, 'route_id', 'direction_id']).diff()
    act['delta'] = act['departure_datetime'] - act['arrival_datetime']
    act['headway'] = d['departure_datetime'] - act['delta']
    act['headway_s'] = act['headway'].dt.seconds

    # Filter Valid & Convert to Minutes/Float Hour
    act = act.dropna(subset=['headway_s'])
    act = act[act['headway_s'] < 5400] 
    
    # Calculate Float Hour for precise cutting (e.g., 9:30 = 9.5)
    act['hour'] = act['departure_datetime'].dt.hour
    act['minute'] = act['departure_datetime'].dt.minute
    act['hour_float'] = act['hour'] + (act['minute'] / 60.0)
    act['headway_min'] = act['headway_s'] / 60.0

    # LOOP LINES
    for TARGET_LINE in all_lines:
        act_line = act[act['route_short_name'] == TARGET_LINE].copy()
        if act_line.empty: continue
        if len(act_line) < 50: continue

        # --- APPLY PROFILE A (METRO) ---
        if '1' in act_line['route_type'].astype(str).values:
            print(f"Applying Profile B (Metro) to Line {TARGET_LINE}")
            # Apply the cut
            act_line['Time_Group'] = act_line['hour_float'].apply(get_metro_profile)
            # Visualize
            classify_and_plot(act_line, TARGET_LINE, "Metro")
        
        # --- APPLY PROFILE B (SURFACE) ---
        if '3' in act_line['route_type'].astype(str).values or '0' in act_line['route_type'].astype(str).values:
            print(f"Applying Profile A (Surface) to Line {TARGET_LINE}")
            # Apply the cut
            act_line['Time_Group'] = act_line['hour_float'].apply(get_surface_profile)
            # Visualize
            classify_and_plot(act_line, TARGET_LINE, "Surface")

    date = date + pd.Timedelta(days=1)