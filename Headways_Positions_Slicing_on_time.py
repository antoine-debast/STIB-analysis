# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
import warnings

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")
sns.set(style="white", context="talk")

# FIX: This variable should be a string path, not the loaded dataframe
INPUT_PKL_FILE = 'data/fusion result/Fused_GTFS_VehiclePositions(2025-09-03).pkl'

# =============================================================================
#  SECTION 1: THE MANUAL PROFILING LOGIC
# =============================================================================

def get_surface_profile(hour_float):
    if hour_float < 7.0:
        return "1. Morning Ramp-up"
    elif 7.0 <= hour_float < 9.5:
        return "2. AM Rush"
    elif 9.5 <= hour_float < 15.5:
        return "3. Day Base"
    elif 15.5 <= hour_float < 18.5:
        return "4. School/PM Rush"
    elif 18.5 <= hour_float < 21.0:
        return "5. Evening Slide"
    else:
        return "6. Night"

def get_metro_profile(hour_float):
    if hour_float < 6.5:
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
    if cv <= 0.25:
        return '#2E7D32' # Strong Green
    elif cv <= 0.75:
        return '#F57F17' # Orange
    else:
        return '#C62828' # Red

def get_los_grade(cv):
    if cv <= 0.25: return "A (Excellent)"
    if cv <= 0.75: return "B (Normal)"
    return "C (Unstable)"

def classify_and_plot(raw_data, line_name, mode_type):
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # 1. Plot Raw Data
    ax.vlines(
        x=raw_data['hour_float'], 
        ymin=0, 
        ymax=raw_data['headway_min'], 
        color='grey', 
        alpha=0.2, 
        linewidth=1
    )

    # 2. Calculate Stats
    group_stats = raw_data.groupby('Time_Group')['headway_min'].agg(['mean', 'std', 'count']).reset_index()
    group_stats['Cv'] = group_stats['std'] / group_stats['mean']
    
    # 3. Draw Lines
    coords = get_profile_coordinates(mode_type)
    
    print(f"\n--- PERFORMANCE REPORT: LINE {line_name} ---")
    print(f"{'Time Group':<20} | {'Mean (m)':<10} | {'Cv':<10} | {'Grade':<15}")
    print("-" * 65)

    for index, row in group_stats.iterrows():
        group_name = row['Time_Group']
        mean_val = row['mean']
        cv = row['Cv']
        
        print(f"{group_name:<20} | {mean_val:.2f}       | {cv:.2f}       | {get_los_grade(cv)}")

        if group_name in coords:
            start_x, end_x = coords[group_name]
            line_color = get_los_color(cv)
            
            ax.hlines(y=mean_val, xmin=start_x, xmax=end_x, 
                      color=line_color, linewidth=5, zorder=5)
            
            label_text = f"{group_name}\n{mean_val:.1f}m ({get_los_grade(cv)[0]})"
            ax.text(x=start_x + 0.2, y=mean_val + 1.5, s=label_text, 
                    color=line_color, fontweight='bold', fontsize=11, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # 4. Formatting
    ax.set_title(f"Real Service Performance: Line {line_name}", fontsize=18, fontweight='bold', loc='left')
    ax.set_ylabel("Headway (minutes)")
    ax.set_xlabel("Time of Day")
    
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
    ax.set_ylim(0, 30 if pd.isna(y_max_limit) or y_max_limit < 30 else y_max_limit)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
#  SECTION 3: REAL DATA PROCESSING
# =============================================================================

print("Loading .pkl file...")

# FIX: Added a check. If INPUT_PKL_FILE is a path string, load it. 
# If it's already a DataFrame (from a previous run), just use it.
if isinstance(INPUT_PKL_FILE, str):
    df = pd.read_pickle(INPUT_PKL_FILE)
else:
    df = INPUT_PKL_FILE.copy()

# Ensure timestamp is datetime (based on your snippet, it is timestamp_dt)
if 'timestamp_dt' in df.columns:
    df['clean_time'] = pd.to_datetime(df['timestamp_dt'])
else:
    # Fallback if only string exists
    df['clean_time'] = pd.to_datetime(df['timestamp'])

# Filter out bad data if necessary (optional)
df = df.dropna(subset=['clean_time', 'route_short_name'])

# --- HEADWAY CALCULATION (REAL DATA) ---
# We sort by Stop, Route, Direction, and Time to calculate the gap between vehicles
sort_cols = ['stop_name', 'route_short_name', 'direction_id', 'clean_time']
df = df.sort_values(sort_cols)

# Calculate difference between current vehicle time and previous vehicle time at the same stop
df['prev_time'] = df.groupby(['stop_name', 'route_short_name', 'direction_id'])['clean_time'].shift(1)
df['headway_s'] = (df['clean_time'] - df['prev_time']).dt.total_seconds()

# Filter invalid headways
# 1. Drop NaNs (first bus of the day has no headway)
# 2. Drop extremely large headways (e.g., > 1.5 hours, likely night gaps or missing data)
# 3. Drop extremely short headways (e.g., < 20 seconds, likely duplicates/phantom detections)
df = df.dropna(subset=['headway_s'])
df = df[(df['headway_s'] < 5400) & (df['headway_s'] > 20)]

# Calculate units for plotting
df['headway_min'] = df['headway_s'] / 60.0
df['hour'] = df['clean_time'].dt.hour
df['minute'] = df['clean_time'].dt.minute
df['hour_float'] = df['hour'] + (df['minute'] / 60.0)

# Adjust for post-midnight service (24:00 - 26:00) logic if your data spans into night
# Simple logic: if hour < 4, add 24.
df.loc[df['hour'] < 4, 'hour_float'] += 24

# Loop through lines
all_lines = sorted(df['route_short_name'].astype(str).unique())

for TARGET_LINE in all_lines:
    act_line = df[df['route_short_name'] == TARGET_LINE].copy()
    
    if act_line.empty: continue
    if len(act_line) < 50: continue

    # Ensure route_type is string for comparison
    r_types = act_line['route_type'].astype(str).values
    
    # --- APPLY PROFILE B (METRO) ---
    # Route Type 1 is usually Metro
    if '1' in r_types:
        print(f"Applying Profile B (Metro) to Line {TARGET_LINE}")
        act_line['Time_Group'] = act_line['hour_float'].apply(get_metro_profile)
        classify_and_plot(act_line, TARGET_LINE, "Metro")
    
    # --- APPLY PROFILE A (SURFACE) ---
    # Route Type 3 is Bus, 0 is Tram
    elif '3' in r_types or '0' in r_types:
        print(f"Applying Profile A (Surface) to Line {TARGET_LINE}")
        act_line['Time_Group'] = act_line['hour_float'].apply(get_surface_profile)
        classify_and_plot(act_line, TARGET_LINE, "Surface")