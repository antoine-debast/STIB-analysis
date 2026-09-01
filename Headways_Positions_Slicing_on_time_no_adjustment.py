# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")
sns.set(style="white", context="talk")

# Input File
INPUT_PKL_FILE = 'data/fusion result/Fused_GTFS_VehiclePositions(2025-09-03).pkl'

# =============================================================================
#  SECTION 1: PROFILING LOGIC
# =============================================================================

def get_surface_profile(hour_float):
    if hour_float < 7.0: return "1. Morning Ramp-up"
    elif 7.0 <= hour_float < 9.5: return "2. AM Rush"
    elif 9.5 <= hour_float < 15.5: return "3. Day Base"
    elif 15.5 <= hour_float < 18.5: return "4. School/PM Rush"
    elif 18.5 <= hour_float < 21.0: return "5. Evening Slide"
    else: return "6. Night"

def get_metro_profile(hour_float):
    if hour_float < 6.5: return "1. Early Service"
    elif 6.5 <= hour_float < 9.5: return "2. AM Peak"
    elif 9.5 <= hour_float < 16.0: return "3. Day Base"
    elif 16.0 <= hour_float < 19.0: return "4. PM Peak"
    elif 19.0 <= hour_float < 22.0: return "5. Evening"
    else: return "6. Late Night"

def get_profile_coordinates(mode_name):
    if mode_name == "Surface":
        return {
            "1. Morning Ramp-up": (5.0, 7.0), "2. AM Rush": (7.0, 9.5),
            "3. Day Base": (9.5, 15.5), "4. School/PM Rush": (15.5, 18.5),
            "5. Evening Slide": (18.5, 21.0), "6. Night": (21.0, 25.0)
        }
    else: # Metro
        return {
            "1. Early Service": (5.0, 6.5), "2. AM Peak": (6.5, 9.5),
            "3. Day Base": (9.5, 16.0), "4. PM Peak": (16.0, 19.0),
            "5. Evening": (19.0, 22.0), "6. Late Night": (22.0, 25.0)
        }

# =============================================================================
#  SECTION 2: CLASSIFICATION & VISUALIZATION
# =============================================================================

def get_los_color(cv):
    if cv <= 0.25: return '#2E7D32' # Green
    elif cv <= 0.75: return '#F57F17' # Orange
    else: return '#C62828' # Red

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
        alpha=0.3, # Slightly higher alpha to see density better
        linewidth=1
    )

    # 2. Stats
    group_stats = raw_data.groupby('Time_Group')['headway_min'].agg(['mean', 'std', 'count']).reset_index()
    group_stats['Cv'] = group_stats['std'] / group_stats['mean']
    
    # 3. Draw Lines
    coords = get_profile_coordinates(mode_type)
    
    print(f"\n--- PERFORMANCE REPORT: LINE {line_name} ({mode_type}) ---")
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
            ax.text(x=start_x + 0.2, y=mean_val + (mean_val*0.1), s=label_text, 
                    color=line_color, fontweight='bold', fontsize=11, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # 4. Formatting
    ax.set_title(f"Real Service Performance: Line {line_name}", fontsize=18, fontweight='bold', loc='left')
    ax.set_ylabel("Headway (minutes)")
    ax.set_xlabel("Time of Day")
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#2E7D32', lw=4, label='Grade A: Reliable (Cv < 0.25)'),
        Line2D([0], [0], color='#F57F17', lw=4, label='Grade B: Variable (0.25 < Cv < 0.75)'),
        Line2D([0], [0], color='#C62828', lw=4, label='Grade C: Unstable (Cv > 0.75)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # X-Axis
    ax.set_xticks(range(5, 26))
    ax.set_xticklabels([f"{h:02d}:00" if h < 24 else f"{h-24:02d}:00" for h in range(5, 26)])
    ax.set_xlim(5, 25)
    
    # Y-Axis (Dynamic: Fits the data, no hard cap)
    data_max = raw_data['headway_min'].max()
    # If data is normal (<30), show up to 30. If huge outliers exist, show full range + 10% buffer
    y_limit = max(30, data_max * 1.1)
    ax.set_ylim(0, y_limit)
    
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# =============================================================================
#  SECTION 3: DATA PROCESSING (NO FILTERS)
# =============================================================================

print("Loading data...")

if isinstance(INPUT_PKL_FILE, str):
    df = pd.read_pickle(INPUT_PKL_FILE)
else:
    df = INPUT_PKL_FILE.copy()

# 1. Clean Datetimes
if 'timestamp_dt' in df.columns:
    df['clean_time'] = pd.to_datetime(df['timestamp_dt'], errors='coerce')
elif 'timestamp' in df.columns:
    df['clean_time'] = pd.to_datetime(df['timestamp'], errors='coerce')

# 2. Minimal Safety Clean
# We only drop rows where we absolutely cannot identify the Line or the Time
df = df.dropna(subset=['clean_time', 'route_short_name'])

# 3. Calculate Headways
df = df.sort_values(['stop_name', 'route_short_name', 'direction_id', 'clean_time'])

grp_cols = ['stop_name', 'route_short_name', 'direction_id']
df['headway_s'] = df.groupby(grp_cols)['clean_time'].diff().dt.total_seconds()

# 4. Filter NaNs ONLY
# We MUST drop NaNs because the first bus of a line has no "headway" (mathematically undefined).
# We DO NOT drop large or small values anymore.
df = df.dropna(subset=['headway_s'])

# 5. Units
df['headway_min'] = df['headway_s'] / 60.0
df['hour'] = df['clean_time'].dt.hour
df['minute'] = df['clean_time'].dt.minute
df['hour_float'] = df['hour'] + (df['minute'] / 60.0)

# Night handling
df.loc[df['hour'] < 4, 'hour_float'] += 24

# =============================================================================
#  SECTION 4: EXECUTION
# =============================================================================

all_lines = sorted(df['route_short_name'].astype(str).unique())

for TARGET_LINE in all_lines:
    act_line = df[df['route_short_name'] == TARGET_LINE].copy()
    
    if act_line.empty: continue
    
    # Check route type
    r_types = act_line['route_type'].astype(str).values
    
    if '1' in r_types:
        act_line['Time_Group'] = act_line['hour_float'].apply(get_metro_profile)
        classify_and_plot(act_line, TARGET_LINE, "Metro")
    elif '3' in r_types or '0' in r_types:
        act_line['Time_Group'] = act_line['hour_float'].apply(get_surface_profile)
        classify_and_plot(act_line, TARGET_LINE, "Surface")
