# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")
sns.set(style="white", context="talk")

# Files
INPUT_PKL_FILE = 'data/fusion result/Fused_GTFS_VehiclePositions(2025-09-02).pkl'
SCHEDULE_FILE = 'GLOBAL_SCHEDULE_OUTPUT.csv'

# Output Files
OUTPUT_PKL_FILE = 'data/fusion result/Fused_GTFS_VehiclePositions(2025-09-02)_with_schedule.pkl'
OUTPUT_CSV_FILE = 'data/fusion result/Fused_GTFS_VehiclePositions(2025-09-02)_with_schedule.csv'

TARGET_DATE = "2025-09-02"

# =============================================================================
#  SECTION 1: LOAD & PROCESS SCHEDULE
# =============================================================================

def load_schedule_lookup(csv_path, target_date):
    """
    Loads the schedule CSV, filters for the date, and builds a lookup dictionary.
    Returns: dict { (line_str, hour_int): (schedule_id, avg_headway, description) }
    """
    print(f"Loading schedule from {csv_path}...")
    df_sched = pd.read_csv(csv_path)
    
    # 1. Filter Date
    df_sched = df_sched[df_sched['Date'] == target_date].copy()
    
    # 2. Generate schedule_id if missing (using index + hash-like string)
    if 'schedule_id' not in df_sched.columns:
        # Fallback if your previous step didn't save it
        df_sched['schedule_id'] = df_sched.index.astype(str).str.zfill(7)
        
    # 3. Parse Times (Handle "HH:MM" or int)
    def parse_hour(t_val):
        try:
            # If string "07:00" -> 7
            if isinstance(t_val, str) and ':' in t_val:
                return int(t_val.split(':')[0])
            # If already int -> return int
            return int(t_val)
        except:
            return 0
            
    df_sched['start_h'] = df_sched['Start_Time'].apply(parse_hour)
    df_sched['end_h'] = df_sched['End_Time'].apply(parse_hour)
    
    # 4. Build Lookup Dictionary
    # Keys: (Line_String, Hour_Int) -> Value: (ID, Headway, Desc)
    schedule_map = {}
    
    for _, row in df_sched.iterrows():
        line = str(row['Line'])
        s_h = row['start_h']
        e_h = row['end_h']
        
        # Handle wrapping. If end is 0 (midnight), treat as 24
        if e_h == 0 and s_h > 0: e_h = 24
        # If the block is e.g. 23:00 to 01:00, we might need logic for 24+
        # But based on your file structure, it seems blocks are split (23-24, 00-01).
        
        for h in range(s_h, e_h):
            schedule_map[(line, h)] = (
                row['schedule_id'], 
                row['Avg_Headway_Min'],
                row['Description']
            )
            
    print(f"Schedule lookup built for {len(schedule_map)} line-hour combinations.")
    return schedule_map

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

def classify_and_plot(raw_data, line_name, schedule_map):
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # 1. Plot Raw Data
    ax.vlines(
        x=raw_data['hour_float'], 
        ymin=0, 
        ymax=raw_data['headway_min'], 
        color='grey', 
        alpha=0.3, 
        linewidth=1
    )

    # 2. Stats per Schedule Group
    # Group by the new schedule_id (and keep Description for labeling)
    group_stats = raw_data.groupby(['schedule_id', 'Time_Group_Desc', 'scheduled_headway'])['headway_min'].agg(['mean', 'std', 'count']).reset_index()
    group_stats['Cv'] = group_stats['std'] / group_stats['mean']
    
    print(f"\n--- PERFORMANCE REPORT: LINE {line_name} ---")
    print(f"{'Time Group':<30} | {'Sched(m)':<8} | {'Real(m)':<8} | {'Cv':<6} | {'Grade':<15}")
    print("-" * 80)

    for index, row in group_stats.iterrows():
        desc = row['Time_Group_Desc']
        sched_hw = row['scheduled_headway']
        real_mean = row['mean']
        cv = row['Cv']
        
        print(f"{desc:<30} | {sched_hw:<8.1f} | {real_mean:<8.2f} | {cv:<6.2f} | {get_los_grade(cv)}")
        
        # Draw Horizontal Lines
        group_data = raw_data[raw_data['schedule_id'] == row['schedule_id']]
        start_x = group_data['hour_float'].min()
        end_x = group_data['hour_float'].max()
        
        line_color = get_los_color(cv)
        
        # Plot Realized Mean
        ax.hlines(y=real_mean, xmin=start_x, xmax=end_x, 
                  color=line_color, linewidth=5, zorder=5)
        
        # Plot Scheduled Mean (Reference dashed line)
        ax.hlines(y=sched_hw, xmin=start_x, xmax=end_x, 
                  color='black', linewidth=2, linestyle='--', zorder=6, alpha=0.7)
        
        label_text = f"{desc}\nSched: {sched_hw:.1f}m\nReal: {real_mean:.1f}m"
        ax.text(x=start_x + 0.1, y=real_mean + 1, s=label_text, 
                color=line_color, fontweight='bold', fontsize=9, 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Formatting
    ax.set_title(f"Service Performance vs Schedule: Line {line_name}", fontsize=18, fontweight='bold', loc='left')
    ax.set_ylabel("Headway (minutes)")
    ax.set_xlabel("Time of Day")
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='Scheduled Headway'),
        Line2D([0], [0], color='#2E7D32', lw=4, label='Grade A (Cv < 0.25)'),
        Line2D([0], [0], color='#F57F17', lw=4, label='Grade B (0.25 < Cv < 0.75)'),
        Line2D([0], [0], color='#C62828', lw=4, label='Grade C (Cv > 0.75)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_xticks(range(5, 26))
    ax.set_xticklabels([f"{h:02d}:00" if h < 24 else f"{h-24:02d}:00" for h in range(5, 26)])
    ax.set_xlim(5, 25)
    
    y_limit = max(30, raw_data['headway_min'].max() * 1.1)
    ax.set_ylim(0, y_limit)
    
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# =============================================================================
#  SECTION 3: DATA PROCESSING & MAPPING
# =============================================================================

print("Loading Vehicle Position Data...")

if isinstance(INPUT_PKL_FILE, str):
    df = pd.read_pickle(INPUT_PKL_FILE)
else:
    df = INPUT_PKL_FILE.copy()

# 1. Clean Datetimes
if 'timestamp_dt' in df.columns:
    df['clean_time'] = pd.to_datetime(df['timestamp_dt'], errors='coerce')
elif 'timestamp' in df.columns:
    df['clean_time'] = pd.to_datetime(df['timestamp'], errors='coerce')

df = df.dropna(subset=['clean_time', 'route_short_name'])

# 2. Calculate Headways
df = df.sort_values(['stop_name', 'route_short_name', 'direction_id', 'clean_time'])
grp_cols = ['stop_name', 'route_short_name', 'direction_id']
df['headway_s'] = df.groupby(grp_cols)['clean_time'].diff().dt.total_seconds()
df = df.dropna(subset=['headway_s'])
df['headway_min'] = df['headway_s'] / 60.0

# 3. Time Processing
df['hour'] = df['clean_time'].dt.hour
df['minute'] = df['clean_time'].dt.minute
df['hour_float'] = df['hour'] + (df['minute'] / 60.0)

# Map night hours (0-3) to 24-27 only for sorting/plotting logic if needed
# But for schedule lookup, we usually keep raw hour (0, 1, 2)
# Adjust plot float for visual continuity
df.loc[df['hour'] < 4, 'hour_float'] += 24

# 4. LOAD SCHEDULE MAP
schedule_lookup = load_schedule_lookup(SCHEDULE_FILE, TARGET_DATE)

# 5. APPLY MAPPING (THE CORE TASK)
print("Mapping vehicles to schedule groups...")

def get_schedule_info(row):
    line = str(row['route_short_name'])
    h = row['hour']
    # Lookup: Default to None if not found
    return schedule_lookup.get((line, h), (None, np.nan, "Unscheduled"))

# Apply returns a DataFrame if we unpack immediately
schedule_data = df.apply(get_schedule_info, axis=1, result_type='expand')
df['schedule_id'] = schedule_data[0]
df['scheduled_headway'] = schedule_data[1]
df['Time_Group_Desc'] = schedule_data[2]

# Optional: Drop unscheduled points
# df = df.dropna(subset=['schedule_id'])

# =============================================================================
#  SECTION 4: SAVE & EXECUTE
# =============================================================================

print(f"Saving updated data to {OUTPUT_PKL_FILE}...")
df.to_pickle(OUTPUT_PKL_FILE)

print(f"Saving CSV export to {OUTPUT_CSV_FILE}...")
df.to_csv(OUTPUT_CSV_FILE, index=False)

# Visualization Loop
all_lines = sorted(df['route_short_name'].astype(str).unique())

for TARGET_LINE in all_lines:
    act_line = df[df['route_short_name'] == TARGET_LINE].copy()
    
    if act_line.empty: continue
    
    # Only plot if we have valid schedule data
    if act_line['schedule_id'].isnull().all():
        continue
        
    classify_and_plot(act_line, TARGET_LINE, schedule_lookup)