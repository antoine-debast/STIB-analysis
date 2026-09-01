# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="talk")

data_in = "data/fusion result/"
# Updated output directory name to reflect broader scope
data_out = "data/analysis_results/network_schedule_groups/" 
Path(data_out).mkdir(parents=True, exist_ok=True)

# Date range covering the project scope
sdate_debut = "2025-09-02"
sdate_fin = "2025-09-02"
date_range = pd.date_range(start=sdate_debut, end=sdate_fin)

# --- CONSTANTS ---
# REMOVED EXCLUSION: We now analyze ALL lines (Metro + Surface)
# EXCLUDED_LINES = ['1', '2', '5', '6'] 

# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================

def calculate_waiting_time(headways):
    """
    Computes Average Waiting Time (AWT) or Scheduled Waiting Time (SWT).
    Formula: Sum(h^2) / (2 * Sum(h))
    """
    if len(headways) < 2 or headways.sum() == 0:
        return np.nan
    
    # Convert timedelta to minutes safely
    if pd.api.types.is_timedelta64_dtype(headways):
        h_min = headways.dt.total_seconds() / 60
    else:
        h_min = headways

    h_min = h_min[h_min > 0] 
    
    if len(h_min) == 0:
        return np.nan
        
    return (h_min ** 2).sum() / (2 * h_min.sum())

# REMOVED: get_surface_time_group function
# We now rely on the pre-calculated 'schedule_id' and 'Time_Group_Desc' in the PKL.

# =============================================================================
# 2. MAIN PROCESSING LOOP
# =============================================================================

all_stats = []

for single_date in date_range:
    sdate = str(single_date.date())
    # UPDATED FILE NAME: Looking for the file with schedule info
    file_path = f"{data_in}Fused_GTFS_VehiclePositions({sdate})_with_schedule.pkl"
    
    try:
        print(f"--- Processing {sdate} ---")
        if not Path(file_path).exists():
            print(f"   File not found: {file_path}")
            # Fallback check for original file if the _with_schedule one isn't there
            # But strictly per instruction, we need the one with schedule groups.
            continue

        df = pd.read_pickle(file_path)
        
        # --- A. NO LINE EXCLUSION ---
        # We process everything.
        initial_len = len(df)
        
        # Check if required columns exist
        if 'schedule_id' not in df.columns or 'Time_Group_Desc' not in df.columns:
            print(f"   Skipping {sdate}: Missing 'schedule_id' or 'Time_Group_Desc' columns.")
            continue

        # --- B. FIX DATETYPES ---
        # Ensure datetimes are correct
        if 'departure_datetime' in df.columns:
             df['departure_datetime'] = pd.to_datetime(df['departure_datetime'], errors='coerce')
        if 'timestamp_dt' in df.columns:
             df['timestamp_dt'] = pd.to_datetime(df['timestamp_dt'], errors='coerce')
        elif 'clean_time' in df.columns:
             # Use clean_time if available from previous script
             df['timestamp_dt'] = df['clean_time']

        # --- C. PREPARE DATA ---
        # Filter rows with valid times and valid schedule group
        mask_valid = (df['departure_datetime'].notna()) & \
                     (df['timestamp_dt'].notna()) & \
                     (df['schedule_id'].notna())
        
        df_clean = df[mask_valid].copy()
        
        if df_clean.empty:
            print("   No valid data after cleaning.")
            continue

        # --- D. USE SCHEDULE GROUPS ---
        # Instead of calculating time_group on the fly, we use the columns from the PKL
        # Renaming for consistency with downstream aggregation code
        df_clean['time_group'] = df_clean['Time_Group_Desc'] # Text description
        df_clean['group_id'] = df_clean['schedule_id']       # Unique ID

        # --- E. CALCULATE HEADWAYS ---
        # 1. Sort
        df_clean = df_clean.sort_values(by=['stop_id', 'route_short_name', 'direction_id', 'departure_datetime'])
        grp_keys = ['stop_id', 'route_short_name', 'direction_id']
        
        # 2. Diff
        df_clean['sched_headway_calc'] = df_clean.groupby(grp_keys)['departure_datetime'].diff()
        df_clean['actual_headway_calc'] = df_clean.groupby(grp_keys)['timestamp_dt'].diff()
        
        # 3. Drop only First Bus of day
        df_calc = df_clean.dropna(subset=['sched_headway_calc', 'actual_headway_calc']).copy()

        # --- F. AGGREGATE ---
        # Group by Route, Direction, AND the unique Schedule ID 
        # (Using ID ensures we don't mix "Morning Peak" of Line 1 with "Morning Peak" of Line 2 if they differ)
        daily_stats = df_calc.groupby(['route_short_name', 'direction_id', 'group_id', 'time_group']).apply(
            lambda x: pd.Series({
                'SWT': calculate_waiting_time(x['sched_headway_calc']),
                'AWT': calculate_waiting_time(x['actual_headway_calc']),
                'num_obs': len(x),
                # Optional: Keep the scheduled headway from the file for reference
                'Target_Headway': x['scheduled_headway'].mean() if 'scheduled_headway' in x.columns else np.nan
            })
        ).reset_index()
        
        daily_stats['date'] = sdate
        all_stats.append(daily_stats)

    except Exception as e:
        print(f"   Error processing {sdate}: {e}")

# =============================================================================
# 3. FINAL CALCULATION & EXPORTS
# =============================================================================

print("\n--- Aggregating Final Network Results ---")
if not all_stats:
    print("No data processed.")
else:
    full_df = pd.concat(all_stats, ignore_index=True)
    full_df['EWT'] = full_df['AWT'] - full_df['SWT']
    
    # Sort for consistency
    full_df.sort_values(by=['route_short_name', 'date', 'group_id'], inplace=True)
    
    # --- EXPORT 1: CSV ---
    csv_path = f"{data_out}Network_EWT_by_ScheduleGroup.csv"
    full_df.to_csv(csv_path, index=False)
    print(f"1. Saved CSV to: {csv_path}")
    
    # --- EXPORT 2: PKL ---
    pkl_path = f"{data_out}Network_EWT_by_ScheduleGroup.pkl"
    full_df.to_pickle(pkl_path)
    print(f"2. Saved PKL to: {pkl_path}")
    
    # --- EXPORT 3: TXT ---
    txt_path = f"{data_out}Network_EWT_by_ScheduleGroup.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_df.to_string(index=False)) 
    print(f"3. Saved TXT to: {txt_path}")
    
    # Summary Report
    summary = full_df.groupby(['route_short_name', 'time_group'])[['SWT', 'AWT', 'EWT']].mean().reset_index()
    summary.to_csv(f"{data_out}Network_EWT_Summary_Report.csv", index=False)

# =============================================================================
# 4. VISUALIZATION
# =============================================================================

    if not full_df.empty:
        # Filter for noise
        viz_data = full_df[full_df['num_obs'] > 10].copy()
        
        # Determine order dynamically based on prevalence or a standard logic
        # Since 'time_group' now comes from your schedule file, we might have many unique strings.
        # We'll sort them alphabetically or by frequency for the plot.
        # Or if they are consistent (e.g. "Morning Peak"), we can rely on that.
        
        # Get top 10 most common time groups for cleaner plotting if there are too many
        top_groups = viz_data['time_group'].value_counts().index[:15].tolist()
        viz_data_filtered = viz_data[viz_data['time_group'].isin(top_groups)]
        
        plt.figure(figsize=(16, 8))
        sns.boxplot(x='time_group', y='EWT', data=viz_data_filtered, palette="viridis", showfliers=False)
        
        plt.title('Network Analysis: Excess Waiting Time (EWT) by Schedule Group', fontsize=16)
        plt.ylabel('EWT (Minutes)', fontsize=12)
        plt.xlabel('Operating Period', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)
        plt.axhline(0, color='black', linewidth=1)
        
        plt.tight_layout()
        plt.savefig(f"{data_out}Network_EWT_Analysis.png")
        plt.show()
        print("Network Analysis Graph Generated.")