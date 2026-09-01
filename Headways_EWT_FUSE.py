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
data_out = "data/analysis_results/surface/" 
Path(data_out).mkdir(parents=True, exist_ok=True)

# Date range covering the project scope
sdate_debut = "2025-08-15"
sdate_fin = "2025-09-15"
date_range = pd.date_range(start=sdate_debut, end=sdate_fin)

# --- CONSTANTS ---
EXCLUDED_LINES = ['1', '2', '5', '6'] # Metro lines to exclude (Surface Analysis)

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

def get_surface_time_group(row):
    """
    Assigns a Time Group based strictly on Surface definitions.
    """
    h = row['hour_float']
    
    # --- SURFACE LOGIC ONLY ---
    if h < 7.0: return "1. Morning Ramp-up"
    elif 7.0 <= h < 9.5: return "2. AM Rush"
    elif 9.5 <= h < 15.5: return "3. Day Base"
    elif 15.5 <= h < 18.5: return "4. School/PM Rush"
    elif 18.5 <= h < 21.0: return "5. Evening Slide"
    else: return "6. Night"

# =============================================================================
# 2. MAIN PROCESSING LOOP
# =============================================================================

all_stats = []

for single_date in date_range:
    sdate = str(single_date.date())
    file_path = f"{data_in}Fused_GTFS_VehiclePositions({sdate}).pkl"
    
    try:
        print(f"--- Processing {sdate} ---")
        if not Path(file_path).exists():
            print(f"   File not found: {file_path}")
            continue

        df = pd.read_pickle(file_path)
        
        # --- A. FILTER FOR SURFACE (Exclude Metro) ---
        df['route_short_name'] = df['route_short_name'].astype(str)
        
        initial_len = len(df)
        df = df[~df['route_short_name'].isin(EXCLUDED_LINES)]
        
        if df.empty:
            print(f"   Skipping (Only excluded lines found)")
            continue
            
        print(f"   Rows: {initial_len} -> {len(df)} (Surface Filter Applied)")

        # --- B. FIX DATETYPES ---
        df['departure_datetime'] = pd.to_datetime(df['departure_datetime'], errors='coerce')
        df['timestamp_dt'] = pd.to_datetime(df['timestamp_dt'], errors='coerce')
        
        # --- C. PREPARE DATA (No Quality Drops) ---
        mask_valid = (df['departure_datetime'].notna()) & (df['timestamp_dt'].notna())
        df_clean = df[mask_valid].copy()

        # --- D. ASSIGN TIME GROUPS ---
        df_clean['hour_float'] = df_clean['departure_datetime'].dt.hour + (df_clean['departure_datetime'].dt.minute / 60.0)
        df_clean['time_group'] = df_clean.apply(get_surface_time_group, axis=1)

        # --- E. CALCULATE HEADWAYS ---
        # 1. Sort
        df_clean = df_clean.sort_values(by=['stop_id', 'route_short_name', 'direction_id', 'departure_datetime'])
        grp_keys = ['stop_id', 'route_short_name', 'direction_id']
        
        # 2. Diff
        df_clean['sched_headway'] = df_clean.groupby(grp_keys)['departure_datetime'].diff()
        df_clean['actual_headway'] = df_clean.groupby(grp_keys)['timestamp_dt'].diff()
        
        # 3. Drop only First Bus of day (Mathematical necessity)
        df_calc = df_clean.dropna(subset=['sched_headway', 'actual_headway']).copy()

        # --- F. AGGREGATE ---
        daily_stats = df_calc.groupby(['route_short_name', 'direction_id', 'time_group']).apply(
            lambda x: pd.Series({
                'SWT': calculate_waiting_time(x['sched_headway']),
                'AWT': calculate_waiting_time(x['actual_headway']),
                'num_obs': len(x)
            })
        ).reset_index()
        
        daily_stats['date'] = sdate
        all_stats.append(daily_stats)

    except Exception as e:
        print(f"   Error processing {sdate}: {e}")

# =============================================================================
# 3. FINAL CALCULATION & EXPORTS
# =============================================================================

print("\n--- Aggregating Final Surface Results ---")
if not all_stats:
    print("No Surface data processed.")
else:
    full_df = pd.concat(all_stats, ignore_index=True)
    full_df['EWT'] = full_df['AWT'] - full_df['SWT']
    
    # Sort for consistency
    full_df.sort_values(by=['route_short_name', 'date', 'time_group'], inplace=True)
    
    # --- EXPORT 1: CSV (Standard) ---
    csv_path = f"{data_out}Surface_EWT_by_TimeGroup.csv"
    full_df.to_csv(csv_path, index=False)
    print(f"1. Saved CSV to: {csv_path}")
    
    # --- EXPORT 2: PKL (Preserves Datatypes) ---
    pkl_path = f"{data_out}Surface_EWT_by_TimeGroup.pkl"
    full_df.to_pickle(pkl_path)
    print(f"2. Saved PKL to: {pkl_path}")
    
    # --- EXPORT 3: TXT (Human Readable) ---
    txt_path = f"{data_out}Surface_EWT_by_TimeGroup.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        # to_string() creates a pretty fixed-width table
        f.write(full_df.to_string(index=False)) 
    print(f"3. Saved TXT to: {txt_path}")
    
    # Summary Report (CSV only usually needed for summary)
    summary = full_df.groupby(['route_short_name', 'time_group'])[['SWT', 'AWT', 'EWT']].mean().reset_index()
    summary.to_csv(f"{data_out}Surface_EWT_Summary_Report.csv", index=False)

# =============================================================================
# 4. VISUALIZATION
# =============================================================================

    # Filter for noise in visualization
    viz_data = full_df[full_df['num_obs'] > 10].copy()
    
    # Surface Specific Order
    time_order = [
        "1. Morning Ramp-up",
        "2. AM Rush",
        "3. Day Base", 
        "4. School/PM Rush",
        "5. Evening Slide",
        "6. Night"
    ]
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='time_group', y='EWT', data=viz_data, order=time_order, palette="viridis", showfliers=False)
    
    plt.title('Surface Network: Excess Waiting Time (EWT)', fontsize=16)
    plt.ylabel('EWT (Minutes)', fontsize=12)
    plt.xlabel('Operating Period', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.axhline(0, color='black', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(f"{data_out}Surface_EWT_Analysis.png")
    plt.show()
    print("Surface Analysis Graph Generated.")