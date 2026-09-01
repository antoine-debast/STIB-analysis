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
data_out = "data/analysis_results/metro/" # Specific output folder for Metro
Path(data_out).mkdir(parents=True, exist_ok=True)

# Date range covering the project scope
sdate_debut = "2025-08-15"
sdate_fin = "2025-09-15"
date_range = pd.date_range(start=sdate_debut, end=sdate_fin)

# --- CONSTANTS ---
METRO_LINES = ['1', '2', '5', '6'] # Specific Metro lines

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

    h_min = h_min[h_min > 0] # Remove 0-min gaps
    
    if len(h_min) == 0:
        return np.nan
        
    return (h_min ** 2).sum() / (2 * h_min.sum())

def get_metro_time_group(row):
    """
    Assigns a Time Group based on Metro definitions.
    """
    h = row['hour_float']
    
    # --- METRO LOGIC ---
    if h < 6.5: return "1. Early Service"
    elif 6.5 <= h < 9.5: return "2. AM Peak"
    elif 9.5 <= h < 16.0: return "3. Day Base"
    elif 16.0 <= h < 19.0: return "4. PM Peak"
    elif 19.0 <= h < 22.0: return "5. Evening"
    else: return "6. Late Night"

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
        
        # --- A. FILTER FOR METRO ONLY ---
        # Ensure route_short_name is string
        df['route_short_name'] = df['route_short_name'].astype(str)
        
        initial_len = len(df)
        df = df[df['route_short_name'].isin(METRO_LINES)]
        
        if df.empty:
            print(f"   Skipping (No Metro lines {METRO_LINES} found in file)")
            continue
            
        print(f"   Rows: {initial_len} -> {len(df)} (Metro Filter Applied)")

        # --- B. FIX DATETYPES ---
        df['departure_datetime'] = pd.to_datetime(df['departure_datetime'], errors='coerce')
        df['timestamp_dt'] = pd.to_datetime(df['timestamp_dt'], errors='coerce')
        
        # --- C. PREPARE DATA (No Quality Drops) ---
        mask_valid = (df['departure_datetime'].notna()) & (df['timestamp_dt'].notna())
        df_clean = df[mask_valid].copy()

        # --- D. ASSIGN TIME GROUPS ---
        df_clean['hour_float'] = df_clean['departure_datetime'].dt.hour + (df_clean['departure_datetime'].dt.minute / 60.0)
        df_clean['time_group'] = df_clean.apply(get_metro_time_group, axis=1)

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
# 3. FINAL CALCULATION & EXPORT
# =============================================================================

print("\n--- Aggregating Final Metro Results ---")
if not all_stats:
    print("No Metro data processed.")
else:
    full_df = pd.concat(all_stats, ignore_index=True)
    full_df['EWT'] = full_df['AWT'] - full_df['SWT']
    
    # Save raw data
    full_df.sort_values(by=['route_short_name', 'date', 'time_group'], inplace=True)
    full_df.to_csv(f"{data_out}Metro_EWT_by_TimeGroup.csv", index=False)
    
    # Summary
    summary = full_df.groupby(['route_short_name', 'time_group'])[['SWT', 'AWT', 'EWT']].mean().reset_index()
    summary.to_csv(f"{data_out}Metro_EWT_Summary_Report.csv", index=False)
    print(f"Saved reports to {data_out}")

# =============================================================================
# 4. VISUALIZATION
# =============================================================================

    # Filter for noise in visualization
    viz_data = full_df[full_df['num_obs'] > 10].copy()
    
    # Metro Specific Order
    time_order = [
        "1. Early Service", 
        "2. AM Peak", 
        "3. Day Base", 
        "4. PM Peak", 
        "5. Evening", 
        "6. Late Night"
    ]
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='time_group', y='EWT', data=viz_data, order=time_order, palette="magma", showfliers=False)
    
    plt.title('Metro Network: Excess Waiting Time (EWT)', fontsize=16)
    plt.ylabel('EWT (Minutes)', fontsize=12)
    plt.xlabel('Operating Period', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.axhline(0, color='black', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(f"{data_out}Metro_EWT_Analysis.png")
    plt.show()
    print("Metro Analysis Complete.")