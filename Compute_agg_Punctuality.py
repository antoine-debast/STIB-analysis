# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 17:10:25 2025

@author: Aurélien
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# --- CONFIGURATION ---

data_in = "data/fusion result/"
data_out = "data/punctuality_results/" 
Path(data_out).mkdir(parents=True, exist_ok=True)

# Date range covering the project scope
sdate_debut = "2025-08-15"
sdate_fin = "2025-09-15"
date_range = pd.date_range(start=sdate_debut, end=sdate_fin)
all_stats=[]
for single_date in date_range:
    sdate = str(single_date.date())
    file_path = f"{data_in}Fused_GTFS_VehiclePositions({sdate}).pkl"
    
    try:
        print(f"--- Processing {sdate} ---")
        if not Path(file_path).exists():
            print(f"   File not found: {file_path}")
            continue

        fused_df = pd.read_pickle(file_path)
        
        # # --- A. FILTER FOR SURFACE (Exclude Metro) ---
        # df['route_short_name'] = df['route_short_name'].astype(str)
        
        # initial_len = len(df)
        # df_surface = df[~df['route_short_name'].isin(METRO_LINES)]
        
        # --- B. FIX DATETYPES ---
        fused_df['departure_datetime'] = pd.to_datetime(fused_df['departure_datetime'])
        fused_df['timestamp_dt_mean'] = pd.to_datetime(fused_df['timestamp_dt_mean'])
        # if df.empty:
        #     print(f"   Skipping (Only excluded lines found)")
        #     continue
            
        # print(f"   Rows: {initial_len} -> {len(df)} (Surface Filter Applied)")

               
        # --- C. PREPARE DATA: Keep only the truly fused or in GTFS ---
        mask_both = (fused_df['departure_datetime'].notna()) & (fused_df['timestamp_dt_mean'].notna())
        mask_GTFS_only = (fused_df['departure_datetime'].notna()) & (fused_df['timestamp_dt_mean'].isna())
        fused_df_both=fused_df[mask_both].copy()
        fused_df_GTFS_only=fused_df[mask_GTFS_only].copy()

        # --- D. ASSIGN TIME GROUPS ---
        fused_df_both['actual_hour']= fused_df_both['timestamp_dt_mean'].dt.hour
        fused_df_both['date']= fused_df_both['timestamp_dt_mean'].dt.date
        # df_clean['hour_float'] = df_clean['departure_datetime'].dt.hour + (df_clean['departure_datetime'].dt.minute / 60.0)
        # df_clean['time_group'] = df_clean.apply(get_surface_time_group, axis=1)


        # --- F. AGGREGATE ---
        daily_stats = fused_df_both.groupby(['date','actual_hour','route_short_name', 'direction_id', 'stop_id', 'stop_name', 'stop_geo'])['delay'].agg(
            mean_delay='mean', std_delay='std',count='count'
        )
        all_stats.append(daily_stats)
    except(FileNotFoundError):
        print(sdate,"manquant. On passe")
    except Exception as e:
        print(f"   Error processing {sdate}: {e}")
full_stats = pd.concat(all_stats, ignore_index=False)
full_stats=full_stats.sort_values(by=['route_short_name', 'direction_id','date'])
# --- EXPORT 1: CSV (Standard) ---
csv_path = f"{data_out}punctuality_results.csv"
full_stats.to_csv(csv_path, index=True, sep=';')
print(f"1. Saved CSV to: {csv_path}")

# --- EXPORT 2: PKL (Preserves Datatypes) ---
pkl_path = f"{data_out}punctuality_results.pkl"
full_stats.to_pickle(pkl_path)
print(f"2. Saved PKL to: {pkl_path}")