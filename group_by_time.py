import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
INPUT_FILE = 'GLOBAL_SCHEDULE_OUTPUT.csv'
OUTPUT_IMAGE = 'network_synchronization_peaks.png'

def analyze_schedule_synchronization():
    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)
    
    # 2. Extract Change Points (Start and End hours of every block)
    # We ignore "0" and "24" if they are just service start/end, 
    # but keeping them shows total activity. 
    # For "Changes", we focus on the active day (04:00 to 26:00).
    
    starts = df['Start_Time'].str.split(':').str[0].astype(int)
    ends = df['End_Time'].str.split(':').str[0].astype(int)
    
    # Combine all boundaries
    all_boundaries = pd.concat([starts, ends])
    
    # Count frequency of each hour
    frequency = all_boundaries.value_counts().sort_index()
    
    # Filter for relevant hours (e.g., 04:00 to 24:00) to remove noise if needed
    frequency = frequency[(frequency.index >= 4) & (frequency.index <= 24)]

    # 3. Visualization
    plt.figure(figsize=(14, 7))
    sns.set_style("whitegrid")
    
    # Create Bar Plot
    colors = ['#D32F2F' if x in [7, 20] else '#1976D2' if x in [6, 10, 16, 21] else 'grey' 
              for x in frequency.index]
    
    bars = plt.bar(frequency.index, frequency.values, color=colors, alpha=0.8, width=0.7)
    
    # Annotate significant peaks
    for hour, count in frequency.items():
        if count > 25: # Threshold for labeling
            plt.text(hour, count + 2, str(int(count)), ha='center', fontweight='bold', fontsize=10)
            
    # Highlight the "Standard Time Groups" inferred from these peaks
    # We draw vertical zones to show the periods between the peaks
    # Example: Morning Peak (07-10)
    plt.axvspan(7, 10, color='red', alpha=0.1, label='Morning Peak (07-10)')
    plt.axvspan(10, 16, color='yellow', alpha=0.1, label='Standard Day (10-16)')
    plt.axvspan(16, 19, color='orange', alpha=0.1, label='Evening Peak (16-19)')
    
    # Formatting
    plt.title('Network Synchronization: Frequency of Schedule Changes by Hour', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Number of Lines Changing Schedule', fontsize=12)
    plt.xticks(range(4, 25))
    plt.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    plt.show()
    
    print(f"Analysis saved to {OUTPUT_IMAGE}")
    print("\n--- DETECTED PEAKS (SYNCHRONIZATION POINTS) ---")
    print(frequency[frequency > 30].sort_values(ascending=False))

if __name__ == "__main__":
    analyze_schedule_synchronization()