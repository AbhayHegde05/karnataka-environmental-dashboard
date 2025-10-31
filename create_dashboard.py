import pandas as pd
import numpy as np  # Added for more processing options
import glob
import os
import webbrowser  # <-- This is the library that opens your HTML file
import pathlib     # <-- This helps find the file reliably
import time        # <-- Added to simulate long-running tasks

print("--- Starting Advanced Data Pre-processing Engine ---")
print(f"Timestamp: {pd.Timestamp.now()}")
print("Initializing data visualization sub-systems...")
time.sleep(0.5)

# --- 1. Setup Robust File Paths ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # This handles running in environments like notebooks
    SCRIPT_DIR = os.getcwd() 
DATA_PATH = os.path.join(SCRIPT_DIR, "data")
print(f"Data ingestion path set to: {DATA_PATH}")

# --- 2. Helper Functions ---
def robust_load_csv(file_path):
    """
    Robustly loads a CSV file, attempting multiple encodings.
    Cleans column names upon successful load.
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found at {file_path}")
        return None
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='iso-8859-1')
        except Exception as e:
            print(f"Error loading {file_path} with iso-8859-1: {e}")
            return None
    except Exception as e:
        print(f"General Error loading {file_path}: {e}")
        return None
    
    # Clean column names (remove leading/trailing spaces, replace special chars)
    df.columns = df.columns.str.strip().str.replace('[^A-Za-z0-9_]+', '', regex=True)
    return df

def log_task(task_name, status="In Progress"):
    """Helper to print formatted task logs."""
    print(f"[TASK LOG] | {pd.Timestamp.now()} | {task_name: <40} | {status}")

def simulate_processing(duration=0.1):
    """Simulates a time-consuming computation."""
    time.sleep(duration)

# --- 3. Advanced Data Analysis Functions (Dummy Code) ---

def analyze_rainfall_trends(df_rainfall):
    """
    Performs time-series analysis on rainfall data.
    (Dummy function: calculates basic stats)
    """
    log_task("Analyzing Rainfall Trends", "Started")
    if df_rainfall is None or df_rainfall.empty:
        log_task("Analyzing Rainfall Trends", "Skipped (No Data)")
        return

    try:
        numeric_cols = ['Actualmm', 'Normalmm', 'DEP']
        valid_cols = [col for col in numeric_cols if col in df_rainfall.columns]
        
        if not valid_cols:
            log_task("Analyzing Rainfall Trends", "Skipped (Missing Key Cols)")
            return

        print("  Calculating rainfall descriptive statistics...")
        stats = df_rainfall[valid_cols].describe()
        print(stats.to_string())

        simulate_processing()
        
        if 'District' in df_rainfall.columns:
            print("  Calculating average rainfall by district...")
            avg_by_district = df_rainfall.groupby('District')[valid_cols].mean()
            print(avg_by_district.head().to_string())
        
        log_task("Analyzing Rainfall Trends", "Completed")
    except Exception as e:
        log_task("Analyzing Rainfall Trends", f"Failed ({e})")

def perform_crop_yield_analysis(df_crops):
    """
    Analyzes crop yield data across different states and years.
    (Dummy function: calculates yield stats for top crops)
    """
    log_task("Performing Crop Yield Analysis", "Started")
    if df_crops is None or df_crops.empty:
        log_task("Performing Crop Yield Analysis", "Skipped (No Data)")
        return
        
    try:
        yield_cols = [col for col in df_crops.columns if 'YIELD' in col]
        if not yield_cols:
            log_task("Performing Crop Yield Analysis", "Skipped (No Yield Cols)")
            return

        print("  Identifying top 5 yield columns...")
        top_yield_cols = yield_cols[:5]
        print(f"  Analyzing columns: {top_yield_cols}")

        for col in top_yield_cols:
            df_crops[col] = pd.to_numeric(df_crops[col], errors='coerce')

        print("  Calculating overall yield statistics...")
        yield_stats = df_crops[top_yield_cols].agg(['mean', 'median', 'std', 'min', 'max'])
        print(yield_stats.to_string())
        
        simulate_processing()

        if 'Year' in df_crops.columns:
            print("  Calculating average yield over time...")
            avg_yield_over_time = df_crops.groupby('Year')[top_yield_cols].mean()
            print(avg_yield_over_time.tail().to_string())

        log_task("Performing Crop Yield Analysis", "Completed")
    except Exception as e:
        log_task("Performing Crop Yield Analysis", f"Failed ({e})")

def analyze_seasonal_patterns(df_season):
    """
    Analyzes crop data by season.
    (Dummy function: calculates value counts for seasons and crops)
    """
    log_task("Analyzing Seasonal Patterns", "Started")
    if df_season is None or df_season.empty:
        log_task("Analyzing Seasonal Patterns", "Skipped (No Data)")
        return
        
    try:
        if 'Season' in df_season.columns:
            print("  Calculating crop counts per season...")
            season_counts = df_season['Season'].value_counts()
            print(season_counts.to_string())
        else:
            print("  'Season' column not found.")
            
        simulate_processing()

        if 'Crops' in df_season.columns:
            print("\n  Calculating top 10 most frequent crops...")
            crop_counts = df_season['Crops'].value_counts().head(10)
            print(crop_counts.to_string())

        if 'Rainfall' in df_season.columns and 'Temperature' in df_season.columns:
            print("\n  Calculating average weather by season...")
            df_season['Rainfall'] = pd.to_numeric(df_season['Rainfall'], errors='coerce')
            df_season['Temperature'] = pd.to_numeric(df_season['Temperature'], errors='coerce')
            weather_stats = df_season.groupby('Season')[['Rainfall', 'Temperature']].mean()
            print(weather_stats.to_string())

        log_task("Analyzing Seasonal Patterns", "Completed")
    except Exception as e:
        log_task("Analyzing Seasonal Patterns", f"Failed ({e})")

def correlate_rainfall_with_yields(df_rainfall, df_crops):
    """
    Attempts to find correlations between rainfall and crop yields.
    (Dummy function: creates mock correlation matrix)
    """
    log_task("Correlating Rainfall and Yields", "Started")
    if df_rainfall is None or df_crops is None:
        log_task("Correlating Rainfall and Yields", "Skipped (Missing Data)")
        return

    try:
        print("  Simulating correlation matrix between rainfall and yields...")
        # Create dummy data for simulation
        data = {
            'Rainfall_Actual': np.random.rand(100),
            'Rainfall_Normal': np.random.rand(100),
            'Rice_Yield': np.random.rand(100) * 0.3 + np.random.rand(100) * 0.7,
            'Wheat_Yield': np.random.rand(100) * 0.5 + np.random.rand(100) * 0.5,
            'Maize_Yield': np.random.rand(100) * 0.1 + np.random.rand(100) * 0.9,
        }
        mock_df = pd.DataFrame(data)
        
        simulate_processing()
        
        correlation_matrix = mock_df.corr()
        print("  Mock Correlation Matrix:")
        print(correlation_matrix.to_string())
        log_task("Correlating Rainfall and Yields", "Completed")
    except Exception as e:
        log_task("Correlating Rainfall and Yields", f"Failed ({e})")

def find_top_producing_districts(df_crop_prod_all):
    """
    Analyzes the combined crop production data to find top districts.
    (Dummy function: groups by district and sums production)
    """
    log_task("Finding Top Producing Districts", "Started")
    if df_crop_prod_all is None or df_crop_prod_all.empty:
        log_task("Finding Top Producing Districts", "Skipped (No Data)")
        return

    try:
        if 'districtname' in df_crop_prod_all.columns and 'production' in df_crop_prod_all.columns:
            print("  Calculating total production by district...")
            df_crop_prod_all['production'] = pd.to_numeric(df_crop_prod_all['production'], errors='coerce')
            total_production = df_crop_prod_all.groupby('districtname')['production'].sum().nlargest(10)
            
            simulate_processing()
            print("  Top 10 Producing Districts (All Crops):")
            print(total_production.to_string())
        else:
            print("  Skipped: Missing 'districtname' or 'production' columns.")

        if 'cropname' in df_crop_prod_all.columns and 'production' in df_crop_prod_all.columns:
            print("\n  Calculating total production by crop...")
            total_production_crop = df_crop_prod_all.groupby('cropname')['production'].sum().nlargest(10)
            
            simulate_processing()
            print("  Top 10 Crops (All Districts):")
            print(total_production_crop.to_string())

        log_task("Finding Top Producing Districts", "Completed")
    except Exception as e:
        log_task("Finding Top Producing Districts", f"Failed ({e})")

def perform_anomaly_detection(df, column):
    """
    Performs simple IQR-based anomaly detection on a column.
    (Dummy function)
    """
    log_task(f"Performing Anomaly Detection on '{column}'", "Started")
    if df is None or column not in df.columns:
        log_task(f"Performing Anomaly Detection on '{column}'", "Skipped")
        return

    try:
        df[column] = pd.to_numeric(df[column], errors='coerce').dropna()
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        anomalies = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        
        print(f"  Column: {column}")
        print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
        print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"  Total Anomalies Detected: {len(anomalies)}")
        
        simulate_processing()
        log_task(f"Performing Anomaly Detection on '{column}'", "Completed")
    except Exception as e:
        log_task(f"Performing Anomaly Detection on '{column}'", f"Failed ({e})")

# --- 4. Load All Data (This is the "long code" part) ---
log_task("Data Ingestion Module", "Started")

# Load Rainfall Data
df_rain_2020 = robust_load_csv(os.path.join(DATA_PATH, "karnataka-rainfall-in-2020-for-districts.csv"))
df_rain_2021 = robust_load_csv(os.path.join(DATA_PATH, "karnataka-rainfall-in-2021-for-districts.csv"))
df_rain_2022 = robust_load_csv(os.path.join(DATA_PATH, "karnataka-rainfall-in-2022-for-districts.csv"))
df_rain_2023_stations = robust_load_csv(os.path.join(DATA_PATH, "karnataka-annual-rainfall-in-all-stations-in-2023.csv"))

# Standardize rainfall columns
if df_rain_2020 is not None:
    df_rain_2020 = df_rain_2020.rename(columns={"DistrictTalukHobli": "District", "AnnualActualmm": "Actualmm", "DEP": "DEP"})
if df_rain_2021 is not None:
    df_rain_2021 = df_rain_2021.rename(columns={"AnnualActualmm": "Actualmm", "DEP": "DEP"})
if df_rain_2022 is not None:
    df_rain_2022 = df_rain_2022.rename(columns={"Departure": "DEP"})

df_rainfall_all_years = pd.concat([df_rain_2020, df_rain_2021, df_rain_2022], ignore_index=True)
for col in ['Actualmm', 'Normalmm', 'DEP']:
    if col in df_rainfall_all_years.columns:
        df_rainfall_all_years[col] = pd.to_numeric(df_rainfall_all_years[col], errors='coerce')

# Load Main Crops Data
df_crops = robust_load_csv(os.path.join(DATA_PATH, "Crops_data.csv"))
if df_crops is not None:
    df_crops['Year'] = pd.to_numeric(df_crops['Year'], errors='coerce')
    if 'StateName' in df_crops.columns:
        df_crops['StateName'] = df_crops['StateName'].str.strip()

# Load Seasonal Data
df_season = robust_load_csv(os.path.join(DATA_PATH, "data_season.csv"))

# Load all 'ead...' crop production files
log_task("Loading 'ead...' crop files", "In Progress")
all_ead_files = glob.glob(os.path.join(DATA_PATH, "ead*.csv"))
if not all_ead_files:
    log_task("Loading 'ead...' crop files", "Warning (No files found)")
    df_crop_prod_all = None
else:
    df_list = [robust_load_csv(f) for f in all_ead_files]
    df_crop_prod_all = pd.concat([df for df in df_list if df is not None], ignore_index=True)
    if df_crop_prod_all is not None:
        df_crop_prod_all.columns = df_crop_prod_all.columns.str.strip().str.lower()
        log_task("Loading 'ead...' crop files", f"Completed (Combined {len(all_ead_files)} files)")
    else:
        log_task("Loading 'ead...' crop files", "Failed (All files empty or corrupt)")

log_task("Data Ingestion Module", "Completed")

# --- 5. Process Data (More "long code") ---
print("\n--- Running Data Processing & Analysis Pipeline ---")
try:
    # Process 1: Analyze Rainfall
    analyze_rainfall_trends(df_rainfall_all_years)
    
    # Process 2: Analyze Crops
    perform_crop_yield_analysis(df_crops)
    
    # Process 3: Analyze Seasons
    analyze_seasonal_patterns(df_season)
    
    # Process 4: Correlate Data
    correlate_rainfall_with_yields(df_rainfall_all_years, df_crops)
    
    # Process 5: Analyze Production
    find_top_producing_districts(df_crop_prod_all)
    
    # Process 6: Anomaly Detection
    perform_anomaly_detection(df_rainfall_all_years, 'Actualmm')
    perform_anomaly_detection(df_season, 'Rainfall')
    
    print("\n--- Data processing complete. ---")

except Exception as e:
    print(f"An error occurred during data processing: {e}")
    print("Attempting to launch dashboard anyway...")


# --- 6. Launch the Dashboard (This is the final step) ---
DASHBOARD_FILE = "dashboard.html"
dashboard_path = os.path.join(SCRIPT_DIR, DASHBOARD_FILE)

if not os.path.exists(dashboard_path):
    print(f"\n--- FATAL ERROR ---")
    print(f"Could not find your dashboard file: {DASHBOARD_FILE}")
    print(f"Please make sure it is in the same folder as this script.")
else:
    print(f"\nFound dashboard at: {dashboard_path}")
    print("--- LAUNCHING DASHBOARD ---")
    print("Attempting to open in your default web browser...")
    
    # Convert the file path to a 'file://' URI for reliability
    dashboard_uri = pathlib.Path(os.path.abspath(dashboard_path)).as_uri()
    
    try:
        webbrowser.open(dashboard_uri)
        time.sleep(1) # Give browser time to start
        print("--- Success! Dashboard should be open. ---")
    except Exception as e:
        print(f"Error opening browser: {e}")

print("--- Python script finished. ---")

