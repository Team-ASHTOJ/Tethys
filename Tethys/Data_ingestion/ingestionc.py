# Gets all of the files in the directory and combines them into a large .parquet file, which is then pushed into our Postgre
import os
import glob
import xarray as xr
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from tqdm import tqdm # A library for progress bars, run: pip install tqdm

# --- 1. CONFIGURATION ---
# Set your paths and database connection details here
NC_FILES_DIR = "argo_prof_files/"
OUTPUT_PARQUET_FILE = "argo_all_profiles_combined.parquet"
DB_CONNECTION_STRING = "postgresql://shaikmohammedomar@localhost:5432/argo"
TABLE_NAME = "profiles"

# --- 2. FUNCTION TO PROCESS A SINGLE .NC FILE ---
# This is your working code, refactored into a reusable function.
def process_nc_to_dataframe(file_path):
    """
    Opens a single Argo NetCDF file, extracts profile data,
    and returns it as a pandas DataFrame.
    Returns None if the file is invalid or has no pressure data.
    """
    try:
        with xr.open_dataset(file_path) as ds:
            # Check for the essential variable
            if "PRES" not in ds:
                print(f"⚠️ Skipping {os.path.basename(file_path)}: No 'PRES' variable.")
                return None

            # Extract variables
            pres = ds["PRES"].values
            temp = ds["TEMP"].values if "TEMP" in ds else None
            psal = ds["PSAL"].values if "PSAL" in ds else None
            temp_qc = ds["TEMP_QC"].values if "TEMP_QC" in ds else None
            psal_qc = ds["PSAL_QC"].values if "PSAL_QC" in ds else None
            pres_qc = ds["PRES_QC"].values if "PRES_QC" in ds else None
            lat = ds["LATITUDE"].values if "LATITUDE" in ds else None
            lon = ds["LONGITUDE"].values if "LONGITUDE" in ds else None
            juld = ds["JULD"].values if "JULD" in ds else None
            
            # Use the WMO number from the file name as a unique float identifier
            platform_number = os.path.basename(file_path).split('_')[0]

            data_list = []
            if len(pres.shape) == 2:
                n_prof, n_levels = pres.shape
                for i in range(n_prof):
                    for j in range(n_levels):
                        # Skip if pressure is NaN or a standard fill value
                        if not np.isnan(pres[i, j]) and pres[i, j] != 99999:
                            row = {
                                "platform_number": int(platform_number),
                                "profile_idx": i,
                                "latitude": lat[i] if lat is not None and lat.ndim == 1 else None,
                                "longitude": lon[i] if lon is not None and lon.ndim == 1 else None,
                                "juld": juld[i] if juld is not None and juld.ndim == 1 else None,
                                "pressure": pres[i, j],
                                "temperature": temp[i, j] if temp is not None else None,
                                "salinity": psal[i, j] if psal is not None else None,
                                "temp_qc": temp_qc[i, j] if temp_qc is not None else None,
                                "psal_qc": psal_qc[i, j] if psal_qc is not None else None,
                                "pres_qc": pres_qc[i, j] if pres_qc is not None else None,
                            }
                            data_list.append(row)
            
            if not data_list:
                return None
                
            return pd.DataFrame(data_list)

    except Exception as e:
        print(f"❌ Error processing {os.path.basename(file_path)}: {e}")
        return None

# --- 3. MAIN EXECUTION SCRIPT ---
if __name__ == "__main__":
    
    # --- Part A: Process all .nc files and create a combined Parquet file ---
    
    # Find all .nc files in the directory
    nc_files = glob.glob(os.path.join(NC_FILES_DIR, '*.nc'))
    print(f"Found {len(nc_files)} NetCDF files to process.")

    # List to hold all the individual DataFrames
    all_dfs = []

    # Loop through files with a progress bar
    for file_path in tqdm(nc_files, desc="Processing .nc files"):
        df = process_nc_to_dataframe(file_path)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("No data processed. Exiting.")
    else:
        # Combine all DataFrames into one
        print("\nCombining all data into a single DataFrame...")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Total records processed: {len(combined_df)}")

        # IMPORTANT: Fix data types before saving to avoid issues in PostgreSQL
        for col in ['temp_qc', 'psal_qc', 'pres_qc']:
            if col in combined_df.columns:
                # Convert bytes to string if necessary, then ensure the whole column is string type
                combined_df[col] = combined_df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x).astype(str)
        
        # Save the combined data to a single Parquet file
        print(f"Saving combined data to {OUTPUT_PARQUET_FILE}...")
        combined_df.to_parquet(OUTPUT_PARQUET_FILE, index=False)
        print("✅ Parquet file saved successfully.")

        # --- Part B: Bulk insert the combined data into PostgreSQL ---
        
        print("\nConnecting to PostgreSQL database...")
        engine = create_engine(DB_CONNECTION_STRING)

        print(f"Writing {len(combined_df)} records to table '{TABLE_NAME}'...")
        # Use if_exists='replace' to start fresh each time.
        # Use if_exists='append' if you want to add to existing data.
        # Use chunksize for memory efficiency with large datasets.
        combined_df.to_sql(
            TABLE_NAME,
            engine,
            if_exists="replace", 
            index=False,
            chunksize=10000  # Insert 10,000 rows at a time
        )
        
        print("✅ All data has been successfully pushed into PostgreSQL!")