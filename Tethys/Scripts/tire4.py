import xarray as xr
import pandas as pd
import numpy as np

# Load file
ds = xr.open_dataset("argo_prof_files/1900121_prof.nc")

# First, let's check the structure of the dataset
print("Dataset dimensions:", ds.dims)
print("Dataset variables:", list(ds.variables.keys()))

# Extract variables
# For profile data, we typically have N_PROF (profiles) and N_LEVELS (depth levels)
# Some variables are per-profile (like lat/lon), others are per-level (like temp/pres)

# Check if these are the actual variable names in your file
if "PRES" in ds:
    pres = ds["PRES"].values
    temp = ds["TEMP"].values if "TEMP" in ds else None
    psal = ds["PSAL"].values if "PSAL" in ds else None
    
    # Quality control flags
    temp_qc = ds["TEMP_QC"].values if "TEMP_QC" in ds else None
    psal_qc = ds["PSAL_QC"].values if "PSAL_QC" in ds else None
    pres_qc = ds["PRES_QC"].values if "PRES_QC" in ds else None
    
    # Position and time (usually one per profile)
    lat = ds["LATITUDE"].values if "LATITUDE" in ds else None
    lon = ds["LONGITUDE"].values if "LONGITUDE" in ds else None
    juld = ds["JULD"].values if "JULD" in ds else None
    
    # Create lists to store flattened data
    data_list = []
    
    # Assuming 2D structure: (n_prof, n_levels)
    if len(pres.shape) == 2:
        n_prof, n_levels = pres.shape
        
        for i in range(n_prof):
            for j in range(n_levels):
                # Skip if pressure is NaN or fill value
                if not np.isnan(pres[i, j]) and pres[i, j] != 99999:
                    row = {
                        "profile_idx": i,
                        "level_idx": j,
                        "latitude": lat[i] if lat is not None and len(lat.shape) == 1 else lat[i, j] if lat is not None else None,
                        "longitude": lon[i] if lon is not None and len(lon.shape) == 1 else lon[i, j] if lon is not None else None,
                        "pressure": pres[i, j],
                        "temperature": temp[i, j] if temp is not None else None,
                        "salinity": psal[i, j] if psal is not None else None,
                        "temp_qc": temp_qc[i, j] if temp_qc is not None else None,
                        "psal_qc": psal_qc[i, j] if psal_qc is not None else None,
                        "pres_qc": pres_qc[i, j] if pres_qc is not None else None,
                        "juld": juld[i] if juld is not None and len(juld.shape) == 1 else juld[i, j] if juld is not None else None
                    }
                    data_list.append(row)
    
    # Create DataFrame from list
    df = pd.DataFrame(data_list)
    
    # Save as Parquet
    df.to_parquet("argo_profile.parquet", index=False)
    print(f"✅ Saved argo_profile.parquet with {len(df)} records")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
else:
    print("❌ PRES variable not found in dataset")
    print("Available variables:", list(ds.variables.keys()))