import xarray as xr
import pandas as pd
import numpy as np

# Load the profile file
ds = xr.open_dataset("nodc_D1901730_388.nc")

# Extract profile-level variables
lat = ds["LATITUDE"].values
lon = ds["LONGITUDE"].values
juld = ds["JULD"].values  # sometimes days since 1950, sometimes already ISO strings

# Handle JULD conversion flexibly
try:
    # Case 1: numeric days since ref
    if np.issubdtype(juld.dtype, np.number):
        ref_date = pd.to_datetime(ds.attrs.get("JULD_REFERENCE", "1950-01-01"))
        dates = pd.to_datetime(juld, unit="D", origin=ref_date)
    else:
        # Case 2: ISO-like timestamps already
        dates = pd.to_datetime(juld)
except Exception as e:
    print("⚠️ Error parsing JULD, storing raw:", e)
    dates = juld  # fallback

# Extract measurement variables
pres = ds["PRES"].values
temp = ds["TEMP"].values
psal = ds["PSAL"].values

# Expand lat/lon/date to match depth levels
n_prof, n_levels = pres.shape
lat_expanded = np.repeat(lat, n_levels)
lon_expanded = np.repeat(lon, n_levels)
date_expanded = np.repeat(dates, n_levels)

# Flatten measurements
data = {
    "Date": date_expanded,
    "Latitude": lat_expanded,
    "Longitude": lon_expanded,
    "Pressure(dbar)": pres.flatten(),
    "Temperature(°C)": temp.flatten(),
    "Salinity(PSU)": psal.flatten()
}

df = pd.DataFrame(data)

# Save to TXT file
with open("table2.txt", "w") as f:
    f.write(df.to_string(index=False))

print("✅ Saved profile data with Lat/Lon/Date to table2.txt")