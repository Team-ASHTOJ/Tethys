import xarray as xr

# Open dataset
ds = xr.open_dataset("7902287_prof.nc")
print(ds)
# # 1. List all variables in the dataset
# print("Variables in the file:\n", list(ds.variables))

# # 2. Inspect the PARAMETER array (tells you what’s recorded)
# print("\nParameters recorded:\n", ds["PARAMETER"].values)

# # 3. Extract the core measurement variables
# if "PRES" in ds.variables:
#     print("\nPressure values (dbar):\n", ds["PRES"].values)

# if "TEMP" in ds.variables:
#     print("\nTemperature values (°C):\n", ds["TEMP"].values)

# if "PSAL" in ds.variables:
#     print("\nSalinity values (PSU):\n", ds["PSAL"].values)

# # 4. Check lat/long & time for profiles
# if "LATITUDE" in ds.variables and "LONGITUDE" in ds.variables:
#     print("\nProfile locations:\n", ds["LATITUDE"].values, ds["LONGITUDE"].values)

# if "JULD" in ds.variables:
#     print("\nJulian dates:\n", ds["JULD"].values)